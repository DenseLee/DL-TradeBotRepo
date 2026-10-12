"""Demo-only C1-SEQ diagnostic runner.

This archived experiment builds a causal rolling-tail signal from synchronized
probe rows. It is not a promoted strategy and cannot be used with real money.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.mt5_executor import MT5Executor
from models.trend.c1_confirmed_spree import (
    ConfirmedSpreeConfig,
    OpenLeg,
    evaluate_confirmed_add,
)
from models.trend.c1_agreement_component_audit import candidate_state
from models.trend.c1_sequence_small_target_policy_test import SIGNAL_COLUMNS, build_candidates
from models.trend.probe_confirm_expand_forward import append_jsonl, atomic_json, read_jsonl_incremental


C1_DEMO_MAGIC = 20260818
C1_TICKMILL_LIVE_MAGIC = 20260825
C1_TICKMILL_OVERLAY_MAGIC = 20260829
C1_TICKMILL_RATCHET_MAGIC = 20260908
C1_TICKMILL_NO_FLOW_MAGIC = 20260913
DEFAULT_ACCOUNT = 0
TICKMILL_LIVE_ACCOUNT = -1
TICKMILL_OVERLAY_ACCOUNT = -2
TICKMILL_NO_FLOW_ACCOUNT = -3
DEFAULT_TERMINAL = None
DEFAULT_SYNC_DIR = ROOT / "data" / "raw" / "live_probe_fbs_demo_v2" / "sync"
DEFAULT_EVENT_LOG = ROOT / "logs" / "c1_sequence_fbs_demo_events.jsonl"
DEFAULT_STATE = ROOT / "logs" / "c1_sequence_fbs_demo_state.json"
STOP_WIDTH_BPS = 100.0
TARGET_R = 0.5
MAX_HOLD_MINUTES = 120
MAX_LIVE_GAP_SECONDS = 120.0
MAX_HISTORY_GAP_MINUTES = 15.0
RISK_OVERLAY_NONE = "none"
RISK_OVERLAY_BE025_DECAY60 = "break_even_025_time_decay_60"
RISK_OVERLAY_RATCHET_050_075_100 = "ratchet_050_075_target100"
SIGNAL_PROFILE_BASELINE = "baseline_three_component"
SIGNAL_PROFILE_NO_FLOW = "no_flow_notional_price"
NO_FLOW_SIGNAL_COLUMNS = ("signed_trade_notional_60s", "fbs_return_60s")
BREAK_EVEN_ACTIVATION_R = 0.25
TIME_DECAY_MINUTES = 60
EXIT_RETRY_SECONDS = 30.0
MAX_BROKER_CLOCK_OFFSET_SECONDS = 14 * 60 * 60
CONFIRMED_SPREE_CONFIG = ConfirmedSpreeConfig(
    max_positions=4,
    minimum_open_r=0.25,
    minimum_minutes_between_adds=5.0,
)


def live_magic(expected_account: int, risk_overlay: str) -> int:
    if int(expected_account) == TICKMILL_LIVE_ACCOUNT and risk_overlay == RISK_OVERLAY_NONE:
        return C1_TICKMILL_LIVE_MAGIC
    if int(expected_account) == TICKMILL_OVERLAY_ACCOUNT and risk_overlay == RISK_OVERLAY_BE025_DECAY60:
        return C1_TICKMILL_OVERLAY_MAGIC
    if int(expected_account) == TICKMILL_OVERLAY_ACCOUNT and risk_overlay == RISK_OVERLAY_RATCHET_050_075_100:
        return C1_TICKMILL_RATCHET_MAGIC
    if int(expected_account) == TICKMILL_NO_FLOW_ACCOUNT and risk_overlay == RISK_OVERLAY_NONE:
        return C1_TICKMILL_NO_FLOW_MAGIC
    raise RuntimeError(
        f"unauthorized live account/overlay pair: account={expected_account} overlay={risk_overlay}"
    )


def finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def sanitize_history_after_gap(state: dict[str, Any]) -> dict[str, Any] | None:
    """Keep only the contiguous feature tail after the latest severe gap."""
    parsed: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for row in state.get("features") or []:
        timestamp = pd.to_datetime(row.get("feature_timestamp"), utc=True, errors="coerce")
        if not pd.isna(timestamp):
            parsed.append((pd.Timestamp(timestamp), row))
    parsed.sort(key=lambda item: item[0])
    reset_index = 0
    gap_start: pd.Timestamp | None = None
    gap_end: pd.Timestamp | None = None
    for index in range(1, len(parsed)):
        gap_minutes = (parsed[index][0] - parsed[index - 1][0]).total_seconds() / 60.0
        if gap_minutes > MAX_HISTORY_GAP_MINUTES:
            reset_index = index
            gap_start = parsed[index - 1][0]
            gap_end = parsed[index][0]
    original_count = len(parsed)
    if reset_index:
        parsed = parsed[reset_index:]
        state["features"] = [row for _, row in parsed]
        state["minute_buckets"] = {
            key: bucket
            for key, bucket in (state.get("minute_buckets") or {}).items()
            if pd.Timestamp(key) >= gap_end.floor("min")
        }
        state["gap_reset_count"] = int(state.get("gap_reset_count", 0)) + 1
    latest_receive: pd.Timestamp | None = None
    for _, row in parsed:
        receive = pd.to_datetime(row.get("feature_max_receive_time"), utc=True, errors="coerce")
        if not pd.isna(receive) and (latest_receive is None or receive > latest_receive):
            latest_receive = pd.Timestamp(receive)
    for bucket in (state.get("minute_buckets") or {}).values():
        receive = pd.to_datetime(bucket.get("last_receive_time"), utc=True, errors="coerce")
        if not pd.isna(receive) and (latest_receive is None or receive > latest_receive):
            latest_receive = pd.Timestamp(receive)
    if latest_receive is not None:
        state["last_valid_receive_time"] = latest_receive.isoformat()
    if not reset_index or gap_start is None or gap_end is None:
        return None
    return {
        "event": "feature_history_reset_after_gap",
        "reason": "startup_history_gap",
        "gap_start": gap_start.isoformat(),
        "gap_end": gap_end.isoformat(),
        "gap_seconds": (gap_end - gap_start).total_seconds(),
        "features_before": original_count,
        "features_after": len(parsed),
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "offsets": {},
            "minute_buckets": {},
            "features": [],
            "last_candidate_time": None,
            "owned_ticket": None,
            "owned_entry_time": None,
            "owned_entry_time_source": None,
            "owned_entry_price": None,
            "owned_risk_distance": None,
            "owned_peak_r": 0.0,
            "break_even_applied": False,
            "ratchet_floor_r": None,
            "last_ratchet_attempt_at": None,
            "last_valid_receive_time": None,
            "gap_reset_count": 0,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    reset_event = sanitize_history_after_gap(state)
    if reset_event is not None:
        state.setdefault("pending_gap_events", []).append(reset_event)
    return state


def update_minute_buckets(state: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = state.setdefault("minute_buckets", {})
    gap_events: list[dict[str, Any]] = []
    previous_receive = pd.to_datetime(
        state.get("last_valid_receive_time"), utc=True, errors="coerce"
    )
    for row in rows:
        if row.get("incomplete") or not row.get("has_fbs_tick"):
            continue
        receive = pd.to_datetime(row.get("local_receive_time"), utc=True, errors="coerce")
        bid = finite(row.get("fbs_bid"))
        ask = finite(row.get("fbs_ask"))
        imbalance = finite(row.get("binance_trade_imbalance_60s"))
        notional = finite(row.get("binance_trade_notional_60s"))
        divergence_bps = finite(
            row.get("broker_vs_binance_mid_divergence_bps", row.get("fbs_vs_binance_mid_divergence_bps"))
        )
        if pd.isna(receive) or bid is None or ask is None or ask < bid or imbalance is None or notional is None:
            continue
        if not pd.isna(previous_receive):
            gap_seconds = (receive - previous_receive).total_seconds()
            if gap_seconds > MAX_LIVE_GAP_SECONDS:
                features_before = len(state.get("features") or [])
                state["features"] = []
                state["minute_buckets"] = {}
                buckets = state["minute_buckets"]
                state["gap_reset_count"] = int(state.get("gap_reset_count", 0)) + 1
                gap_events.append(
                    {
                        "event": "feature_history_reset_after_gap",
                        "reason": "live_receive_gap",
                        "gap_start": pd.Timestamp(previous_receive).isoformat(),
                        "gap_end": pd.Timestamp(receive).isoformat(),
                        "gap_seconds": gap_seconds,
                        "features_before": features_before,
                        "features_after": 0,
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        if pd.isna(previous_receive) or receive > previous_receive:
            previous_receive = receive
            state["last_valid_receive_time"] = pd.Timestamp(receive).isoformat()
        minute = receive.floor("min").isoformat()
        mid = (bid + ask) / 2.0
        bucket = buckets.setdefault(
            minute,
            {
                "first_mid": mid,
                "last_mid": mid,
                "last_imbalance": imbalance,
                "last_notional": notional,
                "last_receive_time": receive.isoformat(),
                "last_venue_divergence_bps": divergence_bps,
                "last_source_after_receive_corrected": row.get("binance_source_after_receive_corrected"),
                "last_source_clock_offset_ms": finite(row.get("binance_source_clock_offset_ms")),
                "rows": 0,
            },
        )
        bucket["last_mid"] = mid
        bucket["last_imbalance"] = imbalance
        bucket["last_notional"] = notional
        bucket["last_receive_time"] = receive.isoformat()
        bucket["last_venue_divergence_bps"] = divergence_bps
        bucket["last_source_after_receive_corrected"] = row.get("binance_source_after_receive_corrected")
        bucket["last_source_clock_offset_ms"] = finite(row.get("binance_source_clock_offset_ms"))
        bucket["rows"] = int(bucket.get("rows", 0)) + 1
    return gap_events


def finalize_minutes(state: dict[str, Any], now: pd.Timestamp) -> int:
    cutoff = now.floor("min")
    buckets = state.setdefault("minute_buckets", {})
    features = state.setdefault("features", [])
    finalized = 0
    for key in sorted(list(buckets)):
        minute = pd.Timestamp(key)
        if minute >= cutoff:
            continue
        bucket = buckets.pop(key)
        first_mid = finite(bucket.get("first_mid"))
        last_mid = finite(bucket.get("last_mid"))
        imbalance = finite(bucket.get("last_imbalance"))
        notional = finite(bucket.get("last_notional"))
        if None in (first_mid, last_mid, imbalance, notional) or int(bucket.get("rows", 0)) < 20:
            continue
        features.append(
            {
                "feature_timestamp": (minute + pd.Timedelta(minutes=1)).isoformat(),
                "feature_max_receive_time": bucket["last_receive_time"],
                "agg_trade_imbalance_60s": imbalance,
                "signed_trade_notional_60s": imbalance * notional,
                "fbs_return_60s": last_mid / max(first_mid, 1e-12) - 1.0,
                "broker_vs_binance_mid_divergence_bps": finite(bucket.get("last_venue_divergence_bps")),
                "binance_source_after_receive_corrected": bucket.get("last_source_after_receive_corrected"),
                "binance_source_clock_offset_ms": finite(bucket.get("last_source_clock_offset_ms")),
            }
        )
        finalized += 1
    deduped = {row["feature_timestamp"]: row for row in features}
    state["features"] = [deduped[key] for key in sorted(deduped)][-480:]
    return finalized


def candidates_for_signal_profile(
    features: pd.DataFrame,
    reference_hours: float,
    quantile: float,
    signal_profile: str,
) -> pd.DataFrame:
    """Build candidates from one explicit, frozen component contract."""
    if signal_profile == SIGNAL_PROFILE_BASELINE:
        return build_candidates(features, reference_hours, quantile, require_t9_flat=False)
    if signal_profile == SIGNAL_PROFILE_NO_FLOW:
        if reference_hours != 6.0 or quantile != 0.10:
            raise ValueError("no-flow profile is frozen at reference_hours=6 and quantile=0.10")
        state = candidate_state(features, NO_FLOW_SIGNAL_COLUMNS, minimum_agreement=2)
        candidates = state.loc[state["new_event"] & state["direction"].ne(0)].copy()
        candidates["anomaly_score"] = np.where(
            candidates["direction"].eq(1),
            candidates["upper_score"],
            candidates["lower_score"],
        )
        return candidates.reset_index(drop=True)
    raise ValueError(f"unknown signal profile: {signal_profile}")


def latest_new_candidate(
    state: dict[str, Any],
    reference_hours: float,
    quantile: float,
    signal_profile: str = SIGNAL_PROFILE_BASELINE,
) -> dict[str, Any] | None:
    features = pd.DataFrame(state.get("features") or [])
    if len(features) < 182:
        return None
    features["feature_timestamp"] = pd.to_datetime(features["feature_timestamp"], utc=True)
    candidates = candidates_for_signal_profile(features, reference_hours, quantile, signal_profile)
    if candidates.empty:
        return None
    last_seen = pd.to_datetime(state.get("last_candidate_time"), utc=True, errors="coerce")
    if not pd.isna(last_seen):
        candidates = candidates[candidates["decision_time"] > last_seen]
    if candidates.empty:
        return None
    candidate = candidates.iloc[-1].to_dict()
    matching = features.loc[features["feature_timestamp"].eq(pd.Timestamp(candidate["decision_time"]))]
    if not matching.empty:
        source = matching.iloc[-1]
        for column in (
            "feature_max_receive_time",
            "broker_vs_binance_mid_divergence_bps",
            "binance_source_after_receive_corrected",
            "binance_source_clock_offset_ms",
        ):
            if column in source:
                candidate[column] = source[column]
    return candidate


def venue_divergence_allowed(candidate: dict[str, Any], maximum_bps: float | None) -> bool:
    """Keep the default runner behavior unless an explicitly tested gate is enabled."""
    if maximum_bps is None:
        return True
    divergence = finite(candidate.get("broker_vs_binance_mid_divergence_bps"))
    return divergence is not None and abs(divergence) <= maximum_bps


def persist_state(state_path: Path, state: dict[str, Any], event_log: Path) -> bool:
    """Treat a locked state file as degraded persistence, never as a trading-process crash."""
    was_degraded = bool(state.get("state_persistence_degraded"))
    if atomic_json(state_path, state):
        if was_degraded:
            append_jsonl(
                event_log,
                {
                    "event": "state_persistence_recovered",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "prior_failures": int(state.get("state_persistence_failures", 0)),
                },
            )
        state["state_persistence_degraded"] = False
        return True
    state["state_persistence_degraded"] = True
    state["state_persistence_failures"] = int(state.get("state_persistence_failures", 0)) + 1
    append_jsonl(
        event_log,
        {
            "event": "state_persistence_degraded",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "failures": state["state_persistence_failures"],
            "state_path": str(state_path),
        },
    )
    return False


def owned_positions(executor: MT5Executor, magic: int = C1_DEMO_MAGIC) -> list[Any]:
    return [
        position
        for position in executor.get_positions("BTCUSD")
        if int(getattr(position, "magic", 0)) == magic
    ]


def owned_ticket_is_open(executor: MT5Executor, ticket: int, magic: int) -> bool:
    """Reconcile an ambiguous close result before sending another close order."""
    return any(int(position.ticket) == int(ticket) for position in owned_positions(executor, magic))


def timeout_close_with_recovery(
    executor: MT5Executor,
    ticket: int,
    magic: int,
    expected_account: int,
) -> tuple[Any, bool, str]:
    """Close once, then reconnect and retry only if the owned ticket still exists.

    A lost order response can coexist with a successful broker-side close. The
    ticket reconciliation prevents a duplicate close request in that case.
    """
    result = executor.close_position(int(ticket))
    if result.success:
        return result, True, "initial_close"
    if not owned_ticket_is_open(executor, ticket, magic):
        return result, True, "settled_after_initial_error"

    if not executor.reconnect():
        return result, False, "reconnect_failed"
    info = executor.mt5.account_info() if executor.mt5 is not None else None
    if info is None or int(getattr(info, "login", -1)) != int(expected_account):
        return result, False, "reconnect_wrong_account"
    if not owned_ticket_is_open(executor, ticket, magic):
        return result, True, "settled_after_reconnect"

    retry = executor.close_position(int(ticket))
    if retry.success:
        return retry, True, "reconnect_retry"
    if not owned_ticket_is_open(executor, ticket, magic):
        return retry, True, "settled_after_retry_error"
    return retry, False, "retry_failed"


def foreign_positions(executor: MT5Executor, magic: int) -> list[Any]:
    return [
        position
        for position in executor.get_positions("BTCUSD")
        if int(getattr(position, "magic", 0)) != magic
    ]


def guard_execution(
    executor: MT5Executor,
    expected_account: int,
    *,
    live_account: bool,
    max_live_balance: float,
    risk_overlay: str = RISK_OVERLAY_NONE,
) -> None:
    info = executor.mt5.account_info() if executor.mt5 is not None else None
    if info is None or int(info.login) != int(expected_account):
        raise RuntimeError(f"wrong account: got {getattr(info, 'login', None)}, expected {expected_account}")
    if int(getattr(info, "margin_mode", -1)) != 2:
        raise RuntimeError("C1 runner requires hedging mode")
    trade_mode = int(getattr(info, "trade_mode", -1))
    if live_account:
        live_magic(expected_account, risk_overlay)
        if trade_mode != 2:
            raise RuntimeError("C1 live runner permits only the authorized Tickmill live account")
        if float(getattr(info, "balance", float("inf"))) > max_live_balance:
            raise RuntimeError(
                f"C1 live runner balance exceeds safety limit: {info.balance} > {max_live_balance}"
            )
    elif trade_mode != 0:
        raise RuntimeError("C1 demo runner refuses non-demo accounts")


def clear_owned_state(state: dict[str, Any]) -> None:
    state["owned_ticket"] = None
    state["owned_entry_time"] = None
    state["owned_entry_time_source"] = None
    state["owned_entry_price"] = None
    state["owned_risk_distance"] = None
    state["owned_peak_r"] = 0.0
    state["break_even_applied"] = False
    state["ratchet_floor_r"] = None
    state["last_ratchet_attempt_at"] = None
    state["last_break_even_attempt_at"] = None
    state["last_overlay_exit_attempt_at"] = None
    state["last_timeout_exit_attempt_at"] = None


def clear_confirmed_spree_state(state: dict[str, Any]) -> None:
    clear_owned_state(state)
    state["spree_positions"] = {}
    state["spree_id"] = None
    state["spree_last_add_time"] = None


def corrected_broker_position_time(
    executor: MT5Executor,
    position: Any,
    now: pd.Timestamp | None = None,
) -> tuple[str, str]:
    """Recover a UTC open time without assuming the broker clock is UTC.

    Tickmill's position timestamp can be expressed in broker-server time.  A
    current broker tick reveals that offset relative to our UTC clock, so use
    it only for recovery when the runner's own persisted fill timestamp is
    unavailable.  A nonsensical/missing tick falls back to the raw timestamp
    rather than inventing an offset.
    """
    raw = pd.to_datetime(int(getattr(position, "time", 0)), unit="s", utc=True, errors="coerce")
    if pd.isna(raw):
        return pd.Timestamp.now(tz="UTC").isoformat(), "recovery_fallback_now"
    reference = now if now is not None else utc_now()
    tick = None
    if getattr(executor, "mt5", None) is not None:
        try:
            tick = executor.mt5.symbol_info_tick("BTCUSD")
        except Exception:
            tick = None
    tick_seconds = finite(getattr(tick, "time_msc", None))
    if tick_seconds is not None and tick_seconds > 0:
        tick_time = pd.to_datetime(int(tick_seconds), unit="ms", utc=True)
    else:
        tick_seconds = finite(getattr(tick, "time", None))
        tick_time = pd.to_datetime(int(tick_seconds), unit="s", utc=True) if tick_seconds else pd.NaT
    if not pd.isna(tick_time):
        offset_seconds = (tick_time - reference).total_seconds()
        if abs(offset_seconds) <= MAX_BROKER_CLOCK_OFFSET_SECONDS:
            return (raw - pd.Timedelta(seconds=offset_seconds)).isoformat(), "broker_clock_corrected"
    return raw.isoformat(), "broker_time_unadjusted"


def _spree_position_record(
    state: dict[str, Any], executor: MT5Executor, position: Any
) -> dict[str, Any]:
    records = state.setdefault("spree_positions", {})
    key = str(int(position.ticket))
    record = records.setdefault(key, {})
    legacy_match = int(state.get("owned_ticket") or -1) == int(position.ticket)
    entry = (
        finite(record.get("entry_price"))
        or (finite(state.get("owned_entry_price")) if legacy_match else None)
        or float(position.price_open)
    )
    configured_sl = finite(getattr(position, "sl", None))
    risk_distance = finite(record.get("risk_distance")) or (
        finite(state.get("owned_risk_distance")) if legacy_match else None
    )
    if risk_distance is None or risk_distance <= 0:
        inferred = abs(entry - configured_sl) if configured_sl not in (None, 0.0) else 0.0
        risk_distance = inferred if inferred > 0 else entry * STOP_WIDTH_BPS / 10000.0
    if not record.get("entry_time"):
        if legacy_match and state.get("owned_entry_time"):
            record["entry_time"] = state["owned_entry_time"]
            record["entry_time_source"] = state.get("owned_entry_time_source", "persisted_runner_fill")
        else:
            entry_time, source = corrected_broker_position_time(executor, position)
            record["entry_time"] = entry_time
            record["entry_time_source"] = source
    record["entry_price"] = float(entry)
    record["risk_distance"] = float(risk_distance)
    record.setdefault("peak_r", float(state.get("owned_peak_r") or 0.0) if legacy_match else 0.0)
    record.setdefault(
        "break_even_applied", bool(state.get("break_even_applied")) if legacy_match else False
    )
    record.setdefault(
        "last_break_even_attempt_at",
        state.get("last_break_even_attempt_at") if legacy_match else None,
    )
    record.setdefault(
        "last_overlay_exit_attempt_at",
        state.get("last_overlay_exit_attempt_at") if legacy_match else None,
    )
    record.setdefault(
        "last_timeout_exit_attempt_at",
        state.get("last_timeout_exit_attempt_at") if legacy_match else None,
    )
    return record


def reconcile_confirmed_spree(
    state: dict[str, Any], executor: MT5Executor, event_log: Path, magic: int
) -> None:
    positions = owned_positions(executor, magic)
    records = state.setdefault("spree_positions", {})
    open_tickets = {str(int(position.ticket)) for position in positions}
    for key in list(records):
        if key not in open_tickets:
            append_jsonl(
                event_log,
                {
                    "event": "spree_leg_closed_by_broker",
                    "ticket": int(key),
                    "spree_id": state.get("spree_id"),
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "magic": magic,
                },
            )
            records.pop(key, None)
    for position in positions:
        _spree_position_record(state, executor, position)
    if positions and not state.get("spree_id"):
        first = min(
            state["spree_positions"].values(),
            key=lambda record: pd.Timestamp(record["entry_time"]),
        )
        state["spree_id"] = f"recovered-{first['entry_time']}"
    if positions and not state.get("spree_last_add_time"):
        state["spree_last_add_time"] = max(
            record["entry_time"] for record in state["spree_positions"].values()
        )
    if not positions:
        clear_confirmed_spree_state(state)


def _spree_open_legs(
    state: dict[str, Any], positions: list[Any], tick: Any
) -> list[OpenLeg]:
    legs: list[OpenLeg] = []
    for position in positions:
        record = _spree_position_record(state, executor, position)
        direction = 1 if int(position.type) == 0 else -1
        executable = float(tick.bid if direction == 1 else tick.ask)
        current_r = direction * (executable - float(record["entry_price"])) / max(
            float(record["risk_distance"]), 1e-9
        )
        legs.append(OpenLeg(int(position.ticket), direction, float(current_r)))
    return legs


def _protect_spree_legs_for_add(
    state: dict[str, Any], executor: MT5Executor, event_log: Path, positions: list[Any], magic: int
) -> bool:
    protected = True
    for position in positions:
        record = _spree_position_record(state, executor, position)
        entry = float(record["entry_price"])
        direction = 1 if int(position.type) == 0 else -1
        current_sl = float(getattr(position, "sl", 0.0) or 0.0)
        already_protected = current_sl >= entry if direction == 1 else 0.0 < current_sl <= entry
        if already_protected:
            record["break_even_applied"] = True
            continue
        result = executor.modify_position_stops(int(position.ticket), entry, float(position.tp))
        append_jsonl(
            event_log,
            {
                "event": "spree_pre_add_break_even_sent",
                "ticket": int(position.ticket),
                "success": result.success,
                "error": result.error,
                "entry_price": entry,
                "spree_id": state.get("spree_id"),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "magic": magic,
            },
        )
        if result.success:
            record["break_even_applied"] = True
        else:
            protected = False
    return protected


def reconcile_position(
    state: dict[str, Any], executor: MT5Executor, event_log: Path, magic: int
) -> None:
    positions = owned_positions(executor, magic)
    if len(positions) > 1:
        raise RuntimeError("multiple C1-owned positions found")
    state_ticket = state.get("owned_ticket")
    if not positions:
        if state_ticket is not None:
            append_jsonl(
                event_log,
                {
                    "event": "position_closed_by_broker",
                    "ticket": state_ticket,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "magic": magic,
                },
            )
        clear_owned_state(state)
        return
    position = positions[0]
    state["owned_ticket"] = int(position.ticket)
    if not state.get("owned_entry_time"):
        entry_time, source = corrected_broker_position_time(executor, position)
        state["owned_entry_time"] = entry_time
        state["owned_entry_time_source"] = source
    entry_price = finite(state.get("owned_entry_price")) or finite(
        getattr(position, "price_open", None)
    )
    if entry_price is not None:
        state["owned_entry_price"] = entry_price
    risk_distance = finite(state.get("owned_risk_distance"))
    if entry_price is not None and (risk_distance is None or risk_distance <= 0):
        configured_sl = finite(getattr(position, "sl", None))
        inferred = abs(entry_price - configured_sl) if configured_sl is not None else 0.0
        state["owned_risk_distance"] = inferred if inferred > 0 else entry_price * STOP_WIDTH_BPS / 10000.0


def overlay_action(elapsed_minutes: float, peak_r: float, break_even_applied: bool) -> str:
    if peak_r >= BREAK_EVEN_ACTIVATION_R and not break_even_applied:
        return "apply_break_even"
    if elapsed_minutes >= TIME_DECAY_MINUTES and peak_r < BREAK_EVEN_ACTIVATION_R:
        return "time_decay_exit"
    return "hold"


def ratchet_floor_for_r(current_r: float) -> float | None:
    """Return the highest live floor newly justified by the executable tick."""
    if current_r >= 0.75:
        return 0.75
    if current_r >= 0.50:
        return 0.50
    return None


def retry_ready(state: dict[str, Any], key: str, now: pd.Timestamp) -> bool:
    previous = pd.to_datetime(
        state.get(key), utc=True, errors="coerce"
    )
    return pd.isna(previous) or (now - previous).total_seconds() >= EXIT_RETRY_SECONDS


def manage_ratchet_overlay(
    state: dict[str, Any],
    executor: MT5Executor,
    event_log: Path,
    now: pd.Timestamp,
    magic: int,
) -> None:
    """Move only this runner's broker SL to +0.5R then +0.75R.

    The floor arms from a currently executable Tickmill quote.  No manual
    close is sent on a pullback; the broker-held SL owns the exit after a
    successful modification.  This is the live analogue of the historical
    ratchet study, with a separate magic/account boundary.
    """
    positions = owned_positions(executor, magic)
    if len(positions) != 1:
        return
    position = positions[0]
    entry = finite(state.get("owned_entry_price")) or float(position.price_open)
    risk_distance = finite(state.get("owned_risk_distance")) or entry * STOP_WIDTH_BPS / 10_000.0
    tick = executor.mt5.symbol_info_tick("BTCUSD") if executor.mt5 is not None else None
    if tick is None or not tick.bid or not tick.ask or risk_distance <= 0.0:
        return
    side = 1 if int(position.type) == 0 else -1
    executable = float(tick.bid if side == 1 else tick.ask)
    current_r = side * (executable - entry) / risk_distance
    peak_r = max(float(state.get("owned_peak_r") or 0.0), current_r)
    state["owned_peak_r"] = peak_r
    desired_floor = ratchet_floor_for_r(current_r)
    active_floor = finite(state.get("ratchet_floor_r"))
    if desired_floor is None or (active_floor is not None and desired_floor <= active_floor):
        return
    if not retry_ready(state, "last_ratchet_attempt_at", now):
        return
    state["last_ratchet_attempt_at"] = now.isoformat()
    desired_sl = entry + side * desired_floor * risk_distance
    result = executor.modify_position_stops(int(position.ticket), desired_sl, float(position.tp))
    append_jsonl(
        event_log,
        {
            "event": "ratchet_floor_modify_sent",
            "ticket": int(position.ticket),
            "success": result.success,
            "error": result.error,
            "current_r": current_r,
            "peak_r": peak_r,
            "prior_floor_r": active_floor,
            "requested_floor_r": desired_floor,
            "requested_sl": desired_sl,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "magic": magic,
            "risk_overlay": RISK_OVERLAY_RATCHET_050_075_100,
        },
    )
    if result.success:
        state["ratchet_floor_r"] = desired_floor


def manage_risk_overlay(
    state: dict[str, Any],
    executor: MT5Executor,
    event_log: Path,
    now: pd.Timestamp,
    magic: int,
    risk_overlay: str,
) -> None:
    if risk_overlay == RISK_OVERLAY_NONE:
        return
    if risk_overlay == RISK_OVERLAY_RATCHET_050_075_100:
        manage_ratchet_overlay(state, executor, event_log, now, magic)
        return
    positions = owned_positions(executor, magic)
    if not positions:
        return
    position = positions[0]
    entry = finite(state.get("owned_entry_price")) or float(position.price_open)
    risk_distance = finite(state.get("owned_risk_distance")) or entry * STOP_WIDTH_BPS / 10000.0
    entry_time = pd.to_datetime(state.get("owned_entry_time"), utc=True, errors="coerce")
    tick = executor.mt5.symbol_info_tick("BTCUSD") if executor.mt5 is not None else None
    if pd.isna(entry_time) or tick is None or not tick.bid or not tick.ask or risk_distance <= 0:
        return
    side = 1 if int(position.type) == 0 else -1
    executable_price = float(tick.bid if side == 1 else tick.ask)
    current_r = side * (executable_price - entry) / risk_distance
    peak_r = max(float(state.get("owned_peak_r") or 0.0), current_r)
    state["owned_peak_r"] = peak_r
    elapsed_minutes = (now - entry_time).total_seconds() / 60.0
    action = overlay_action(elapsed_minutes, peak_r, bool(state.get("break_even_applied")))

    if action == "apply_break_even" and retry_ready(state, "last_break_even_attempt_at", now):
        state["last_break_even_attempt_at"] = now.isoformat()
        result = executor.modify_position_stops(int(position.ticket), entry, float(position.tp))
        append_jsonl(
            event_log,
            {
                "event": "risk_overlay_break_even_sent",
                "ticket": int(position.ticket),
                "success": result.success,
                "error": result.error,
                "entry_price": entry,
                "current_r": current_r,
                "peak_r": peak_r,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "magic": magic,
                "risk_overlay": risk_overlay,
            },
        )
        if result.success:
            state["break_even_applied"] = True
        return

    if action == "time_decay_exit" and retry_ready(state, "last_overlay_exit_attempt_at", now):
        state["last_overlay_exit_attempt_at"] = now.isoformat()
        result = executor.close_position(int(position.ticket))
        append_jsonl(
            event_log,
            {
                "event": "risk_overlay_time_decay_exit_sent",
                "ticket": int(position.ticket),
                "success": result.success,
                "price": result.price,
                "error": result.error,
                "current_r": current_r,
                "peak_r": peak_r,
                "elapsed_minutes": elapsed_minutes,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "magic": magic,
                "risk_overlay": risk_overlay,
            },
        )
        if result.success:
            clear_owned_state(state)


def manage_confirmed_spree_overlay(
    state: dict[str, Any],
    executor: MT5Executor,
    event_log: Path,
    now: pd.Timestamp,
    magic: int,
) -> None:
    tick = executor.mt5.symbol_info_tick("BTCUSD") if executor.mt5 is not None else None
    if tick is None or not tick.bid or not tick.ask:
        return
    for position in owned_positions(executor, magic):
        record = _spree_position_record(state, executor, position)
        entry = float(record["entry_price"])
        risk_distance = max(float(record["risk_distance"]), 1e-9)
        entry_time = pd.to_datetime(record.get("entry_time"), utc=True, errors="coerce")
        if pd.isna(entry_time):
            continue
        direction = 1 if int(position.type) == 0 else -1
        executable = float(tick.bid if direction == 1 else tick.ask)
        current_r = direction * (executable - entry) / risk_distance
        peak_r = max(float(record.get("peak_r") or 0.0), current_r)
        record["peak_r"] = peak_r
        elapsed_minutes = (now - entry_time).total_seconds() / 60.0
        action = overlay_action(elapsed_minutes, peak_r, bool(record.get("break_even_applied")))
        if action == "apply_break_even" and retry_ready(record, "last_break_even_attempt_at", now):
            record["last_break_even_attempt_at"] = now.isoformat()
            result = executor.modify_position_stops(int(position.ticket), entry, float(position.tp))
            append_jsonl(
                event_log,
                {
                    "event": "risk_overlay_break_even_sent",
                    "ticket": int(position.ticket),
                    "success": result.success,
                    "error": result.error,
                    "entry_price": entry,
                    "current_r": current_r,
                    "peak_r": peak_r,
                    "spree_id": state.get("spree_id"),
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "magic": magic,
                    "risk_overlay": RISK_OVERLAY_BE025_DECAY60,
                },
            )
            if result.success:
                record["break_even_applied"] = True
        elif action == "time_decay_exit" and retry_ready(record, "last_overlay_exit_attempt_at", now):
            record["last_overlay_exit_attempt_at"] = now.isoformat()
            result = executor.close_position(int(position.ticket))
            append_jsonl(
                event_log,
                {
                    "event": "risk_overlay_time_decay_exit_sent",
                    "ticket": int(position.ticket),
                    "success": result.success,
                    "price": result.price,
                    "error": result.error,
                    "current_r": current_r,
                    "peak_r": peak_r,
                    "elapsed_minutes": elapsed_minutes,
                    "spree_id": state.get("spree_id"),
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "magic": magic,
                    "risk_overlay": RISK_OVERLAY_BE025_DECAY60,
                },
            )
            if result.success:
                state.setdefault("spree_positions", {}).pop(str(int(position.ticket)), None)


def manage_confirmed_spree_timeout(
    state: dict[str, Any],
    executor: MT5Executor,
    event_log: Path,
    now: pd.Timestamp,
    magic: int,
    expected_account: int,
) -> None:
    for position in owned_positions(executor, magic):
        record = _spree_position_record(state, executor, position)
        entry_time = pd.to_datetime(record.get("entry_time"), utc=True, errors="coerce")
        if pd.isna(entry_time) or now < entry_time + pd.Timedelta(minutes=MAX_HOLD_MINUTES):
            continue
        if not retry_ready(record, "last_timeout_exit_attempt_at", now):
            continue
        record["last_timeout_exit_attempt_at"] = now.isoformat()
        result, settled, recovery_action = timeout_close_with_recovery(
            executor, int(position.ticket), magic, expected_account
        )
        append_jsonl(
            event_log,
            {
                "event": "time_exit_sent",
                "ticket": int(position.ticket),
                "success": settled,
                "close_request_success": result.success,
                "recovery_action": recovery_action,
                "price": result.price,
                "error": result.error,
                "spree_id": state.get("spree_id"),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "magic": magic,
            },
        )
        if settled:
            state.setdefault("spree_positions", {}).pop(str(int(position.ticket)), None)


def close_all_confirmed_spree_positions(
    state: dict[str, Any],
    executor: MT5Executor,
    event_log: Path,
    magic: int,
    event_name: str,
    **metadata: Any,
) -> None:
    for position in owned_positions(executor, magic):
        result = executor.close_position(int(position.ticket))
        append_jsonl(
            event_log,
            {
                "event": event_name,
                "ticket": int(position.ticket),
                "success": result.success,
                "price": result.price,
                "error": result.error,
                "spree_id": state.get("spree_id"),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "magic": magic,
                **metadata,
            },
        )
        if result.success:
            state.setdefault("spree_positions", {}).pop(str(int(position.ticket)), None)
    if not owned_positions(executor, magic):
        clear_confirmed_spree_state(state)


def manage_timeout(
    state: dict[str, Any],
    executor: MT5Executor,
    event_log: Path,
    now: pd.Timestamp,
    magic: int,
    expected_account: int,
) -> None:
    ticket = state.get("owned_ticket")
    entry_time = pd.to_datetime(state.get("owned_entry_time"), utc=True, errors="coerce")
    if ticket is None or pd.isna(entry_time) or now < entry_time + pd.Timedelta(minutes=MAX_HOLD_MINUTES):
        return
    if not retry_ready(state, "last_timeout_exit_attempt_at", now):
        return
    state["last_timeout_exit_attempt_at"] = now.isoformat()
    result, settled, recovery_action = timeout_close_with_recovery(
        executor, int(ticket), magic, expected_account
    )
    append_jsonl(
        event_log,
        {
            "event": "time_exit_sent",
            "ticket": int(ticket),
            "success": settled,
            "close_request_success": result.success,
            "recovery_action": recovery_action,
            "price": result.price,
            "error": result.error,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "magic": magic,
        },
    )
    if settled:
        clear_owned_state(state)


def close_owned_after_gap_reset(
    state: dict[str, Any],
    executor: MT5Executor,
    event_log: Path,
    gap_events: list[dict[str, Any]],
    magic: int,
) -> None:
    if not gap_events:
        return
    reconcile_position(state, executor, event_log, magic)
    ticket = state.get("owned_ticket")
    if ticket is None:
        return
    result = executor.close_position(int(ticket))
    append_jsonl(
        event_log,
        {
            "event": "gap_invalidated_position_exit",
            "ticket": int(ticket),
            "success": result.success,
            "price": result.price,
            "error": result.error,
            "gap_reasons": [event.get("reason") for event in gap_events],
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "magic": magic,
        },
    )
    if result.success:
        clear_owned_state(state)


def close_owned_for_foreign_overlap(
    state: dict[str, Any], executor: MT5Executor, event_log: Path, magic: int
) -> None:
    positions = owned_positions(executor, magic)
    if not positions or not foreign_positions(executor, magic):
        return
    position = positions[0]
    result = executor.close_position(int(position.ticket))
    append_jsonl(
        event_log,
        {
            "event": "foreign_overlap_exit",
            "ticket": int(position.ticket),
            "success": result.success,
            "price": result.price,
            "error": result.error,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "magic": magic,
        },
    )
    if result.success:
        clear_owned_state(state)


def open_candidate(
    state: dict[str, Any],
    executor: MT5Executor,
    event_log: Path,
    candidate: dict[str, Any],
    max_signal_age_seconds: float,
    magic: int = C1_DEMO_MAGIC,
    block_foreign_exposure: bool = False,
    live_account: bool = False,
    risk_overlay: str = RISK_OVERLAY_NONE,
    confirmed_spree: bool = False,
    max_venue_divergence_bps: float | None = None,
    signal_profile: str = SIGNAL_PROFILE_BASELINE,
) -> None:
    decision = pd.Timestamp(candidate["decision_time"])
    decision_iso = decision.isoformat()
    state["last_candidate_time"] = decision_iso
    age = (utc_now() - decision).total_seconds()
    if age < 0 or age > max_signal_age_seconds:
        append_jsonl(
            event_log,
            {
                "event": "candidate_rejected",
                "reason": "stale",
                "decision_time": decision_iso,
                "age_seconds": age,
            },
        )
        return
    positions = owned_positions(executor, magic)
    if positions and not confirmed_spree:
        append_jsonl(
            event_log,
            {
                "event": "candidate_rejected",
                "reason": "c1_position_open",
                "decision_time": decision_iso,
            },
        )
        return
    if block_foreign_exposure and foreign_positions(executor, magic):
        append_jsonl(
            event_log,
            {
                "event": "candidate_rejected",
                "reason": "foreign_position_open",
                "decision_time": decision_iso,
            },
        )
        return
    if not venue_divergence_allowed(candidate, max_venue_divergence_bps):
        append_jsonl(
            event_log,
            {
                "event": "candidate_rejected",
                "reason": "venue_divergence_gate",
                "decision_time": decision_iso,
                "venue_divergence_bps": finite(candidate.get("broker_vs_binance_mid_divergence_bps")),
                "max_venue_divergence_bps": max_venue_divergence_bps,
            },
        )
        return
    tick = executor.mt5.symbol_info_tick("BTCUSD") if executor.mt5 is not None else None
    if tick is None or not tick.bid or not tick.ask:
        append_jsonl(
            event_log,
            {
                "event": "candidate_rejected",
                "reason": "no_tick",
                "decision_time": decision_iso,
            },
        )
        return
    side = int(candidate["direction"])
    spree_decision = None
    if confirmed_spree:
        last_add = pd.to_datetime(state.get("spree_last_add_time"), utc=True, errors="coerce")
        minutes_since_last_add = (
            None if pd.isna(last_add) else (utc_now() - last_add).total_seconds() / 60.0
        )
        spree_decision = evaluate_confirmed_add(
            _spree_open_legs(state, positions, tick),
            side,
            minutes_since_last_add,
            CONFIRMED_SPREE_CONFIG,
        )
        if not spree_decision.allow_add:
            append_jsonl(
                event_log,
                {
                    "event": "candidate_rejected",
                    "reason": spree_decision.reason,
                    "decision_time": decision_iso,
                    "direction": side,
                    "sequence": spree_decision.sequence,
                    "minimum_leg_r": spree_decision.minimum_leg_r,
                    "spree_id": state.get("spree_id"),
                },
            )
            return
    mid = (float(tick.bid) + float(tick.ask)) / 2.0
    spread_bps = (float(tick.ask) - float(tick.bid)) / mid * 10000.0
    if spread_bps > 10.0:
        append_jsonl(
            event_log,
            {
                "event": "candidate_rejected",
                "reason": "spread_gate",
                "spread_bps": spread_bps,
                "decision_time": decision_iso,
            },
        )
        return
    if confirmed_spree and positions and not _protect_spree_legs_for_add(
        state, executor, event_log, positions, magic
    ):
        append_jsonl(
            event_log,
            {
                "event": "candidate_rejected",
                "reason": "spree_break_even_protection_failed",
                "decision_time": decision_iso,
                "direction": side,
                "sequence": spree_decision.sequence if spree_decision else len(positions),
                "spree_id": state.get("spree_id"),
            },
        )
        return
    entry_reference = float(tick.ask if side == 1 else tick.bid)
    stop_distance = entry_reference * STOP_WIDTH_BPS / 10000.0
    sl = entry_reference - side * stop_distance
    target_r = 1.0 if risk_overlay == RISK_OVERLAY_RATCHET_050_075_100 else TARGET_R
    tp = entry_reference + side * target_r * stop_distance
    lots = max(float(executor.volume_min or 0.01), 0.01)
    result = executor.open_position(
        side,
        lots,
        sl,
        tp,
        comment=(
            "C1SEQ_SPREE_LIVE"
            if confirmed_spree
            else (
                "C1SEQ_RATCHET_LIVE"
                if risk_overlay == RISK_OVERLAY_RATCHET_050_075_100
                else ("C1SEQ_RISK_LIVE" if risk_overlay != RISK_OVERLAY_NONE else "C1SEQ_BASE_LIVE")
            )
        ) if live_account else "C1SEQ_DEMO",
    )
    event = {
        "event": "live_order_result" if live_account else "demo_order_result",
        "decision_time": decision_iso,
        "direction": side,
        "anomaly_score": finite(candidate.get("anomaly_score")),
        "signal_profile": signal_profile,
        "agreement_components": candidate.get("agreement_components"),
        "lots": lots,
        "spread_bps": spread_bps,
        "sl": sl,
        "tp": tp,
        "target_r": target_r,
        "success": result.success,
        "ticket": result.ticket,
        "fill_price": result.price,
        "error": result.error,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "magic": magic,
        "execution_mode": "tickmill_live" if live_account else "fbs_demo",
        "risk_overlay": risk_overlay,
        "entry_kind": "addon" if positions else "seed",
        "sequence": len(positions),
        "confirmed_spree": confirmed_spree,
        "venue_divergence_bps": finite(candidate.get("broker_vs_binance_mid_divergence_bps")),
        "binance_source_after_receive_corrected": candidate.get("binance_source_after_receive_corrected"),
        "binance_source_clock_offset_ms": finite(candidate.get("binance_source_clock_offset_ms")),
        "spree_id": state.get("spree_id") or decision_iso,
        "experimental_unpromoted_model": True,
    }
    append_jsonl(event_log, event)
    if result.success:
        opened_at = utc_now().isoformat()
        actual_entry = float(result.price or entry_reference)
        if confirmed_spree:
            state["spree_id"] = state.get("spree_id") or decision_iso
            state["spree_last_add_time"] = opened_at
            state.setdefault("spree_positions", {})[str(int(result.ticket))] = {
                "entry_time": opened_at,
                "entry_time_source": "persisted_runner_fill",
                "entry_price": actual_entry,
                "risk_distance": abs(actual_entry - sl),
                "peak_r": 0.0,
                "break_even_applied": False,
                "last_break_even_attempt_at": None,
                "last_overlay_exit_attempt_at": None,
                "last_timeout_exit_attempt_at": None,
                "sequence": len(positions),
            }
        else:
            state["owned_ticket"] = int(result.ticket)
            state["owned_entry_time"] = opened_at
            state["owned_entry_time_source"] = "persisted_runner_fill"
            state["owned_entry_price"] = actual_entry
            state["owned_risk_distance"] = abs(actual_entry - sl)
            state["owned_peak_r"] = 0.0
            state["break_even_applied"] = False
            state["ratchet_floor_r"] = None
            state["last_ratchet_attempt_at"] = None
            state["last_break_even_attempt_at"] = None
            state["last_overlay_exit_attempt_at"] = None
            state["last_timeout_exit_attempt_at"] = None


def run(args: argparse.Namespace) -> int:
    if not args.enable_demo_orders:
        raise PermissionError("pass --enable-demo-orders; this archive does not support real-money execution")
    live_account = False
    confirmed_spree = bool(args.enable_confirmed_spree)
    if confirmed_spree and (
        not live_account
        or int(args.account) != TICKMILL_OVERLAY_ACCOUNT
        or args.risk_overlay != RISK_OVERLAY_BE025_DECAY60
    ):
        raise PermissionError(
            "confirmed spree is authorized only for the Tickmill risk-overlay live account"
        )
    if args.signal_profile == SIGNAL_PROFILE_NO_FLOW and live_account and int(args.account) != TICKMILL_NO_FLOW_ACCOUNT:
        raise PermissionError("no-flow profile is not authorized for this configuration")
    if args.signal_profile == SIGNAL_PROFILE_NO_FLOW and confirmed_spree:
        raise PermissionError("no-flow profile is seed-only; confirmed spree is not authorized")
    magic = C1_DEMO_MAGIC
    executor = MT5Executor(symbol="BTCUSD", magic=magic, deviation=20, terminal_path=args.terminal)
    if not executor.connect():
        raise RuntimeError(f"MT5 connect failed: {getattr(executor, 'mt5', None) and executor.mt5.last_error()}")
    guard_execution(
        executor,
        args.account,
        live_account=live_account,
        max_live_balance=args.max_live_balance,
        risk_overlay=args.risk_overlay,
    )
    state_path = Path(args.state)
    event_log = Path(args.event_log)
    state = load_state(state_path)
    append_jsonl(
        event_log,
        {
            "event": "runner_started",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "account": args.account,
            "magic": magic,
            "execution_mode": "tickmill_live" if live_account else "fbs_demo",
            "risk_overlay": args.risk_overlay,
            "signal_profile": args.signal_profile,
            "confirmed_spree": confirmed_spree,
            "spree_max_positions": CONFIRMED_SPREE_CONFIG.max_positions if confirmed_spree else 1,
            "spree_minimum_open_r": CONFIRMED_SPREE_CONFIG.minimum_open_r if confirmed_spree else None,
            "spree_minimum_minutes_between_adds": (
                CONFIRMED_SPREE_CONFIG.minimum_minutes_between_adds if confirmed_spree else None
            ),
            "minimum_history_minutes": 182,
            "orders_enabled": True,
            "experimental_unpromoted_model": True,
        },
    )
    startup_gap_events = state.pop("pending_gap_events", [])
    for gap_event in startup_gap_events:
        append_jsonl(event_log, gap_event)
    if confirmed_spree and startup_gap_events:
        close_all_confirmed_spree_positions(
            state,
            executor,
            event_log,
            magic,
            "gap_invalidated_position_exit",
            gap_reasons=[event.get("reason") for event in startup_gap_events],
        )
    else:
        close_owned_after_gap_reset(state, executor, event_log, startup_gap_events, magic)
    end_time = time.monotonic() + args.duration_seconds
    try:
        while time.monotonic() < end_time:
            guard_execution(
                executor,
                args.account,
                live_account=live_account,
                max_live_balance=args.max_live_balance,
                risk_overlay=args.risk_overlay,
            )
            now = utc_now()
            rows, malformed, offsets = read_jsonl_incremental(
                Path(args.sync_dir),
                state.get("offsets") or {},
                file_start=now - pd.Timedelta(hours=8),
            )
            state["offsets"] = offsets
            state["malformed_rows"] = int(state.get("malformed_rows", 0)) + malformed
            live_gap_events = update_minute_buckets(state, rows)
            for gap_event in live_gap_events:
                append_jsonl(event_log, gap_event)
            if confirmed_spree and live_gap_events:
                close_all_confirmed_spree_positions(
                    state,
                    executor,
                    event_log,
                    magic,
                    "gap_invalidated_position_exit",
                    gap_reasons=[event.get("reason") for event in live_gap_events],
                )
            else:
                close_owned_after_gap_reset(state, executor, event_log, live_gap_events, magic)
            finalize_minutes(state, now)
            if confirmed_spree:
                reconcile_confirmed_spree(state, executor, event_log, magic)
            else:
                reconcile_position(state, executor, event_log, magic)
            if live_account:
                if confirmed_spree and foreign_positions(executor, magic):
                    close_all_confirmed_spree_positions(
                        state, executor, event_log, magic, "foreign_overlap_exit"
                    )
                else:
                    close_owned_for_foreign_overlap(state, executor, event_log, magic)
            if confirmed_spree:
                manage_confirmed_spree_overlay(state, executor, event_log, now, magic)
                manage_confirmed_spree_timeout(
                    state, executor, event_log, now, magic, args.account
                )
                reconcile_confirmed_spree(state, executor, event_log, magic)
            else:
                manage_risk_overlay(state, executor, event_log, now, magic, args.risk_overlay)
                manage_timeout(state, executor, event_log, now, magic, args.account)
            candidate = latest_new_candidate(
                state,
                args.reference_hours,
                args.quantile,
                signal_profile=args.signal_profile,
            )
            if candidate is not None:
                open_candidate(
                    state,
                    executor,
                    event_log,
                    candidate,
                    args.max_signal_age_seconds,
                    magic=magic,
                    block_foreign_exposure=live_account,
                    live_account=live_account,
                    risk_overlay=args.risk_overlay,
                    confirmed_spree=confirmed_spree,
                    max_venue_divergence_bps=args.max_venue_divergence_bps,
                    signal_profile=args.signal_profile,
                )
            persist_state(state_path, state, event_log)
            time.sleep(args.poll_seconds)
    finally:
        persist_state(state_path, state, event_log)
        executor.disconnect()
    return 0


def smoke() -> dict[str, Any]:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=182, freq="1min")
    state = {"features": [], "last_candidate_time": None}
    for index, timestamp in enumerate(timestamps):
        value = 1.0 if index >= 180 else 0.0
        state["features"].append(
            {
                "feature_timestamp": timestamp.isoformat(),
                "feature_max_receive_time": (timestamp - pd.Timedelta(milliseconds=1)).isoformat(),
                "agg_trade_imbalance_60s": value,
                "signed_trade_notional_60s": value,
                "fbs_return_60s": value,
            }
        )
    candidate = latest_new_candidate(state, 6.0, 0.10)
    assert candidate is not None and int(candidate["direction"]) == 1
    assert pd.Timestamp(candidate["decision_time"]) >= timestamps[-2]
    assert venue_divergence_allowed(candidate, None)
    assert venue_divergence_allowed(
        {"broker_vs_binance_mid_divergence_bps": 4.0}, 5.0
    )
    no_flow_candidate = latest_new_candidate(
        state, 6.0, 0.10, signal_profile=SIGNAL_PROFILE_NO_FLOW
    )
    assert no_flow_candidate is not None and int(no_flow_candidate["direction"]) == 1
    assert live_magic(TICKMILL_NO_FLOW_ACCOUNT, RISK_OVERLAY_NONE) == C1_TICKMILL_NO_FLOW_MAGIC
    assert not venue_divergence_allowed(
        {"broker_vs_binance_mid_divergence_bps": -5.1}, 5.0
    )
    assert not venue_divergence_allowed({}, 5.0)
    gap_state = {
        "features": [
            {
                "feature_timestamp": timestamp.isoformat(),
                "feature_max_receive_time": (timestamp - pd.Timedelta(milliseconds=1)).isoformat(),
            }
            for timestamp in [*timestamps[:180], timestamps[-1] + pd.Timedelta(hours=2)]
        ],
        "minute_buckets": {},
    }
    startup_gap = sanitize_history_after_gap(gap_state)
    assert startup_gap is not None and startup_gap["reason"] == "startup_history_gap"
    assert len(gap_state["features"]) == 1
    live_gap_state = {
        "features": [{"feature_timestamp": timestamps[-1].isoformat()}],
        "minute_buckets": {},
        "last_valid_receive_time": timestamps[-1].isoformat(),
    }
    live_gap_events = update_minute_buckets(
        live_gap_state,
        [
            {
                "local_receive_time": (timestamps[-1] + pd.Timedelta(minutes=3)).isoformat(),
                "has_fbs_tick": 1,
                "incomplete": False,
                "fbs_bid": 100.0,
                "fbs_ask": 101.0,
                "binance_trade_imbalance_60s": 0.5,
                "binance_trade_notional_60s": 1000.0,
            }
        ],
    )
    assert len(live_gap_events) == 1 and live_gap_events[0]["reason"] == "live_receive_gap"
    assert live_gap_state["features"] == []
    broker_now = pd.Timestamp("2026-09-11T00:00:00Z")
    broker_offset = pd.Timedelta(hours=3)

    class BrokerClockExecutor:
        def __init__(self) -> None:
            self.mt5 = SimpleNamespace(
                symbol_info_tick=lambda _symbol: SimpleNamespace(
                    time_msc=int((broker_now + broker_offset).timestamp() * 1000)
                )
            )

    recovered_time, recovered_source = corrected_broker_position_time(
        BrokerClockExecutor(),  # type: ignore[arg-type]
        SimpleNamespace(time=int((broker_now + broker_offset).timestamp())),
        now=broker_now,
    )
    assert pd.Timestamp(recovered_time) == broker_now
    assert recovered_source == "broker_clock_corrected"
    with TemporaryDirectory() as directory:
        event_log = Path(directory) / "events.jsonl"
        stale = dict(candidate)
        stale["decision_time"] = pd.Timestamp("2026-01-01T00:00:00Z")
        open_candidate(state, None, event_log, stale, max_signal_age_seconds=10.0)  # type: ignore[arg-type]
        serialized = json.loads(event_log.read_text(encoding="utf-8").strip())
        assert serialized["reason"] == "stale"
        assert isinstance(serialized["decision_time"], str)
        owned_state = {"owned_ticket": 123, "owned_entry_time": timestamps[-1].isoformat()}

        class FakeExecutor:
            def get_positions(self, _symbol: str) -> list[Any]:
                return [SimpleNamespace(ticket=123, magic=C1_DEMO_MAGIC, time=1_767_229_200)]

            def close_position(self, ticket: int) -> Any:
                assert ticket == 123
                return SimpleNamespace(success=True, price=100.0, error="")

        close_owned_after_gap_reset(
            owned_state,
            FakeExecutor(),  # type: ignore[arg-type]
            event_log,
            [{"reason": "synthetic_gap"}],
            C1_DEMO_MAGIC,
        )
        terminal_event = json.loads(event_log.read_text(encoding="utf-8").splitlines()[-1])
        assert terminal_event["event"] == "gap_invalidated_position_exit"
        assert owned_state["owned_ticket"] is None

        overlap_state = {"owned_ticket": 456, "owned_entry_time": timestamps[-1].isoformat()}

        class OverlapExecutor:
            def get_positions(self, _symbol: str) -> list[Any]:
                return [
                    SimpleNamespace(ticket=456, magic=C1_TICKMILL_LIVE_MAGIC),
                    SimpleNamespace(ticket=789, magic=12345),
                ]

            def close_position(self, ticket: int) -> Any:
                assert ticket == 456
                return SimpleNamespace(success=True, price=100.0, error="")

        close_owned_for_foreign_overlap(
            overlap_state,
            OverlapExecutor(),  # type: ignore[arg-type]
            event_log,
            C1_TICKMILL_LIVE_MAGIC,
        )
        overlap_event = json.loads(event_log.read_text(encoding="utf-8").splitlines()[-1])
        assert overlap_event["event"] == "foreign_overlap_exit"
        assert overlap_state["owned_ticket"] is None

        spree_positions = [
            SimpleNamespace(
                ticket=901,
                magic=C1_TICKMILL_OVERLAY_MAGIC,
                time=1_767_229_200,
                type=0,
                price_open=100.0,
                price_current=103.0,
                sl=90.0,
                tp=105.0,
            ),
            SimpleNamespace(
                ticket=902,
                magic=C1_TICKMILL_OVERLAY_MAGIC,
                time=1_767_229_260,
                type=0,
                price_open=101.0,
                price_current=103.0,
                sl=90.9,
                tp=106.05,
            ),
        ]

        class SpreeExecutor:
            def __init__(self) -> None:
                self.modified: list[tuple[int, float]] = []

            def get_positions(self, _symbol: str) -> list[Any]:
                return spree_positions

            def modify_position_stops(self, ticket: int, sl: float, _tp: float) -> Any:
                self.modified.append((ticket, sl))
                return SimpleNamespace(success=True, price=103.0, error="")

        spree_state: dict[str, Any] = {
            "owned_ticket": 901,
            "owned_entry_time": timestamps[-1].isoformat(),
            "owned_entry_price": 99.5,
            "owned_risk_distance": 9.5,
            "owned_peak_r": 0.31,
            "break_even_applied": True,
            "spree_positions": {},
        }
        spree_executor = SpreeExecutor()
        reconcile_confirmed_spree(
            spree_state,
            spree_executor,  # type: ignore[arg-type]
            event_log,
            C1_TICKMILL_OVERLAY_MAGIC,
        )
        assert len(spree_state["spree_positions"]) == 2
        assert spree_state["spree_positions"]["901"]["entry_price"] == 99.5
        assert spree_state["spree_positions"]["901"]["risk_distance"] == 9.5
        assert spree_state["spree_positions"]["901"]["peak_r"] == 0.31
        assert spree_state["spree_positions"]["901"]["break_even_applied"] is True
        assert str(spree_state["spree_id"]).startswith("recovered-")
        assert _protect_spree_legs_for_add(
            spree_state,
            spree_executor,  # type: ignore[arg-type]
            event_log,
            spree_positions,
            C1_TICKMILL_OVERLAY_MAGIC,
        )
        assert spree_executor.modified == [(901, 99.5), (902, 101.0)]

        timeout_position = SimpleNamespace(ticket=987, magic=C1_TICKMILL_LIVE_MAGIC)

        class RecoveringTimeoutExecutor:
            def __init__(self) -> None:
                self.is_open = True
                self.close_calls = 0
                self.reconnect_calls = 0
                self.mt5 = SimpleNamespace(
                    account_info=lambda: SimpleNamespace(login=TICKMILL_LIVE_ACCOUNT)
                )

            def get_positions(self, _symbol: str) -> list[Any]:
                return [timeout_position] if self.is_open else []

            def close_position(self, ticket: int) -> Any:
                assert ticket == 987
                self.close_calls += 1
                if self.close_calls == 1:
                    return SimpleNamespace(success=False, price=0.0, error="disconnected")
                self.is_open = False
                return SimpleNamespace(success=True, price=100.0, error="")

            def reconnect(self) -> bool:
                self.reconnect_calls += 1
                return True

        timeout_executor = RecoveringTimeoutExecutor()
        timeout_result, timeout_settled, timeout_action = timeout_close_with_recovery(
            timeout_executor, 987, C1_TICKMILL_LIVE_MAGIC, TICKMILL_LIVE_ACCOUNT  # type: ignore[arg-type]
        )
        assert timeout_settled and timeout_action == "reconnect_retry"
        assert timeout_result.success and timeout_executor.close_calls == 2
        assert timeout_executor.reconnect_calls == 1

    class GuardExecutor:
        def __init__(self, login: int, trade_mode: int, balance: float) -> None:
            self.mt5 = SimpleNamespace(
                account_info=lambda: SimpleNamespace(
                    login=login,
                    trade_mode=trade_mode,
                    margin_mode=2,
                    balance=balance,
                )
            )

    guard_execution(
        GuardExecutor(TICKMILL_LIVE_ACCOUNT, 2, 100.0),  # type: ignore[arg-type]
        TICKMILL_LIVE_ACCOUNT,
        live_account=True,
        max_live_balance=200.0,
    )
    guard_execution(
        GuardExecutor(TICKMILL_OVERLAY_ACCOUNT, 2, 100.0),  # type: ignore[arg-type]
        TICKMILL_OVERLAY_ACCOUNT,
        live_account=True,
        max_live_balance=200.0,
        risk_overlay=RISK_OVERLAY_BE025_DECAY60,
    )
    guard_execution(
        GuardExecutor(TICKMILL_OVERLAY_ACCOUNT, 2, 100.0),  # type: ignore[arg-type]
        TICKMILL_OVERLAY_ACCOUNT,
        live_account=True,
        max_live_balance=200.0,
        risk_overlay=RISK_OVERLAY_RATCHET_050_075_100,
    )
    assert overlay_action(10.0, 0.26, False) == "apply_break_even"
    assert overlay_action(60.0, 0.24, False) == "time_decay_exit"
    assert overlay_action(60.0, 0.25, True) == "hold"
    assert ratchet_floor_for_r(0.49) is None
    assert ratchet_floor_for_r(0.50) == 0.50
    assert ratchet_floor_for_r(0.76) == 0.75
    retry_now = pd.Timestamp("2026-01-01T00:00:00Z")
    with TemporaryDirectory() as directory:
        event_log = Path(directory) / "ratchet.jsonl"
        position = SimpleNamespace(ticket=777, magic=C1_TICKMILL_RATCHET_MAGIC, type=0, price_open=100.0, tp=101.0)

        class RatchetExecutor:
            def __init__(self) -> None:
                self.bid = 100.5
                self.modified: list[tuple[int, float, float]] = []
                self.mt5 = SimpleNamespace(
                    symbol_info_tick=lambda _symbol: SimpleNamespace(bid=self.bid, ask=self.bid + 0.1)
                )

            def get_positions(self, _symbol: str) -> list[Any]:
                return [position]

            def modify_position_stops(self, ticket: int, sl: float, tp: float) -> Any:
                self.modified.append((ticket, sl, tp))
                return SimpleNamespace(success=True, price=self.bid, error="")

        ratchet_state = {"owned_entry_price": 100.0, "owned_risk_distance": 1.0, "owned_peak_r": 0.0}
        ratchet_executor = RatchetExecutor()
        manage_ratchet_overlay(
            ratchet_state, ratchet_executor, event_log, retry_now, C1_TICKMILL_RATCHET_MAGIC  # type: ignore[arg-type]
        )
        assert ratchet_executor.modified == [(777, 100.5, 101.0)]
        ratchet_executor.bid = 100.8
        manage_ratchet_overlay(
            ratchet_state, ratchet_executor, event_log,
            retry_now + pd.Timedelta(seconds=31), C1_TICKMILL_RATCHET_MAGIC  # type: ignore[arg-type]
        )
        assert ratchet_executor.modified[-1] == (777, 100.75, 101.0)
        assert ratchet_state["ratchet_floor_r"] == 0.75
    retry_state: dict[str, Any] = {}
    assert retry_ready(retry_state, "last_attempt", retry_now)
    retry_state["last_attempt"] = retry_now.isoformat()
    assert not retry_ready(retry_state, "last_attempt", retry_now + pd.Timedelta(seconds=29))
    assert retry_ready(retry_state, "last_attempt", retry_now + pd.Timedelta(seconds=30))
    for rejected_executor, expected_account in (
        (GuardExecutor(123, 2, 100.0), TICKMILL_LIVE_ACCOUNT),
        (GuardExecutor(TICKMILL_LIVE_ACCOUNT, 2, 201.0), TICKMILL_LIVE_ACCOUNT),
    ):
        try:
            guard_execution(
                rejected_executor,  # type: ignore[arg-type]
                expected_account,
                live_account=True,
                max_live_balance=200.0,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe live account configuration was accepted")
    return {
        "status": "smoke_passed",
        "causal_candidate": True,
        "startup_gap_history_reset": True,
        "live_gap_history_reset": True,
        "gap_owned_position_exit": True,
        "event_timestamp_serializable": True,
        "account_guard_present": True,
        "tickmill_live_guard_present": True,
        "tickmill_foreign_overlap_exit": True,
        "dedicated_magic": C1_DEMO_MAGIC,
        "tickmill_live_magic": C1_TICKMILL_LIVE_MAGIC,
        "tickmill_overlay_magic": C1_TICKMILL_OVERLAY_MAGIC,
        "tickmill_ratchet_magic": C1_TICKMILL_RATCHET_MAGIC,
        "tickmill_no_flow_magic": C1_TICKMILL_NO_FLOW_MAGIC,
        "tickmill_overlay_guard_present": True,
        "confirmed_spree_multi_ticket_reconciliation": True,
        "confirmed_spree_break_even_before_add": True,
        "minimum_lot": 0.01,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.set_defaults(func=lambda _: print(json.dumps(smoke(), indent=2)) or 0)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--account", type=int, default=DEFAULT_ACCOUNT)
    run_parser.add_argument("--terminal", default=DEFAULT_TERMINAL)
    run_parser.add_argument("--sync-dir", default=str(DEFAULT_SYNC_DIR))
    run_parser.add_argument("--event-log", default=str(DEFAULT_EVENT_LOG))
    run_parser.add_argument("--state", default=str(DEFAULT_STATE))
    run_parser.add_argument("--reference-hours", type=float, default=6.0)
    run_parser.add_argument("--quantile", type=float, default=0.10)
    run_parser.add_argument(
        "--signal-profile",
        choices=(SIGNAL_PROFILE_BASELINE, SIGNAL_PROFILE_NO_FLOW),
        default=SIGNAL_PROFILE_BASELINE,
        help="Frozen C1 component set. The no-flow profile excludes trade-imbalance flow.",
    )
    run_parser.add_argument("--max-signal-age-seconds", type=float, default=10.0)
    run_parser.add_argument(
        "--max-venue-divergence-bps",
        type=float,
        default=None,
        help="Optional absolute Binance/Tickmill mid-divergence gate; disabled by default.",
    )
    run_parser.add_argument("--poll-seconds", type=float, default=1.0)
    run_parser.add_argument("--duration-seconds", type=float, default=7_776_000.0)
    run_parser.add_argument("--enable-demo-orders", action="store_true")
    run_parser.add_argument("--enable-confirmed-spree", action="store_true")
    # Retained for diagnostic compatibility. The archive only accepts a demo
    # account, and its executor independently blocks every real-money mode.
    run_parser.add_argument("--max-live-balance", type=float, default=1_000_000.0)
    run_parser.add_argument(
        "--risk-overlay",
        choices=(
            RISK_OVERLAY_NONE,
            RISK_OVERLAY_BE025_DECAY60,
            RISK_OVERLAY_RATCHET_050_075_100,
        ),
        default=RISK_OVERLAY_NONE,
    )
    run_parser.set_defaults(func=run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
