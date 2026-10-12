"""Demo-only one-slot experiment for frozen T9 seeds and C1 no-flow.

This runner is intentionally separate from ``bot_manager.py``.  It gives the
two research strategies one shared BTCUSD position slot so they cannot create
accidental concurrent exposure on a single demo account.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.mt5_executor import MT5Executor
from models.trend.c1_sequence_demo_runner import (
    SIGNAL_PROFILE_NO_FLOW,
    finalize_minutes,
    latest_new_candidate,
    load_state as load_c1_feature_state,
    update_minute_buckets,
)
from models.trend.probe_confirm_expand_forward import atomic_json, read_jsonl_incremental
from models.trend.t9_champion_shadow_evaluator import ChampionShadowEvaluator, append_jsonl, json_ready


SYMBOL = "BTCUSD"
ACCOUNT = 0
T9_MAGIC = 20260915
C1_MAGIC = 20260916
LOT = 0.01
C1_STOP_BPS = 100.0
C1_TARGET_R = 0.5
C1_MAX_HOLD_MINUTES = 120
MAX_C1_SIGNAL_AGE_SECONDS = 10.0
MAX_T9_SIGNAL_AGE_SECONDS = 180.0
# This is a restart guard, not sizing or risk control. The portfolio remains
# limited by its fixed lot, one-position slot, and account/ticket checks.
MAX_DEMO_BALANCE = 1_000_000.0


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def default_state() -> dict[str, Any]:
    # Keep the established C1 feature-state contract, plus coordinator state.
    state = load_c1_feature_state(Path("__portfolio_state_does_not_exist__.json"))
    state.update(
        {
            "position": None,
            "last_t9_decision_time": None,
            "malformed_rows": 0,
        }
    )
    return state


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_state()
    defaults = default_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    return state


def persist(path: Path, state: dict[str, Any]) -> None:
    if not atomic_json(path, json_ready(state)):
        raise RuntimeError(f"cannot persist portfolio state: {path}")


def event(log: Path, kind: str, **values: Any) -> None:
    append_jsonl(log, {"event": kind, "observed_at": datetime.now(timezone.utc).isoformat(), **values})


def owned_positions(executor: MT5Executor) -> list[Any]:
    return [
        position
        for position in executor.get_positions(SYMBOL)
        if int(getattr(position, "magic", 0)) in {T9_MAGIC, C1_MAGIC}
    ]


def validate_demo_account(
    executor: MT5Executor,
    expected_account: int,
    max_balance: float,
    *,
    require_trading_enabled: bool = True,
) -> bool:
    info = executor.mt5.account_info() if executor.mt5 is not None else None
    terminal = executor.mt5.terminal_info() if executor.mt5 is not None else None
    if info is None or int(getattr(info, "login", 0)) != int(expected_account):
        raise RuntimeError(f"wrong account: {getattr(info, 'login', None)}; expected {expected_account}")
    if int(getattr(info, "trade_mode", -1)) != 0:
        raise RuntimeError("portfolio archive refuses non-demo accounts")
    if int(getattr(info, "margin_mode", -1)) != 2:
        raise RuntimeError("portfolio runner requires hedging mode")
    trading_enabled = bool(getattr(info, "trade_allowed", False)) and bool(getattr(terminal, "trade_allowed", False))
    if not trading_enabled and require_trading_enabled:
        raise RuntimeError("MT5 account or terminal AutoTrading is disabled")
    if float(getattr(info, "balance", float("inf"))) > max_balance:
        raise RuntimeError(f"balance safety cap exceeded: {info.balance} > {max_balance}")
    if executor.volume_min > LOT + 1e-9 or executor.volume_step <= 0 or abs(round(LOT / executor.volume_step) * executor.volume_step - LOT) > 1e-9:
        raise RuntimeError(f"BTCUSD 0.01 lot is not valid: min={executor.volume_min}, step={executor.volume_step}")
    return trading_enabled


def spread_and_price(executor: MT5Executor, direction: int) -> tuple[float, float]:
    tick = executor.mt5.symbol_info_tick(SYMBOL) if executor.mt5 is not None else None
    if tick is None or not tick.bid or not tick.ask:
        raise RuntimeError("no executable BTCUSD quote")
    bid, ask = float(tick.bid), float(tick.ask)
    mid = (bid + ask) / 2.0
    return (ask if direction == 1 else bid), (ask - bid) / max(mid, 1e-12) * 10000.0


def open_owned(
    executor: MT5Executor,
    *,
    owner: str,
    direction: int,
    stop_distance: float,
    target_distance: float,
    decision_time: pd.Timestamp,
    payload: dict[str, Any],
    state: dict[str, Any],
    log: Path,
) -> bool:
    magic = T9_MAGIC if owner == "t9" else C1_MAGIC
    reference, spread_bps = spread_and_price(executor, direction)
    sl = reference - direction * stop_distance
    tp = reference + direction * target_distance
    prior_magic = executor.magic
    try:
        executor.magic = magic
        result = executor.open_position(direction, LOT, sl, tp, comment="PFT9_SEED" if owner == "t9" else "PFC1_NOFLOW")
    finally:
        executor.magic = prior_magic
    event(
        log,
        "live_order_result",
        owner=owner,
        magic=magic,
        decision_time=decision_time.isoformat(),
        direction=direction,
        expected_entry=reference,
        fill_price=result.price,
        stop_distance=stop_distance,
        sl=sl,
        tp=tp,
        spread_bps=spread_bps,
        ticket=result.ticket,
        success=result.success,
        error=result.error,
        signal=json_ready(payload),
        execution_mode="demo_experimental",
        pyramiding_enabled=False,
    )
    if not result.success:
        return False
    fill = float(result.price or reference)
    hold_minutes = int(payload.get("max_hold", 24)) * 15 if owner == "t9" else C1_MAX_HOLD_MINUTES
    state["position"] = {
        "owner": owner,
        "magic": magic,
        "ticket": int(result.ticket),
        "direction": direction,
        "entry_time": now_utc().isoformat(),
        "entry_price": fill,
        "sl": sl,
        "tp": tp,
        "hold_until": (now_utc() + pd.Timedelta(minutes=hold_minutes)).isoformat(),
        "decision_time": decision_time.isoformat(),
    }
    return True


def reconcile_owned_position(executor: MT5Executor, state: dict[str, Any], log: Path) -> bool:
    """Return whether shared capacity is occupied; never adopt foreign exposure."""
    all_positions = executor.get_positions(SYMBOL)
    foreign = [p for p in all_positions if int(getattr(p, "magic", 0)) not in {T9_MAGIC, C1_MAGIC}]
    if foreign:
        event(log, "foreign_position_block", tickets=[int(p.ticket) for p in foreign])
        return True
    owned = owned_positions(executor)
    record = state.get("position")
    if record is None:
        if owned:
            raise RuntimeError("owned portfolio position exists without persisted state; refusing to adopt it")
        return False
    ticket = int(record["ticket"])
    matching = [p for p in owned if int(p.ticket) == ticket and int(getattr(p, "magic", 0)) == int(record["magic"])]
    if matching:
        return True
    event(log, "position_closed_by_broker", owner=record.get("owner"), ticket=ticket, state_record=record)
    state["position"] = None
    return False


def close_timeout(executor: MT5Executor, state: dict[str, Any], log: Path) -> None:
    record = state.get("position")
    if not record or now_utc() < pd.Timestamp(record["hold_until"]):
        return
    ticket = int(record["ticket"])
    positions = [p for p in owned_positions(executor) if int(p.ticket) == ticket and int(getattr(p, "magic", 0)) == int(record["magic"])]
    if not positions:
        return
    prior_magic = executor.magic
    try:
        executor.magic = int(record["magic"])
        result = executor.close_position(ticket)
    finally:
        executor.magic = prior_magic
    event(log, "time_exit_result", owner=record["owner"], ticket=ticket, success=result.success, fill_price=result.price, error=result.error)


def fresh_candidate(candidate: dict[str, Any] | None, max_age: float) -> tuple[pd.Timestamp, dict[str, Any]] | None:
    if not candidate:
        return None
    decision = pd.to_datetime(candidate.get("decision_time"), utc=True, errors="coerce")
    if pd.isna(decision):
        return None
    age = (now_utc() - decision).total_seconds()
    return (decision, candidate) if 0 <= age <= max_age else None


def fresh_t9_candidate(candidate: dict[str, Any] | None) -> tuple[pd.Timestamp, dict[str, Any]] | None:
    """Use the evaluator's UTC observation time, not MT5 server-bar time.

    ``decision_time`` is a terminal-bar timestamp and can be server-local.  It
    remains in the payload for auditability, but must not be compared directly
    to the UTC process clock when deciding whether an order is stale.
    """
    if not candidate:
        return None
    observed = pd.to_datetime(candidate.get("local_time"), utc=True, errors="coerce")
    if pd.isna(observed):
        return None
    age = (now_utc() - observed).total_seconds()
    return (observed, candidate) if 0 <= age <= MAX_T9_SIGNAL_AGE_SECONDS else None


def run(args: argparse.Namespace) -> None:
    if not args.enable_demo_orders:
        raise RuntimeError("refusing to start without --enable-demo-orders")
    state_path, event_log, sync_dir = Path(args.state), Path(args.event_log), Path(args.sync_dir)
    state = load_state(state_path)
    executor = MT5Executor(SYMBOL, magic=T9_MAGIC, terminal_path=args.terminal)
    if not executor.connect():
        raise RuntimeError("unable to connect to MT5 terminal")
    validate_demo_account(executor, args.account, args.max_demo_balance)
    if state.get("position") is None and executor.get_positions(SYMBOL):
        raise RuntimeError("BTCUSD position already open; close it or use a clean account before starting")
    evaluator = ChampionShadowEvaluator(executor, Path(args.t9_shadow_log), Path(args.t9_live_log))
    event(event_log, "runner_started", t9_magic=T9_MAGIC, c1_magic=C1_MAGIC, lots=LOT, mode="demo_experimental")
    deadline = time.monotonic() + args.duration_seconds
    try:
        while time.monotonic() < deadline:
            trading_enabled = validate_demo_account(
                executor,
                args.account,
                args.max_demo_balance,
                require_trading_enabled=False,
            )
            if not trading_enabled:
                event(event_log, "trading_disabled_wait", reason="mt5_autotrading_disabled")
                persist(state_path, state)
                time.sleep(max(0.2, args.poll_seconds))
                continue
            occupied = reconcile_owned_position(executor, state, event_log)
            close_timeout(executor, state, event_log)
            occupied = reconcile_owned_position(executor, state, event_log)

            rows, malformed, offsets = read_jsonl_incremental(sync_dir, state.get("offsets", {}), file_start=now_utc() - pd.Timedelta(hours=8))
            state["offsets"] = offsets
            state["malformed_rows"] = int(state.get("malformed_rows", 0)) + malformed
            for gap in update_minute_buckets(state, rows):
                event(event_log, gap.pop("event"), **gap)
            finalize_minutes(state, now_utc())
            c1 = fresh_candidate(latest_new_candidate(state, 6.0, 0.10, SIGNAL_PROFILE_NO_FLOW), MAX_C1_SIGNAL_AGE_SECONDS)
            t9_result = evaluator.run_once()
            t9_payload = t9_result.get("payload") if t9_result.get("logged") else None
            t9 = fresh_t9_candidate(t9_payload if t9_payload and t9_payload.get("stage") == "would_enter" else None)

            # A candidate is consumed exactly once even if shared capacity is occupied.
            if c1:
                state["last_candidate_time"] = c1[0].isoformat()
            if t9:
                state["last_t9_decision_time"] = t9[0].isoformat()
            candidates = [("t9", t9), ("c1", c1)]  # deterministic T9 precedence on same poll
            for owner, candidate in candidates:
                if not candidate:
                    continue
                decision, payload = candidate
                if occupied:
                    event(event_log, "candidate_suppressed", owner=owner, decision_time=decision.isoformat(), reason="shared_position_occupied", signal=json_ready(payload))
                    continue
                direction = int(payload["direction"])
                if owner == "t9":
                    stop_distance = finite(payload.get("stop_distance"))
                    if stop_distance is None or stop_distance <= 0:
                        event(event_log, "candidate_rejected", owner=owner, reason="invalid_stop_distance", signal=json_ready(payload))
                        continue
                    target_distance = stop_distance * 1.75
                else:
                    reference, _ = spread_and_price(executor, direction)
                    stop_distance = reference * C1_STOP_BPS / 10000.0
                    target_distance = stop_distance * C1_TARGET_R
                if open_owned(executor, owner=owner, direction=direction, stop_distance=stop_distance, target_distance=target_distance, decision_time=decision, payload=payload, state=state, log=event_log):
                    occupied = True
                else:
                    event(event_log, "candidate_rejected", owner=owner, reason="order_rejected", decision_time=decision.isoformat())
            persist(state_path, state)
            time.sleep(max(0.2, args.poll_seconds))
    finally:
        persist(state_path, state)
        executor.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo-only T9 seed + C1 no-flow one-position portfolio experiment")
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--enable-demo-orders", action="store_true")
    parser.add_argument("--account", type=int, default=ACCOUNT)
    parser.add_argument("--terminal", required=True)
    parser.add_argument("--sync-dir", required=True)
    parser.add_argument("--event-log", default=str(ROOT / "logs" / "portfolio_events.jsonl"))
    parser.add_argument("--state", default=str(ROOT / "logs" / "portfolio_state.json"))
    parser.add_argument("--t9-shadow-log", default=str(ROOT / "logs" / "t9_shadow.jsonl"))
    parser.add_argument("--t9-reference-log", dest="t9_live_log", default=str(ROOT / "logs" / "t9_reference.jsonl"))
    parser.add_argument("--duration-seconds", type=float, default=7 * 24 * 3600)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--max-demo-balance", type=float, default=MAX_DEMO_BALANCE)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
