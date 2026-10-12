"""Forward virtual-memory and guarded demo runner for probe-confirm-expand.

The runner is intentionally separate from bot_manager.py. It consumes frozen
T9 shadow decisions and synchronized FBS quotes, owns only its dedicated MT5
magic, and defaults to virtual-only operation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.mt5_executor import MT5Executor
from models.trend.probe_confirm_expand import (
    ProbeCandidate,
    ProbeConfig,
    ProbeLifecycle,
    ProbeMemory,
    ProbeMemoryRecord,
    assigned_arm,
    first_executable_quote,
    json_ready,
    utc_timestamp,
)


PCE_MAGIC = 20260730
T9_MAGIC = 20260517
EXPECTED_ACCOUNT = 0
DEFAULT_TERMINAL = None
DEFAULT_SHADOW_LOG = ROOT / "logs" / "champion_shadow_decisions.jsonl"
DEFAULT_SYNC_DIR = ROOT / "data" / "raw" / "live_probe" / "sync"
DEFAULT_EVENT_LOG = ROOT / "logs" / "probe_confirm_expand_events.jsonl"
DEFAULT_STATE = ROOT / "logs" / "probe_confirm_expand_state.json"
DEFAULT_REPORT_DIR = ROOT / "research" / "validation_runs"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(json_ready(payload), sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)


def atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    retries: int = 6,
    retry_delay_seconds: float = 0.05,
) -> bool:
    """Atomically persist JSON without letting a transient Windows file lock kill a runner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_ready(payload), indent=2, sort_keys=True).encode("utf-8")
    for attempt in range(max(1, retries)):
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            return True
        except PermissionError:
            if attempt + 1 >= max(1, retries):
                return False
            time.sleep(retry_delay_seconds * (attempt + 1))
        finally:
            try:
                temp.unlink(missing_ok=True)
            except PermissionError:
                pass
    return False


def read_jsonl(
    path: Path,
    *,
    file_start: Optional[pd.Timestamp] = None,
) -> tuple[list[dict[str, Any]], int]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if path.is_dir() and file_start is not None:
        first_date = utc_timestamp(file_start).date().isoformat()
        files = [file for file in files if file.stem >= first_date]
    rows: list[dict[str, Any]] = []
    malformed = 0
    for file in files:
        if not file.exists():
            continue
        with file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
    return rows, malformed


def read_jsonl_incremental(
    path: Path,
    offsets: dict[str, int],
    *,
    file_start: Optional[pd.Timestamp] = None,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    files = sorted(path.glob("*.jsonl"))
    if file_start is not None:
        first_date = utc_timestamp(file_start).date().isoformat()
        files = [file for file in files if file.stem >= first_date]
    rows: list[dict[str, Any]] = []
    malformed = 0
    updated = dict(offsets)
    for file in files:
        key = str(file.resolve())
        size = int(file.stat().st_size)
        offset = int(updated.get(key, 0))
        if offset < 0 or offset > size:
            offset = 0
        with file.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
            updated[key] = int(handle.tell())
    return rows, malformed, updated


def prepare_quotes(rows: list[dict[str, Any]]) -> pd.DataFrame:
    prepared = []
    for row in rows:
        if not row.get("has_fbs_tick") or row.get("incomplete"):
            continue
        bid = _finite(row.get("fbs_bid"))
        ask = _finite(row.get("fbs_ask"))
        receive = row.get("local_receive_time")
        if bid is None or ask is None or ask < bid or receive is None:
            continue
        ts = pd.to_datetime(receive, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        prepared.append(
            {
                "_ts": ts,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "spread_bps": (ask - bid) / max((bid + ask) / 2.0, 1e-12) * 10000.0,
            }
        )
    if not prepared:
        return pd.DataFrame(columns=["_ts", "bid", "ask", "mid", "spread_bps"])
    return (
        pd.DataFrame(prepared)
        .sort_values("_ts")
        .drop_duplicates("_ts", keep="last")
        .reset_index(drop=True)
    )


def completed_m5_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["timestamp", "close_time", "open", "high", "low", "close", "spread_bps"])
    frame = quotes.copy()
    frame["timestamp"] = frame["_ts"].dt.floor("5min")
    grouped = frame.groupby("timestamp", sort=True)
    bars = grouped.agg(
        open=("mid", "first"),
        high=("mid", "max"),
        low=("mid", "min"),
        close=("mid", "last"),
        spread_bps=("spread_bps", "last"),
        quote_count=("_ts", "size"),
        first_quote_time=("_ts", "first"),
        last_quote_time=("_ts", "last"),
    ).reset_index()
    bars["close_time"] = bars["timestamp"] + pd.Timedelta(minutes=5)
    latest_quote = quotes["_ts"].max()
    bars["quote_complete"] = (
        (bars["quote_count"] >= 120)
        & (bars["first_quote_time"] <= bars["timestamp"] + pd.Timedelta(seconds=5))
        & (bars["last_quote_time"] >= bars["close_time"] - pd.Timedelta(seconds=5))
    )
    return bars[
        (bars["close_time"] <= latest_quote) & bars["quote_complete"]
    ].reset_index(drop=True)


def shadow_candidate(row: dict[str, Any]) -> Optional[tuple[ProbeCandidate, str]]:
    action = str(row.get("action", ""))
    reason = str(row.get("reason", ""))
    direction = int(row.get("direction", 0) or 0)
    if direction not in (-1, 1):
        return None
    if action == "champion_shadow_would_enter":
        cohort = "accepted_seed"
        arm = "memory"
    elif action == "champion_shadow_blocked" and reason.startswith("entry_quality:"):
        cohort = "quality_blocked"
        arm = ""
    else:
        return None
    observed_raw = row.get("probe_join_time") or row.get("local_time")
    if observed_raw is None:
        return None
    observed = pd.to_datetime(observed_raw, utc=True, errors="coerce")
    if pd.isna(observed):
        return None
    price = _finite(row.get("price"))
    atr = _finite(row.get("atr"))
    if price is None or atr is None or price <= 0 or atr <= 0:
        return None
    decision_raw = row.get("decision_time") or observed
    decision = pd.to_datetime(decision_raw, utc=True, errors="coerce")
    if pd.isna(decision):
        decision = observed
    candidate = ProbeCandidate.build(
        source="t9_champion_shadow",
        cohort=cohort,
        decision_time=decision,
        observed_time=observed,
        direction=direction,
        reference_price=price,
        atr=atr,
        reason=reason,
        regime=str(row.get("regime", "")),
        strength=str(row.get("strength", "")),
        context={
            "shadow_action": action,
            "broker_decision_time": str(row.get("decision_time", "")),
            "quality_score": row.get("quality_score"),
            "spread_bps_at_decision": row.get("spread_bps"),
        },
    )
    return candidate, (arm or assigned_arm(candidate.candidate_id))


@dataclass
class ForwardArm:
    name: str
    engine: Optional[ProbeLifecycle] = None
    last_bar_close: Optional[pd.Timestamp] = None
    broker_tickets: dict[str, int] = field(default_factory=dict)
    broker_fills: dict[str, float] = field(default_factory=dict)
    actual_probe_sent: bool = False
    expansion_order_attempted: bool = False
    confirmation_comparison_recorded: bool = False

    @property
    def occupied(self) -> bool:
        return self.engine is not None and not self.engine.state.is_terminal

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine": None if self.engine is None else self.engine.to_dict(),
            "last_bar_close": self.last_bar_close,
            "broker_tickets": self.broker_tickets,
            "broker_fills": self.broker_fills,
            "actual_probe_sent": self.actual_probe_sent,
            "expansion_order_attempted": self.expansion_order_attempted,
            "confirmation_comparison_recorded": self.confirmation_comparison_recorded,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForwardArm":
        engine_payload = payload.get("engine")
        return cls(
            name=str(payload["name"]),
            engine=None if engine_payload is None else ProbeLifecycle.from_dict(engine_payload),
            last_bar_close=(
                None if payload.get("last_bar_close") in (None, "") else utc_timestamp(payload["last_bar_close"])
            ),
            broker_tickets={str(key): int(value) for key, value in payload.get("broker_tickets", {}).items()},
            broker_fills={str(key): float(value) for key, value in payload.get("broker_fills", {}).items()},
            actual_probe_sent=bool(payload.get("actual_probe_sent", False)),
            expansion_order_attempted=bool(payload.get("expansion_order_attempted", False)),
            confirmation_comparison_recorded=bool(
                payload.get("confirmation_comparison_recorded", False)
            ),
        )


class ForwardRunner:
    def __init__(
        self,
        *,
        shadow_log: Path,
        sync_dir: Path,
        event_log: Path,
        state_path: Path,
        config: ProbeConfig,
        enable_demo_orders: bool,
        historical_report: Optional[Path],
        terminal: str,
        max_candidate_age_seconds: int = 120,
        max_quote_delay_seconds: float = 2.0,
        backfill_hours: float = 0.0,
    ) -> None:
        self.shadow_log = shadow_log
        self.sync_dir = sync_dir
        self.event_log = event_log
        self.state_path = state_path
        self.config = config
        self.enable_demo_orders = bool(enable_demo_orders)
        self.historical_report = historical_report
        self.terminal = terminal
        self.max_candidate_age_seconds = int(max_candidate_age_seconds)
        self.max_quote_delay_seconds = float(max_quote_delay_seconds)
        self.backfill_hours = max(float(backfill_hours), 0.0)
        self.collection_started_at = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            hours=self.backfill_hours
        )
        self.seen: set[str] = set()
        self.memory = ProbeMemory()
        self.memory_recorded: set[str] = set()
        self.closed_tickets: set[int] = set()
        self.arms = {name: ForwardArm(name) for name in ("memory", "traded_demo")}
        self.executor: Optional[MT5Executor] = None
        self.issues: list[str] = []
        self.sync_offsets: dict[str, int] = {}
        self.quote_buffer = pd.DataFrame(columns=["_ts", "bid", "ask", "mid", "spread_bps"])
        self.state_loaded = False
        self._load_state()
        if not self.state_loaded and self.backfill_hours <= 0:
            self.sync_offsets = {
                str(file.resolve()): int(file.stat().st_size)
                for file in sorted(self.sync_dir.glob("*.jsonl"))
            }

    def run_once(self) -> dict[str, Any]:
        shadow_rows, shadow_bad = read_jsonl(self.shadow_log)
        candidates = []
        for row in shadow_rows:
            parsed = shadow_candidate(row)
            if parsed is None or parsed[0].candidate_id in self.seen:
                continue
            if parsed[0].observed_time < self.collection_started_at:
                continue
            candidates.append(parsed)
        candidates.sort(key=lambda item: (item[0].observed_time, item[0].candidate_id))
        active_starts = [
            arm.engine.state.candidate.observed_time
            for arm in self.arms.values()
            if arm.occupied and arm.engine is not None
        ]
        candidate_starts = [candidate.observed_time for candidate, _ in candidates]
        needed_starts = [*active_starts, *candidate_starts]
        sync_start = (
            min(needed_starts) - pd.Timedelta(minutes=5)
            if needed_starts
            else pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=10)
        )
        sync_rows, sync_bad, self.sync_offsets = read_jsonl_incremental(
            self.sync_dir,
            self.sync_offsets,
            file_start=sync_start,
        )
        fresh_quotes = prepare_quotes(sync_rows)
        quotes = pd.concat([self.quote_buffer, fresh_quotes], ignore_index=True)
        if not quotes.empty:
            quotes["_ts"] = pd.to_datetime(quotes["_ts"], utc=True, errors="coerce")
            quotes = (
                quotes.dropna(subset=["_ts"])
                .sort_values("_ts")
                .drop_duplicates("_ts", keep="last")
            )
            quotes = quotes[quotes["_ts"] >= sync_start].reset_index(drop=True)
        self.quote_buffer = quotes
        bars = completed_m5_quotes(quotes)

        if self.enable_demo_orders:
            self._connect_and_guard()
            self._reconcile_owned_positions()

        for candidate, arm_name in candidates:
            self._advance_arms(bars, candidate.observed_time)
            self._start_candidate(candidate, arm_name, quotes)
        self._advance_arms(bars, None)
        self._persist()
        event_rows, event_bad = read_jsonl(self.event_log)
        if self.executor is not None:
            self.executor.disconnect()
            self.executor = None
        return {
            "status": "pass" if not self.issues else "warning",
            "new_candidates": len(candidates),
            "seen_candidates": len(self.seen),
            "quote_rows": len(quotes),
            "completed_m5_bars": len(bars),
            "shadow_malformed": shadow_bad,
            "sync_malformed": sync_bad,
            "event_malformed": event_bad,
            "arms": {
                name: {
                    "occupied": arm.occupied,
                    "candidate_id": None if arm.engine is None else arm.engine.state.candidate.candidate_id,
                    "status": None if arm.engine is None else arm.engine.state.status.value,
                    "actual_probe_sent": arm.actual_probe_sent,
                }
                for name, arm in self.arms.items()
            },
            "issues": list(self.issues),
            "demo_orders_enabled": self.enable_demo_orders,
            "evidence": summarize_forward_events(event_rows),
        }

    def _start_candidate(self, candidate: ProbeCandidate, arm_name: str, quotes: pd.DataFrame) -> None:
        arm = self.arms[arm_name]
        if arm.occupied:
            self.seen.add(candidate.candidate_id)
            self._log(candidate, arm_name, "candidate_rejected", "arm_occupied")
            return
        quote = first_executable_quote(quotes, candidate.observed_time)
        if quote is None:
            age = (pd.Timestamp.now(tz="UTC") - candidate.observed_time).total_seconds()
            if age > self.max_quote_delay_seconds:
                self.seen.add(candidate.candidate_id)
                self._log(
                    candidate,
                    arm_name,
                    "candidate_rejected",
                    "no_executable_quote_within_limit",
                    {"age_seconds": age},
                )
            return
        self.seen.add(candidate.candidate_id)
        quote_delay = (utc_timestamp(quote["_ts"]) - candidate.observed_time).total_seconds()
        if quote_delay > self.max_quote_delay_seconds:
            self._log(
                candidate,
                arm_name,
                "candidate_rejected",
                "executable_quote_too_late",
                {"quote_delay_seconds": quote_delay},
            )
            return
        stop_width_bps = (
            abs(float(quote["mid"]) - candidate.stop_price)
            / max(abs(float(quote["mid"])), 1e-12)
            * 10000.0
        )
        modeled_probe_cost_r = float(quote["spread_bps"]) / max(stop_width_bps, 1e-12)
        if modeled_probe_cost_r > self.config.max_expansion_cost_r:
            self._log(
                candidate,
                arm_name,
                "candidate_rejected",
                "probe_cost_gate",
                {"modeled_cost_r": modeled_probe_cost_r},
            )
            return
        engine = ProbeLifecycle(candidate, self.config)
        memory_snapshot = self.memory.snapshot(candidate.observed_time, candidate.direction)
        engine.start(quote["_ts"], float(quote["mid"]), float(quote["spread_bps"]))
        arm.engine = engine
        arm.last_bar_close = None
        arm.broker_tickets = {}
        arm.broker_fills = {}
        arm.actual_probe_sent = False
        arm.expansion_order_attempted = False
        arm.confirmation_comparison_recorded = False
        self._flush_engine_events(arm)
        self._log(candidate, arm_name, "memory_snapshot", fields=memory_snapshot)

        if arm_name != "traded_demo" or candidate.cohort != "quality_blocked":
            return
        age = (pd.Timestamp.now(tz="UTC") - candidate.observed_time).total_seconds()
        if not self.enable_demo_orders:
            self._log(candidate, arm_name, "demo_not_sent", "demo_orders_disabled")
        elif age > self.max_candidate_age_seconds:
            self._log(candidate, arm_name, "demo_not_sent", "stale_candidate")
        elif self._has_t9_exposure():
            self._log(candidate, arm_name, "demo_not_sent", "t9_overlap")
        else:
            self._send_probe_order(arm)

    def _advance_arms(self, bars: pd.DataFrame, until: Optional[pd.Timestamp]) -> None:
        for arm in self.arms.values():
            if not arm.occupied or arm.engine is None:
                continue
            candidate = arm.engine.state.candidate
            for row in bars.itertuples(index=False):
                close_time = utc_timestamp(row.close_time)
                if close_time <= candidate.observed_time:
                    continue
                if arm.last_bar_close is not None and close_time <= arm.last_bar_close:
                    continue
                if until is not None and close_time > until:
                    break
                before = len(arm.engine.state.events)
                arm.engine.observe_bar(
                    open_mid=float(row.open),
                    high_mid=float(row.high),
                    low_mid=float(row.low),
                    close_mid=float(row.close),
                    close_time=close_time,
                    spread_bps=float(row.spread_bps),
                )
                arm.last_bar_close = close_time
                self._flush_engine_events(arm, start=before)
                self._mirror_broker_actions(arm)
                if arm.engine.state.is_terminal:
                    self._record_memory(arm)
                    self._close_owned(arm, "virtual_terminal")
                    break
            if self.enable_demo_orders and arm.actual_probe_sent and self._has_t9_exposure():
                self._close_owned(arm, "t9_overlap_exit")
                if not arm.engine.state.is_terminal:
                    before = len(arm.engine.state.events)
                    latest = bars.iloc[-1] if not bars.empty else None
                    mid = (
                        float(latest["close"])
                        if latest is not None
                        else float(arm.engine.state.candidate.reference_price)
                    )
                    arm.engine.force_close(pd.Timestamp.now(tz="UTC"), mid, "t9_overlap_exit")
                    self._flush_engine_events(arm, start=before)
                    self._record_memory(arm)

    def _record_memory(self, arm: ForwardArm) -> None:
        if arm.engine is None:
            return
        identifier = arm.engine.state.candidate.candidate_id
        if identifier in self.memory_recorded or not arm.engine.state.is_terminal:
            return
        self.memory.add_state(arm.engine.state)
        self.memory_recorded.add(identifier)
        self._log(
            arm.engine.state.candidate,
            arm.name,
            "probe_memory_completed",
            fields=arm.engine.state.summary(),
        )

    def _flush_engine_events(self, arm: ForwardArm, start: int = 0) -> None:
        if arm.engine is None:
            return
        for event in arm.engine.state.events[start:]:
            payload = event.to_dict()
            payload["arm"] = arm.name
            payload["source_cohort"] = arm.engine.state.candidate.cohort
            append_jsonl(self.event_log, payload)

    def _mirror_broker_actions(self, arm: ForwardArm) -> None:
        if not self.enable_demo_orders or not arm.actual_probe_sent or arm.engine is None:
            return
        events = arm.engine.state.events
        if (
            any(event.action == "expansion_opened" for event in events)
            and not arm.expansion_order_attempted
        ):
            arm.expansion_order_attempted = True
            self._send_expansion_order(arm)
        if (
            not arm.confirmation_comparison_recorded
            and any(event.action == "probe_confirmed" for event in events)
        ):
            arm.confirmation_comparison_recorded = True
            event = next(event for event in events if event.action == "probe_confirmed")
            actual_fill = arm.broker_fills.get("probe")
            if actual_fill is not None:
                candidate = arm.engine.state.candidate
                actual_risk = abs(actual_fill - candidate.stop_price)
                actual_progress = (
                    candidate.direction * (float(event.price) - actual_fill)
                    / max(actual_risk, 1e-12)
                )
                self._log(
                    candidate,
                    arm.name,
                    "demo_confirmation_comparison",
                    fields={
                        "virtual_confirmed": True,
                        "actual_confirmed": actual_progress >= self.config.confirmation_r,
                        "actual_progress_r": actual_progress,
                        "agreement": actual_progress >= self.config.confirmation_r,
                    },
                )

    def _connect_and_guard(self) -> None:
        gate = self._historical_demo_gate()
        if gate is None:
            raise RuntimeError("historical PCE report does not permit demo orders")
        self.config = ProbeConfig(**gate["demo_config"])
        executor = MT5Executor(symbol="BTCUSD", magic=PCE_MAGIC, deviation=20, terminal_path=self.terminal)
        if not executor.connect():
            raise RuntimeError("MT5 connection failed")
        info = executor.mt5.account_info() if executor.mt5 is not None else None
        if info is None or int(info.login) != EXPECTED_ACCOUNT:
            executor.disconnect()
            raise RuntimeError(f"wrong account: got {getattr(info, 'login', None)}, expected {EXPECTED_ACCOUNT}")
        if int(getattr(info, "trade_mode", -1)) != 0:
            executor.disconnect()
            raise RuntimeError("PCE demo runner refuses non-demo accounts")
        hedging = int(getattr(executor.mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2))
        if int(getattr(info, "margin_mode", -1)) != hedging:
            executor.disconnect()
            raise RuntimeError("PCE demo runner requires a hedging account")
        self.executor = executor

    def _historical_demo_gate(self) -> Optional[dict[str, Any]]:
        if self.historical_report is None or not self.historical_report.exists():
            return None
        payload = json.loads(self.historical_report.read_text(encoding="utf-8"))
        decision = payload.get("decision", {})
        if not decision.get("enable_demo_orders") or not decision.get("demo_config"):
            return None
        return decision

    def _positions(self) -> list[Any]:
        return [] if self.executor is None else self.executor.get_positions("BTCUSD")

    def _has_t9_exposure(self) -> bool:
        return any(int(getattr(pos, "magic", 0)) == T9_MAGIC for pos in self._positions())

    def _owned_positions(self) -> list[Any]:
        return [pos for pos in self._positions() if int(getattr(pos, "magic", 0)) == PCE_MAGIC]

    def _reconcile_owned_positions(self) -> None:
        known = {ticket for arm in self.arms.values() for ticket in arm.broker_tickets.values()}
        actual = {int(pos.ticket) for pos in self._owned_positions()}
        orphan = sorted(actual - known)
        if orphan:
            self.issues.append(f"orphan_owned_tickets:{orphan}")
            raise RuntimeError(f"unreconciled PCE tickets: {orphan}")
        for arm in self.arms.values():
            for kind, ticket in list(arm.broker_tickets.items()):
                if ticket in actual or ticket in self.closed_tickets:
                    continue
                fields = self._deal_summary(ticket)
                self._log(
                    None if arm.engine is None else arm.engine.state.candidate,
                    arm.name,
                    "demo_position_reconciled_closed",
                    fields=fields,
                )
                self.closed_tickets.add(ticket)
                arm.broker_tickets.pop(kind, None)
                if kind == "probe":
                    arm.actual_probe_sent = False
                    arm.expansion_order_attempted = True

    def _send_probe_order(self, arm: ForwardArm) -> None:
        assert self.executor is not None and arm.engine is not None and arm.engine.state.probe is not None
        candidate = arm.engine.state.candidate
        sent_at = time.perf_counter()
        result = self.executor.open_position(
            candidate.direction,
            0.01,
            candidate.stop_price,
            candidate.target_price,
            comment=f"PCE:q:{candidate.candidate_id[:8]}",
        )
        latency_ms = (time.perf_counter() - sent_at) * 1000.0
        if result.success:
            ticket = self._find_owned_ticket(candidate.candidate_id[:8])
            arm.broker_tickets["probe"] = int(ticket or result.ticket)
            arm.broker_fills["probe"] = float(result.price)
            arm.actual_probe_sent = True
            virtual_fill = float(arm.engine.state.probe.entry_fill)
            self._log(
                candidate,
                arm.name,
                "demo_probe_opened",
                "",
                {
                    "ticket": arm.broker_tickets["probe"],
                    "fill": result.price,
                    "latency_ms": latency_ms,
                    "fill_divergence_bps": (
                        candidate.direction * (float(result.price) - virtual_fill)
                        / max(abs(virtual_fill), 1e-12)
                        * 10000.0
                    ),
                },
            )
        else:
            self._log(candidate, arm.name, "demo_order_rejected", result.error)

    def _send_expansion_order(self, arm: ForwardArm) -> None:
        assert self.executor is not None and arm.engine is not None and arm.engine.state.expansion is not None
        leg = arm.engine.state.expansion
        probe = arm.engine.state.probe
        if probe is None or self._has_t9_exposure():
            return
        ratio = leg.units / max(probe.units, 1e-12)
        lots = min(0.04, _normalize_volume(0.01 * ratio, self.executor.volume_min, self.executor.volume_step))
        current = sum(float(pos.volume) for pos in self._owned_positions())
        lots = min(lots, max(0.05 - current, 0.0))
        lots = _normalize_volume(lots, self.executor.volume_min, self.executor.volume_step)
        if lots < max(self.executor.volume_min, 0.01):
            self._log(arm.engine.state.candidate, arm.name, "demo_expansion_rejected", "volume_budget")
            return
        sent_at = time.perf_counter()
        result = self.executor.open_position(
            leg.direction,
            lots,
            leg.stop_price,
            leg.target_price,
            comment=f"PCE:x:{arm.engine.state.candidate.candidate_id[:8]}",
        )
        latency_ms = (time.perf_counter() - sent_at) * 1000.0
        if result.success:
            ticket = self._find_owned_ticket(arm.engine.state.candidate.candidate_id[:8], exclude=set(arm.broker_tickets.values()))
            arm.broker_tickets["expansion"] = int(ticket or result.ticket)
            arm.broker_fills["expansion"] = float(result.price)
            self._log(
                arm.engine.state.candidate,
                arm.name,
                "demo_expansion_opened",
                "",
                {
                    "ticket": arm.broker_tickets["expansion"],
                    "lots": lots,
                    "fill": result.price,
                    "latency_ms": latency_ms,
                    "fill_divergence_bps": (
                        leg.direction * (float(result.price) - float(leg.entry_fill))
                        / max(abs(float(leg.entry_fill)), 1e-12)
                        * 10000.0
                    ),
                },
            )
        else:
            self._log(arm.engine.state.candidate, arm.name, "demo_order_rejected", result.error)

    def _find_owned_ticket(self, prefix: str, exclude: Optional[set[int]] = None) -> Optional[int]:
        exclude = exclude or set()
        rows = [
            pos for pos in self._owned_positions()
            if prefix in str(getattr(pos, "comment", "")) and int(pos.ticket) not in exclude
        ]
        return int(max(rows, key=lambda pos: int(getattr(pos, "time_msc", 0))).ticket) if rows else None

    def _close_owned(self, arm: ForwardArm, reason: str) -> None:
        if self.executor is None:
            return
        actual = {int(pos.ticket) for pos in self._owned_positions()}
        for kind, ticket in list(arm.broker_tickets.items()):
            if ticket not in actual:
                continue
            result = self.executor.close_position(ticket)
            self._log(
                arm.engine.state.candidate if arm.engine is not None else None,
                arm.name,
                "demo_position_closed" if result.success else "demo_close_failed",
                reason if result.success else result.error,
                {"kind": kind, "ticket": ticket, "fill": result.price},
            )
            if result.success:
                self.closed_tickets.add(ticket)
                arm.broker_tickets.pop(kind, None)

    def _deal_summary(self, ticket: int) -> dict[str, Any]:
        if self.executor is None or self.executor.mt5 is None:
            return {"ticket": int(ticket), "history_available": False}
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        deals = self.executor.mt5.history_deals_get(start, datetime.now(timezone.utc))
        matched = [
            deal
            for deal in list(deals or [])
            if int(
                getattr(deal, "position_id", 0)
                or getattr(deal, "position", 0)
                or 0
            )
            == int(ticket)
        ]
        return {
            "ticket": int(ticket),
            "history_available": deals is not None,
            "deal_count": len(matched),
            "realized_usd": float(
                sum(
                    float(getattr(deal, field, 0.0) or 0.0)
                    for deal in matched
                    for field in ("profit", "commission", "swap", "fee")
                )
            ),
            "last_price": (
                None if not matched else float(getattr(matched[-1], "price", 0.0) or 0.0)
            ),
        }

    def _log(
        self,
        candidate: Optional[ProbeCandidate],
        arm: str,
        action: str,
        reason: str = "",
        fields: Optional[dict[str, Any]] = None,
    ) -> None:
        append_jsonl(
            self.event_log,
            {
                "candidate_id": None if candidate is None else candidate.candidate_id,
                "event_time": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "arm": arm,
                "reason": reason,
                "fields": fields or {},
            },
        )

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.state_loaded = True
        self.collection_started_at = utc_timestamp(
            payload.get("collection_started_at", self.collection_started_at)
        )
        self.seen = set(payload.get("seen_candidates", []))
        self.memory_recorded = set(payload.get("memory_recorded", []))
        self.closed_tickets = {int(value) for value in payload.get("closed_tickets", [])}
        self.sync_offsets = {
            str(key): int(value) for key, value in payload.get("sync_offsets", {}).items()
        }
        quote_rows = payload.get("quote_buffer", [])
        self.quote_buffer = prepare_quotes(
            [
                {
                    "local_receive_time": row["_ts"],
                    "has_fbs_tick": 1,
                    "incomplete": False,
                    "fbs_bid": row["bid"],
                    "fbs_ask": row["ask"],
                }
                for row in quote_rows
            ]
        )
        self.memory = ProbeMemory(
            ProbeMemoryRecord(
                candidate_id=str(row["candidate_id"]),
                close_time=utc_timestamp(row["close_time"]),
                direction=int(row["direction"]),
                net_r=float(row["net_r"]),
                confirmed=bool(row["confirmed"]),
                failed=bool(row["failed"]),
            )
            for row in payload.get("memory_records", [])
        )
        for name, row in payload.get("arms", {}).items():
            if name in self.arms:
                self.arms[name] = ForwardArm.from_dict(row)

    def _persist(self) -> None:
        atomic_json(
            self.state_path,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "collection_started_at": self.collection_started_at,
                "seen_candidates": sorted(self.seen),
                "memory_recorded": sorted(self.memory_recorded),
                "closed_tickets": sorted(self.closed_tickets),
                "sync_offsets": self.sync_offsets,
                "quote_buffer": [
                    {
                        "_ts": utc_timestamp(row["_ts"]).isoformat(),
                        "bid": float(row["bid"]),
                        "ask": float(row["ask"]),
                    }
                    for row in self.quote_buffer.to_dict(orient="records")
                ],
                "memory_records": [asdict(row) for row in self.memory.records],
                "arms": {name: arm.to_dict() for name, arm in self.arms.items()},
                "issues": self.issues,
            },
        )


def _normalize_volume(value: float, minimum: float, step: float) -> float:
    minimum = max(float(minimum), 0.01)
    step = max(float(step), minimum)
    if value < minimum:
        return 0.0
    return float(math.floor((float(value) + 1e-12) / step) * step)


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def summarize_forward_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm_name in ("memory", "traded_demo"):
        arm_rows = [row for row in rows if row.get("arm") == arm_name]
        completed = [
            row.get("fields", {})
            for row in arm_rows
            if row.get("action") == "probe_memory_completed"
        ]
        assigned = {
            str(row.get("candidate_id"))
            for row in arm_rows
            if row.get("candidate_id")
        }
        arms[arm_name] = {
            "assigned_candidates": len(assigned),
            "completed_candidates": len(completed),
            "confirmed_candidates": sum(bool(row.get("confirmed")) for row in completed),
            "expanded_candidates": sum(bool(row.get("expanded")) for row in completed),
            **_virtual_economics(completed),
            "by_cohort": {
                cohort: _virtual_economics(
                    [row for row in completed if row.get("cohort") == cohort]
                )
                for cohort in ("accepted_seed", "quality_blocked")
            },
            "rejections": sum(row.get("action") == "candidate_rejected" for row in arm_rows),
            "demo_orders": sum(
                row.get("action") in {"demo_probe_opened", "demo_expansion_opened"}
                for row in arm_rows
            ),
        }
    divergences = [
        _finite(row.get("fields", {}).get("fill_divergence_bps"))
        for row in rows
        if row.get("action") in {"demo_probe_opened", "demo_expansion_opened"}
    ]
    divergence_values = np.abs(
        np.asarray([value for value in divergences if value is not None], dtype=float)
    )
    comparisons = [
        row.get("fields", {})
        for row in rows
        if row.get("action") == "demo_confirmation_comparison"
    ]
    return {
        "arms": arms,
        "actual_fill_divergence_bps": {
            "n": int(len(divergence_values)),
            "median": float(np.median(divergence_values)) if len(divergence_values) else None,
            "p95": float(np.percentile(divergence_values, 95)) if len(divergence_values) else None,
        },
        "traded_virtual_confirmation": {
            "n": len(comparisons),
            "agreement_rate": (
                float(np.mean([bool(row.get("agreement")) for row in comparisons]))
                if comparisons else None
            ),
        },
        "actual_cost_predictiveness": "insufficient_completed_demo_pairs",
    }


def _virtual_economics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row.get("net_r", 0.0)) for row in rows], dtype=float)
    wins = values[values > 0]
    losses = values[values <= 0]
    return {
        "trades": int(len(values)),
        "virtual_net_r": float(values.sum()) if len(values) else 0.0,
        "virtual_avg_r": float(values.mean()) if len(values) else 0.0,
        "virtual_pf": (
            float(wins.sum() / max(abs(losses.sum()), 1e-12))
            if len(values) else 0.0
        ),
    }


def write_report(payload: dict[str, Any], run_id: str) -> tuple[Path, Path]:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEFAULT_REPORT_DIR / f"probe_confirm_expand_forward_{run_id}.json"
    md_path = DEFAULT_REPORT_DIR / f"probe_confirm_expand_forward_{run_id}.md"
    json_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    lines = [
        "# Probe-Confirm-Expand Forward Report",
        "",
        f"- Status: `{payload['status']}`",
        f"- Demo orders enabled: `{payload['demo_orders_enabled']}`",
        f"- New candidates: `{payload['new_candidates']}`",
        f"- Seen candidates: `{payload['seen_candidates']}`",
        f"- FBS quote rows: `{payload['quote_rows']}`",
        f"- Completed M5 bars: `{payload['completed_m5_bars']}`",
        f"- Issues: `{len(payload['issues'])}`",
        "",
        "## Evidence By Arm",
        "",
        "| arm | assigned | completed | confirmed | expanded | virtual R | avg R | PF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in payload["evidence"]["arms"].items():
        lines.append(
            f"| {arm} | {row['assigned_candidates']} | {row['completed_candidates']} | "
            f"{row['confirmed_candidates']} | {row['expanded_candidates']} | "
            f"{row['virtual_net_r']:+.2f} | {row['virtual_avg_r']:+.3f} | {row['virtual_pf']:.3f} |"
        )
    lines.extend(
        [
            "",
        "This runner is separate from T9. Virtual collection is the default; demo orders require an explicit historical gate.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def smoke() -> dict[str, Any]:
    now = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = [
        {"local_receive_time": now.isoformat(), "has_fbs_tick": 1, "incomplete": False, "fbs_bid": 99.9, "fbs_ask": 100.1},
        {"local_receive_time": (now + pd.Timedelta(minutes=5)).isoformat(), "has_fbs_tick": 1, "incomplete": False, "fbs_bid": 104.9, "fbs_ask": 105.1},
        {"local_receive_time": (now + pd.Timedelta(minutes=10)).isoformat(), "has_fbs_tick": 1, "incomplete": False, "fbs_bid": 109.9, "fbs_ask": 110.1},
    ]
    quotes = prepare_quotes(rows)
    assert first_executable_quote(quotes, now + pd.Timedelta(seconds=1))["_ts"] == now + pd.Timedelta(minutes=5)
    assert assigned_arm("0" * 64) == "memory"
    candidate = ProbeCandidate.build(
        source="smoke",
        cohort="quality_blocked",
        decision_time=now,
        observed_time=now,
        direction=1,
        reference_price=100.0,
        atr=5.0,
        reason="entry_quality:smoke",
    )
    engine = ProbeLifecycle(candidate, ProbeConfig(spread_bps=10.0))
    engine.start(now, 100.0, 10.0)
    restored = ProbeLifecycle.from_dict(engine.to_dict())
    assert restored.state.candidate.candidate_id == candidate.candidate_id
    assert restored.state.probe is not None and restored.state.probe.open
    assert _normalize_volume(0.039, 0.01, 0.01) == 0.03
    guarded = ForwardRunner(
        shadow_log=Path("missing-shadow.jsonl"),
        sync_dir=Path("missing-sync"),
        event_log=Path("missing-events.jsonl"),
        state_path=Path("missing-state.json"),
        config=ProbeConfig(),
        enable_demo_orders=True,
        historical_report=None,
        terminal=DEFAULT_TERMINAL,
    )
    try:
        guarded._connect_and_guard()
    except RuntimeError as exc:
        assert "historical PCE report" in str(exc)
    else:
        raise AssertionError("demo order guard accepted a missing historical gate")
    with tempfile.TemporaryDirectory() as temp_dir:
        sync_dir = Path(temp_dir)
        assert atomic_json(sync_dir / "atomic.json", {"state": "ok"})
        assert json.loads((sync_dir / "atomic.json").read_text(encoding="utf-8"))["state"] == "ok"
        file = sync_dir / "2026-01-01.jsonl"
        file.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        first_rows, bad, offsets = read_jsonl_incremental(sync_dir, {})
        assert len(first_rows) == 1 and bad == 0
        with file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rows[1]) + "\n")
        second_rows, bad, _ = read_jsonl_incremental(sync_dir, offsets)
        assert len(second_rows) == 1 and bad == 0
        state_path = sync_dir / "state.json"
        persisted = ForwardRunner(
            shadow_log=sync_dir / "shadow.jsonl",
            sync_dir=sync_dir,
            event_log=sync_dir / "events.jsonl",
            state_path=state_path,
            config=ProbeConfig(),
            enable_demo_orders=False,
            historical_report=None,
            terminal=DEFAULT_TERMINAL,
        )
        persisted.quote_buffer = quotes
        persisted._persist()
        recovered = ForwardRunner(
            shadow_log=sync_dir / "shadow.jsonl",
            sync_dir=sync_dir,
            event_log=sync_dir / "events.jsonl",
            state_path=state_path,
            config=ProbeConfig(),
            enable_demo_orders=False,
            historical_report=None,
            terminal=DEFAULT_TERMINAL,
        )
        assert len(recovered.quote_buffer) == len(quotes)
    return {"status": "pass", "candidate_id": candidate.candidate_id}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    run = sub.add_parser("run")
    run.add_argument("--shadow-log", default=str(DEFAULT_SHADOW_LOG))
    run.add_argument("--sync-dir", default=str(DEFAULT_SYNC_DIR))
    run.add_argument("--event-log", default=str(DEFAULT_EVENT_LOG))
    run.add_argument("--state", default=str(DEFAULT_STATE))
    run.add_argument("--run-id", default="pce_forward")
    run.add_argument("--interval-seconds", type=int, default=30)
    run.add_argument("--once", action="store_true")
    run.add_argument("--enable-demo-orders", action="store_true")
    run.add_argument("--historical-report")
    run.add_argument("--terminal", default=DEFAULT_TERMINAL)
    run.add_argument("--backfill-hours", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(smoke(), indent=2))
        return 0
    while True:
        runner = ForwardRunner(
            shadow_log=Path(args.shadow_log),
            sync_dir=Path(args.sync_dir),
            event_log=Path(args.event_log),
            state_path=Path(args.state),
            config=ProbeConfig(),
            enable_demo_orders=bool(args.enable_demo_orders),
            historical_report=None if not args.historical_report else Path(args.historical_report),
            terminal=str(args.terminal),
            backfill_hours=float(args.backfill_hours),
        )
        payload = runner.run_once()
        paths = write_report(payload, str(args.run_id))
        print(json.dumps({**payload, "artifacts": [str(path) for path in paths]}, indent=2))
        if args.once:
            return 0
        time.sleep(max(int(args.interval_seconds), 5))


if __name__ == "__main__":
    raise SystemExit(main())
