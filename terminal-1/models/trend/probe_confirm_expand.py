"""Shared, live-safe probe-confirm-expand lifecycle.

The module contains no signal generation and no broker calls. Historical,
shadow, and demo runners feed it candidates and completed M5 observations.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from statistics import median
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


class ProbeStatus(str, Enum):
    IDLE = "idle"
    PROBING = "probing"
    CONFIRMED = "confirmed"
    EXPANDED = "expanded"
    FAILED = "failed"
    EXPIRED = "expired"
    CLOSED = "closed"
    REJECTED = "rejected"


TERMINAL_STATUSES = {
    ProbeStatus.FAILED,
    ProbeStatus.EXPIRED,
    ProbeStatus.CLOSED,
    ProbeStatus.REJECTED,
}


def utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def candidate_id(source: str, observed_time: Any, direction: int, reason: str) -> str:
    raw = f"{source}|{utc_timestamp(observed_time).isoformat()}|{int(direction)}|{reason}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def assigned_arm(identifier: str) -> str:
    """Deterministically assign a candidate before its outcome is known."""
    return "traded_demo" if int(identifier[:8], 16) % 2 else "memory"


@dataclass(frozen=True)
class ProbeCandidate:
    candidate_id: str
    source: str
    cohort: str
    decision_time: pd.Timestamp
    observed_time: pd.Timestamp
    direction: int
    reference_price: float
    atr: float
    stop_price: float
    target_price: float
    max_hold_minutes: int = 360
    reason: str = ""
    regime: str = ""
    strength: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.direction) not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if not math.isfinite(float(self.reference_price)) or float(self.reference_price) <= 0:
            raise ValueError("reference_price must be positive")
        if not math.isfinite(float(self.atr)) or float(self.atr) <= 0:
            raise ValueError("atr must be positive")
        if abs(float(self.reference_price) - float(self.stop_price)) <= 0:
            raise ValueError("stop distance must be positive")
        object.__setattr__(self, "decision_time", utc_timestamp(self.decision_time))
        object.__setattr__(self, "observed_time", utc_timestamp(self.observed_time))

    @property
    def stop_distance(self) -> float:
        return abs(float(self.reference_price) - float(self.stop_price))

    @classmethod
    def build(
        cls,
        *,
        source: str,
        cohort: str,
        decision_time: Any,
        observed_time: Any,
        direction: int,
        reference_price: float,
        atr: float,
        sl_atr: float = 2.0,
        tp_atr: float = 3.5,
        max_hold_minutes: int = 360,
        reason: str = "",
        regime: str = "",
        strength: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> "ProbeCandidate":
        direction = int(direction)
        price = float(reference_price)
        atr = float(atr)
        observed = utc_timestamp(observed_time)
        return cls(
            candidate_id=candidate_id(source, observed, direction, reason),
            source=str(source),
            cohort=str(cohort),
            decision_time=utc_timestamp(decision_time),
            observed_time=observed,
            direction=direction,
            reference_price=price,
            atr=atr,
            stop_price=price - direction * atr * float(sl_atr),
            target_price=price + direction * atr * float(tp_atr),
            max_hold_minutes=int(max_hold_minutes),
            reason=str(reason),
            regime=str(regime),
            strength=str(strength),
            context=context or {},
        )


@dataclass(frozen=True)
class ProbeConfig:
    probe_risk_fraction: float = 0.10
    confirmation_r: float = 0.50
    failure_r: float = 0.25
    expiry_minutes: int = 30
    spread_bps: float = 10.0
    max_expansion_cost_r: float = 0.15
    min_remaining_reward_risk: float = 1.25
    expansion_stop_mode: str = "probe_entry"
    reverse_after_failure: bool = False
    total_risk_budget: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < float(self.probe_risk_fraction) < float(self.total_risk_budget):
            raise ValueError("probe_risk_fraction must be inside total risk budget")
        if float(self.confirmation_r) <= 0 or float(self.failure_r) <= 0:
            raise ValueError("confirmation/failure thresholds must be positive")
        if int(self.expiry_minutes) <= 0:
            raise ValueError("expiry_minutes must be positive")
        if self.expansion_stop_mode not in {"probe_entry", "original_stop"}:
            raise ValueError("unknown expansion_stop_mode")


@dataclass
class ProbeEvent:
    candidate_id: str
    event_time: pd.Timestamp
    action: str
    status: str
    direction: int
    price: Optional[float] = None
    reason: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["event_time"] = utc_timestamp(self.event_time).isoformat()
        return row


@dataclass
class ProbeLeg:
    kind: str
    direction: int
    entry_time: pd.Timestamp
    entry_mid: float
    entry_fill: float
    units: float
    stop_price: float
    target_price: float
    entry_spread_cost_r: float
    open: bool = True
    exit_time: Optional[pd.Timestamp] = None
    exit_mid: Optional[float] = None
    exit_fill: Optional[float] = None
    exit_reason: str = ""
    gross_r: float = 0.0
    net_r: float = 0.0
    total_cost_r: float = 0.0


@dataclass
class ProbeState:
    candidate: ProbeCandidate
    config: ProbeConfig
    status: ProbeStatus = ProbeStatus.IDLE
    probe: Optional[ProbeLeg] = None
    expansion: Optional[ProbeLeg] = None
    pending_expansion_mode: Optional[str] = None
    confirmation_time: Optional[pd.Timestamp] = None
    terminal_time: Optional[pd.Timestamp] = None
    mfe_r: float = 0.0
    mae_r: float = 0.0
    events: list[ProbeEvent] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES and not self.pending_expansion_mode

    @property
    def open_legs(self) -> list[ProbeLeg]:
        return [leg for leg in [self.probe, self.expansion] if leg is not None and leg.open]

    @property
    def gross_r(self) -> float:
        return float(sum(leg.gross_r for leg in [self.probe, self.expansion] if leg is not None and not leg.open))

    @property
    def net_r(self) -> float:
        return float(sum(leg.net_r for leg in [self.probe, self.expansion] if leg is not None and not leg.open))

    @property
    def total_cost_r(self) -> float:
        return float(sum(leg.total_cost_r for leg in [self.probe, self.expansion] if leg is not None and not leg.open))

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "source": self.candidate.source,
            "cohort": self.candidate.cohort,
            "decision_time": self.candidate.decision_time.isoformat(),
            "observed_time": self.candidate.observed_time.isoformat(),
            "direction": int(self.candidate.direction),
            "reason": self.candidate.reason,
            "regime": self.candidate.regime,
            "status": self.status.value,
            "terminal_time": None if self.terminal_time is None else self.terminal_time.isoformat(),
            "confirmed": self.confirmation_time is not None,
            "confirmation_time": None if self.confirmation_time is None else self.confirmation_time.isoformat(),
            "expanded": self.expansion is not None,
            "expansion_mode": None if self.expansion is None else self.expansion.kind,
            "gross_r": self.gross_r,
            "net_r": self.net_r,
            "total_cost_r": self.total_cost_r,
            "mfe_r": float(self.mfe_r),
            "mae_r": float(self.mae_r),
            "probe_r": None if self.probe is None or self.probe.open else float(self.probe.net_r),
            "expansion_r": None if self.expansion is None or self.expansion.open else float(self.expansion.net_r),
            "probe_exit_reason": "" if self.probe is None else self.probe.exit_reason,
            "expansion_exit_reason": "" if self.expansion is None else self.expansion.exit_reason,
            "event_count": len(self.events),
            "open_leg_count": len(self.open_legs),
        }


@dataclass(frozen=True)
class ProbeMemoryRecord:
    candidate_id: str
    close_time: pd.Timestamp
    direction: int
    net_r: float
    confirmed: bool
    failed: bool


class ProbeMemory:
    """Rolling probe outcomes with a strict completed-before-decision rule."""

    def __init__(self, records: Optional[Iterable[ProbeMemoryRecord]] = None) -> None:
        self.records: list[ProbeMemoryRecord] = sorted(
            list(records or []), key=lambda row: utc_timestamp(row.close_time)
        )

    def add_state(self, state: ProbeState) -> None:
        if not state.is_terminal or state.terminal_time is None:
            raise ValueError("only terminal probe states may enter memory")
        record = ProbeMemoryRecord(
            candidate_id=state.candidate.candidate_id,
            close_time=utc_timestamp(state.terminal_time),
            direction=int(state.candidate.direction),
            net_r=float(state.net_r),
            confirmed=state.confirmation_time is not None,
            failed=state.status == ProbeStatus.FAILED,
        )
        self.records.append(record)
        self.records.sort(key=lambda row: row.close_time)

    def snapshot(self, decision_time: Any, direction: int) -> dict[str, float]:
        decision = utc_timestamp(decision_time)
        eligible = [row for row in self.records if utc_timestamp(row.close_time) <= decision]
        same = [row for row in eligible if int(row.direction) == int(direction)]
        out: dict[str, float] = {
            "memory_count": float(len(eligible)),
            "same_direction_memory_count": float(len(same)),
        }
        for window in (3, 5, 10):
            rows = eligible[-window:]
            values = [float(row.net_r) for row in rows]
            out[f"last_{window}_mean_r"] = float(np.mean(values)) if values else 0.0
            out[f"last_{window}_median_r"] = float(median(values)) if values else 0.0
            out[f"last_{window}_success_rate"] = (
                float(np.mean([value > 0 for value in values])) if values else 0.0
            )
        same_values = [float(row.net_r) for row in same[-5:]]
        out["same_direction_last_5_success_rate"] = (
            float(np.mean([value > 0 for value in same_values])) if same_values else 0.0
        )
        out["failure_streak"] = float(_tail_streak(eligible, lambda row: row.failed))
        out["confirmation_streak"] = float(_tail_streak(eligible, lambda row: row.confirmed))
        confirmed = [row for row in eligible if row.confirmed]
        failed = [row for row in eligible if row.failed]
        out["hours_since_confirmation"] = _hours_since(confirmed, decision)
        out["hours_since_failure"] = _hours_since(failed, decision)
        for half_life in (6.0, 12.0, 24.0):
            weights = []
            values = []
            for row in eligible:
                age_h = max((decision - utc_timestamp(row.close_time)).total_seconds() / 3600.0, 0.0)
                weights.append(0.5 ** (age_h / half_life))
                values.append(float(row.net_r))
            out[f"ewma_r_{int(half_life)}h"] = (
                float(np.average(values, weights=weights)) if weights and sum(weights) > 0 else 0.0
            )
        return out


class ProbeLifecycle:
    def __init__(
        self,
        candidate: ProbeCandidate,
        config: Optional[ProbeConfig] = None,
        *,
        record_events: bool = True,
    ) -> None:
        self.state = ProbeState(candidate=candidate, config=config or ProbeConfig())
        self.record_events = bool(record_events)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, record_events: bool = True) -> "ProbeLifecycle":
        candidate_payload = dict(payload["candidate"])
        candidate_payload["decision_time"] = utc_timestamp(candidate_payload["decision_time"])
        candidate_payload["observed_time"] = utc_timestamp(candidate_payload["observed_time"])
        candidate = ProbeCandidate(**candidate_payload)
        config = ProbeConfig(**payload["config"])
        engine = cls(candidate, config, record_events=record_events)
        engine.state.status = ProbeStatus(payload["status"])
        engine.state.probe = _leg_from_dict(payload.get("probe"))
        engine.state.expansion = _leg_from_dict(payload.get("expansion"))
        engine.state.pending_expansion_mode = payload.get("pending_expansion_mode")
        engine.state.confirmation_time = _optional_timestamp(payload.get("confirmation_time"))
        engine.state.terminal_time = _optional_timestamp(payload.get("terminal_time"))
        engine.state.mfe_r = float(payload.get("mfe_r", 0.0))
        engine.state.mae_r = float(payload.get("mae_r", 0.0))
        engine.state.events = [
            ProbeEvent(
                candidate_id=str(row["candidate_id"]),
                event_time=utc_timestamp(row["event_time"]),
                action=str(row["action"]),
                status=str(row["status"]),
                direction=int(row["direction"]),
                price=None if row.get("price") is None else float(row["price"]),
                reason=str(row.get("reason", "")),
                fields=dict(row.get("fields", {})),
            )
            for row in payload.get("events", [])
        ]
        return engine

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "candidate": asdict(self.state.candidate),
            "config": asdict(self.state.config),
            "status": self.state.status.value,
            "probe": _leg_to_dict(self.state.probe),
            "expansion": _leg_to_dict(self.state.expansion),
            "pending_expansion_mode": self.state.pending_expansion_mode,
            "confirmation_time": self.state.confirmation_time,
            "terminal_time": self.state.terminal_time,
            "mfe_r": self.state.mfe_r,
            "mae_r": self.state.mae_r,
            "events": [event.to_dict() for event in self.state.events],
        }
        return json_ready(payload)

    def reject(self, when: Any, reason: str) -> ProbeState:
        self.state.status = ProbeStatus.REJECTED
        self.state.terminal_time = utc_timestamp(when)
        self._event(when, "candidate_rejected", reason=reason)
        return self.state

    def start(self, when: Any, mid: float, spread_bps: Optional[float] = None) -> ProbeState:
        if self.state.status != ProbeStatus.IDLE:
            raise RuntimeError("probe already started")
        when = utc_timestamp(when)
        if when < self.state.candidate.observed_time:
            raise ValueError("probe fill cannot precede observed_time")
        direction = self.state.candidate.direction
        spread = self._spread(spread_bps)
        fill = executable_price(mid, direction, spread)
        risk_distance = abs(fill - self.state.candidate.stop_price)
        units = float(self.state.config.probe_risk_fraction) / max(risk_distance, 1e-12)
        entry_cost = abs(fill - float(mid)) * units
        self.state.probe = ProbeLeg(
            kind="probe",
            direction=direction,
            entry_time=when,
            entry_mid=float(mid),
            entry_fill=fill,
            units=units,
            stop_price=float(self.state.candidate.stop_price),
            target_price=float(self.state.candidate.target_price),
            entry_spread_cost_r=entry_cost,
        )
        self.state.status = ProbeStatus.PROBING
        self._event(when, "probe_opened", fill, fields={"mid": mid, "spread_bps": spread, "units": units})
        return self.state

    def observe_bar(
        self,
        *,
        open_mid: float,
        high_mid: float,
        low_mid: float,
        close_mid: float,
        close_time: Any,
        spread_bps: Optional[float] = None,
    ) -> ProbeState:
        close_time = utc_timestamp(close_time)
        if self.state.status == ProbeStatus.IDLE or self.state.is_terminal:
            return self.state
        if close_time != close_time.floor("5min"):
            raise ValueError("probe decisions require a completed M5 boundary")
        if self.state.probe is not None and close_time <= self.state.probe.entry_time:
            raise ValueError("completed bar must be after probe entry")
        spread = self._spread(spread_bps)

        if self.state.pending_expansion_mode is not None:
            self._open_expansion(close_time - pd.Timedelta(minutes=5), float(open_mid), spread)

        self._update_excursions(float(high_mid), float(low_mid), spread)
        for leg in list(self.state.open_legs):
            reason, exit_price, gross_exit_mid = self._bar_exit(
                leg,
                float(open_mid),
                float(high_mid),
                float(low_mid),
                spread,
            )
            if reason:
                self._close_leg(
                    leg,
                    close_time,
                    exit_price,
                    spread,
                    reason,
                    exit_price_is_fill=True,
                    gross_exit_mid=gross_exit_mid,
                )

        if self.state.probe is not None and self.state.probe.open and self.state.status == ProbeStatus.PROBING:
            close_fill = executable_price(close_mid, -self.state.candidate.direction, spread)
            progress_r = (
                self.state.candidate.direction
                * (close_fill - self.state.probe.entry_fill)
                / self.state.candidate.stop_distance
            )
            elapsed = (close_time - self.state.probe.entry_time).total_seconds() / 60.0
            if progress_r >= float(self.state.config.confirmation_r):
                self.state.status = ProbeStatus.CONFIRMED
                self.state.confirmation_time = close_time
                self.state.pending_expansion_mode = "same_direction"
                self._event(close_time, "probe_confirmed", close_fill, fields={"progress_r": progress_r})
            elif progress_r <= -float(self.state.config.failure_r):
                self._close_leg(self.state.probe, close_time, float(close_mid), spread, "probe_failure")
                if self.state.config.reverse_after_failure:
                    self.state.pending_expansion_mode = "reversal"
                    self.state.status = ProbeStatus.CONFIRMED
                    self.state.confirmation_time = close_time
                    self._event(close_time, "reversal_queued", close_fill, reason="probe_failure")
                else:
                    self._terminal(ProbeStatus.FAILED, close_time, "probe_failure")
            elif elapsed >= int(self.state.config.expiry_minutes):
                self._close_leg(self.state.probe, close_time, float(close_mid), spread, "probe_expiry")
                self._terminal(ProbeStatus.EXPIRED, close_time, "probe_expiry")

        max_end = self.state.candidate.observed_time + pd.Timedelta(
            minutes=int(self.state.candidate.max_hold_minutes)
        )
        if close_time >= max_end and self.state.open_legs:
            for leg in list(self.state.open_legs):
                self._close_leg(leg, close_time, float(close_mid), spread, "time_exit")
            self._terminal(ProbeStatus.CLOSED, close_time, "time_exit")
        elif not self.state.open_legs and not self.state.pending_expansion_mode and not self.state.is_terminal:
            final = ProbeStatus.FAILED if self.state.confirmation_time is None else ProbeStatus.CLOSED
            self._terminal(final, close_time, "all_legs_closed")
        return self.state

    def force_close(self, when: Any, mid: float, reason: str, spread_bps: Optional[float] = None) -> ProbeState:
        when = utc_timestamp(when)
        spread = self._spread(spread_bps)
        for leg in list(self.state.open_legs):
            self._close_leg(leg, when, float(mid), spread, reason)
        self.state.pending_expansion_mode = None
        self._terminal(ProbeStatus.CLOSED, when, reason)
        return self.state

    def _open_expansion(self, when: pd.Timestamp, mid: float, spread_bps: float) -> None:
        mode = str(self.state.pending_expansion_mode)
        self.state.pending_expansion_mode = None
        direction = self.state.candidate.direction if mode == "same_direction" else -self.state.candidate.direction
        fill = executable_price(mid, direction, spread_bps)
        if mode == "same_direction":
            if self.state.config.expansion_stop_mode == "probe_entry":
                stop = float(self.state.probe.entry_mid if self.state.probe is not None else self.state.candidate.reference_price)
            else:
                stop = float(self.state.candidate.stop_price)
            target = float(self.state.candidate.target_price)
        else:
            stop = float(mid - direction * self.state.candidate.stop_distance)
            target = float(mid + direction * 1.75 * self.state.candidate.stop_distance)
        risk_distance = abs(fill - stop)
        reward_distance = direction * (target - fill)
        stop_width_bps = risk_distance / max(abs(fill), 1e-12) * 10000.0
        modeled_cost_r = float(spread_bps) / max(stop_width_bps, 1e-12)
        reward_risk = reward_distance / max(risk_distance, 1e-12)
        if risk_distance <= 0 or reward_distance <= 0:
            self._event(when, "expansion_rejected", fill, reason="invalid_geometry")
            return
        if modeled_cost_r > float(self.state.config.max_expansion_cost_r):
            self._event(
                when, "expansion_rejected", fill, reason="cost_gate",
                fields={"modeled_cost_r": modeled_cost_r},
            )
            return
        if reward_risk < float(self.state.config.min_remaining_reward_risk):
            self._event(
                when, "expansion_rejected", fill, reason="reward_risk_gate",
                fields={"reward_risk": reward_risk},
            )
            return
        remaining = max(
            float(self.state.config.total_risk_budget) - float(self.state.config.probe_risk_fraction),
            0.0,
        )
        units = remaining / max(risk_distance, 1e-12)
        self.state.expansion = ProbeLeg(
            kind=mode,
            direction=direction,
            entry_time=utc_timestamp(when),
            entry_mid=float(mid),
            entry_fill=fill,
            units=units,
            stop_price=stop,
            target_price=target,
            entry_spread_cost_r=abs(fill - float(mid)) * units,
        )
        self.state.status = ProbeStatus.EXPANDED
        self._event(
            when,
            "expansion_opened",
            fill,
            fields={
                "mode": mode,
                "direction": direction,
                "units": units,
                "stop": stop,
                "target": target,
                "modeled_cost_r": modeled_cost_r,
                "reward_risk": reward_risk,
            },
        )

    def _bar_exit(
        self,
        leg: ProbeLeg,
        open_mid: float,
        high_mid: float,
        low_mid: float,
        spread_bps: float,
    ) -> tuple[str, float, float]:
        open_exit = executable_price(open_mid, -leg.direction, spread_bps)
        high_exit = executable_price(high_mid, -leg.direction, spread_bps)
        low_exit = executable_price(low_mid, -leg.direction, spread_bps)
        if leg.direction == 1:
            if low_exit <= leg.stop_price:
                fill = min(float(leg.stop_price), open_exit)
                gross_mid = (
                    open_mid
                    if open_exit < leg.stop_price
                    else mid_for_executable_price(fill, -leg.direction, spread_bps)
                )
                return "sl_hit", fill, gross_mid
            if high_exit >= leg.target_price:
                fill = float(leg.target_price)
                return "tp_hit", fill, mid_for_executable_price(fill, -leg.direction, spread_bps)
        else:
            if high_exit >= leg.stop_price:
                fill = max(float(leg.stop_price), open_exit)
                gross_mid = (
                    open_mid
                    if open_exit > leg.stop_price
                    else mid_for_executable_price(fill, -leg.direction, spread_bps)
                )
                return "sl_hit", fill, gross_mid
            if low_exit <= leg.target_price:
                fill = float(leg.target_price)
                return "tp_hit", fill, mid_for_executable_price(fill, -leg.direction, spread_bps)
        return "", 0.0, 0.0

    def _close_leg(
        self,
        leg: ProbeLeg,
        when: pd.Timestamp,
        exit_mid: float,
        spread_bps: float,
        reason: str,
        *,
        exit_price_is_fill: bool = False,
        gross_exit_mid: Optional[float] = None,
    ) -> None:
        if not leg.open:
            return
        exit_fill = (
            float(exit_mid)
            if exit_price_is_fill
            else executable_price(exit_mid, -leg.direction, spread_bps)
        )
        neutral_exit = float(exit_mid if gross_exit_mid is None else gross_exit_mid)
        gross = leg.direction * (neutral_exit - leg.entry_mid) * leg.units
        net = leg.direction * (exit_fill - leg.entry_fill) * leg.units
        leg.open = False
        leg.exit_time = utc_timestamp(when)
        leg.exit_mid = neutral_exit
        leg.exit_fill = float(exit_fill)
        leg.exit_reason = str(reason)
        leg.gross_r = float(gross)
        leg.net_r = float(net)
        leg.total_cost_r = float(gross - net)
        self._event(
            when,
            f"{leg.kind}_closed",
            exit_fill,
            reason=reason,
            fields={"gross_r": gross, "net_r": net, "cost_r": gross - net},
        )

    def _update_excursions(self, high_mid: float, low_mid: float, spread_bps: float) -> None:
        if self.state.probe is None:
            return
        direction = self.state.candidate.direction
        favorable_mid = high_mid if direction == 1 else low_mid
        adverse_mid = low_mid if direction == 1 else high_mid
        favorable_fill = executable_price(favorable_mid, -direction, spread_bps)
        adverse_fill = executable_price(adverse_mid, -direction, spread_bps)
        favorable = direction * (favorable_fill - self.state.probe.entry_fill)
        adverse = direction * (adverse_fill - self.state.probe.entry_fill)
        sd = self.state.candidate.stop_distance
        self.state.mfe_r = max(float(self.state.mfe_r), favorable / sd)
        self.state.mae_r = min(float(self.state.mae_r), adverse / sd)

    def _terminal(self, status: ProbeStatus, when: Any, reason: str) -> None:
        if self.state.terminal_time is not None and self.state.status in TERMINAL_STATUSES:
            return
        self.state.status = status
        self.state.terminal_time = utc_timestamp(when)
        self.state.pending_expansion_mode = None
        self._event(when, "candidate_terminal", reason=reason)

    def _event(
        self,
        when: Any,
        action: str,
        price: Optional[float] = None,
        reason: str = "",
        fields: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.record_events:
            return
        self.state.events.append(
            ProbeEvent(
                candidate_id=self.state.candidate.candidate_id,
                event_time=utc_timestamp(when),
                action=str(action),
                status=self.state.status.value,
                direction=int(self.state.candidate.direction),
                price=None if price is None else float(price),
                reason=str(reason),
                fields=fields or {},
            )
        )

    def _spread(self, value: Optional[float]) -> float:
        return float(self.state.config.spread_bps if value is None else value)


def executable_price(mid: float, trade_side: int, spread_bps: float) -> float:
    half = abs(float(spread_bps)) / 20000.0
    return float(mid) * (1.0 + (1 if int(trade_side) >= 0 else -1) * half)


def mid_for_executable_price(fill: float, trade_side: int, spread_bps: float) -> float:
    half = abs(float(spread_bps)) / 20000.0
    factor = 1.0 + (1 if int(trade_side) >= 0 else -1) * half
    return float(fill) / max(factor, 1e-12)


def first_executable_quote(
    quotes: pd.DataFrame,
    observed_time: Any,
    *,
    time_column: str = "_ts",
) -> Optional[pd.Series]:
    """Return the first quote at/after observation; never fill from a prior quote."""
    if quotes.empty or time_column not in quotes.columns:
        return None
    observed = utc_timestamp(observed_time)
    times = pd.to_datetime(quotes[time_column], utc=True, errors="coerce")
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    idx = int(np.searchsorted(time_ns, observed.value, side="left"))
    if idx >= len(quotes):
        return None
    row = quotes.iloc[idx]
    row_time = utc_timestamp(row[time_column])
    if row_time < observed:
        raise AssertionError("first executable quote preceded observation")
    return row


def _optional_timestamp(value: Any) -> Optional[pd.Timestamp]:
    return None if value in (None, "") else utc_timestamp(value)


def _leg_to_dict(leg: Optional[ProbeLeg]) -> Optional[dict[str, Any]]:
    return None if leg is None else json_ready(asdict(leg))


def _leg_from_dict(payload: Optional[dict[str, Any]]) -> Optional[ProbeLeg]:
    if payload is None:
        return None
    values = dict(payload)
    values["entry_time"] = utc_timestamp(values["entry_time"])
    values["exit_time"] = _optional_timestamp(values.get("exit_time"))
    return ProbeLeg(**values)


def _tail_streak(rows: list[ProbeMemoryRecord], predicate) -> int:
    count = 0
    for row in reversed(rows):
        if not predicate(row):
            break
        count += 1
    return count


def _hours_since(rows: list[ProbeMemoryRecord], decision: pd.Timestamp) -> float:
    if not rows:
        return 1e9
    return float(max((decision - utc_timestamp(rows[-1].close_time)).total_seconds() / 3600.0, 0.0))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return utc_timestamp(value).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
