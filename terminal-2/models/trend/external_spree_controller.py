"""Research-only external spree add-on controller.

T9 remains the first-entry trigger. This controller schedules add-ons from
decision-time spree state instead of requiring fresh T9 voting signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from models.trend.trade_spree import SpreePosition


@dataclass
class ExternalSpreeDecision:
    allow_add: bool
    size_multiplier: float
    max_positions_override: Optional[int]
    reason: str
    features: Dict[str, float]


@dataclass
class ExternalSpreeState:
    positions: List[SpreePosition]
    closed_pnls: List[float]
    realized_r: float
    sequence: int
    bars_since_start: int
    bars_since_last_add: int
    m15_range_expansion: float
    directional_momentum_r: float
    peak_spree_r: float
    stop_width_bps: float
    modeled_cost_r: float
    h1_atr_percentile: Optional[float] = None


@dataclass
class ExternalSpreeConfig:
    enabled: bool = False
    mode: str = "combined"
    disable_t9_addon_policy_when_external: bool = True
    danger_start_minutes: float = 120.0
    danger_end_minutes: float = 480.0
    late_tail_minutes: float = 960.0
    idle_expiry_minutes: float = 960.0
    min_stop_width_bps: float = 150.0
    late_tail_min_stop_width_bps: float = 200.0
    max_cost_r: float = 0.075
    min_open_r: float = 0.75
    danger_min_open_r: float = 1.5
    max_giveback_r: float = 1.0
    min_bars_between_adds: int = 1
    min_m15_range_expansion: float = 1.0
    min_directional_momentum_r: float = 0.0
    base_size_multiplier: float = 0.5
    strong_open_r: float = 1.5
    strong_size_multiplier: float = 1.0
    max_size_multiplier: float = 1.0
    max_positions_override: Optional[int] = None


class ExternalSpreeController:
    """Decide add-ons from live-safe expansion state."""

    def __init__(self, config: ExternalSpreeConfig | None = None) -> None:
        self.config = config or ExternalSpreeConfig()

    def evaluate(self, state: ExternalSpreeState) -> ExternalSpreeDecision:
        if not self.config.enabled:
            return ExternalSpreeDecision(False, 0.0, None, "external_disabled", {})

        features = self._features(state)
        mode = str(self.config.mode)
        allow = True
        reason = f"{mode}_allowed"

        if state.sequence <= 0:
            allow, reason = False, "no_seed"
        elif features["bars_since_last_add"] < float(self.config.min_bars_between_adds):
            allow, reason = False, "external_cooldown"
        elif features["modeled_cost_r"] > float(self.config.max_cost_r):
            allow, reason = False, "external_cost_r_too_high"
        elif features["stop_width_bps"] < float(self.config.min_stop_width_bps):
            allow, reason = False, "external_stop_width_too_narrow"
        elif features["confirmation_r"] < float(self.config.min_open_r):
            allow, reason = False, "external_open_r_unconfirmed"
        elif features["giveback_from_peak_r"] > float(self.config.max_giveback_r):
            allow, reason = False, "external_giveback_too_large"

        if allow and mode in {"time_guard", "combined", "strict"}:
            if features["in_danger_zone"] and features["confirmation_r"] < float(self.config.danger_min_open_r):
                allow, reason = False, "external_danger_zone_no_expansion"

        if allow and mode in {"tail_allowed", "combined", "strict"}:
            if features["late_tail"] and features["stop_width_bps"] < float(self.config.late_tail_min_stop_width_bps):
                allow, reason = False, "external_tail_stop_width_too_narrow"

        if allow and mode in {"state_confirmed", "combined", "strict"}:
            if features["m15_range_expansion"] < float(self.config.min_m15_range_expansion):
                allow, reason = False, "external_range_too_low"
            elif features["directional_momentum_r"] < float(self.config.min_directional_momentum_r):
                allow, reason = False, "external_momentum_too_low"

        if allow and mode == "strict":
            strict_width = max(float(self.config.min_stop_width_bps), 200.0)
            strict_danger_open = max(float(self.config.danger_min_open_r), 2.0)
            if features["stop_width_bps"] < strict_width:
                allow, reason = False, "external_strict_stop_width"
            elif features["in_danger_zone"] and features["confirmation_r"] < strict_danger_open:
                allow, reason = False, "external_strict_danger_open_r"

        multiplier = float(self.config.base_size_multiplier)
        if features["confirmation_r"] >= float(self.config.strong_open_r):
            multiplier = float(self.config.strong_size_multiplier)
        multiplier = float(np.clip(multiplier, 0.0, float(self.config.max_size_multiplier)))
        if not allow:
            multiplier = 0.0
        return ExternalSpreeDecision(
            allow_add=bool(allow),
            size_multiplier=multiplier,
            max_positions_override=self.config.max_positions_override,
            reason=reason,
            features=features,
        )

    def _features(self, state: ExternalSpreeState) -> Dict[str, float]:
        open_r = self._open_r(state.positions)
        lead_mfe = max((float(getattr(pos, "mfe_r", 0.0)) for pos in state.positions), default=0.0)
        total_r = float(state.realized_r) + open_r
        peak_r = max(float(state.peak_spree_r), total_r)
        elapsed_minutes = float(state.bars_since_start) * 15.0
        giveback = max(peak_r - total_r, 0.0)
        open_count = len(state.positions)
        confirmation_r = open_r if open_count > 0 else max(float(state.realized_r), 0.0)
        return {
            "open_r": float(open_r),
            "confirmation_r": float(confirmation_r),
            "lead_mfe_r": float(lead_mfe),
            "realized_r": float(state.realized_r),
            "total_open_realized_r": float(total_r),
            "peak_spree_r": float(peak_r),
            "giveback_from_peak_r": float(giveback),
            "elapsed_minutes": elapsed_minutes,
            "bars_since_start": float(state.bars_since_start),
            "bars_since_last_add": float(state.bars_since_last_add),
            "sequence": float(state.sequence),
            "open_count": float(open_count),
            "closed_count": float(len(state.closed_pnls)),
            "worst_case_loss_after_next_r": float(total_r - open_count - 1),
            "m15_range_expansion": float(state.m15_range_expansion),
            "directional_momentum_r": float(state.directional_momentum_r),
            "stop_width_bps": float(state.stop_width_bps),
            "modeled_cost_r": float(state.modeled_cost_r),
            "h1_atr_percentile": float(state.h1_atr_percentile or 0.0),
            "in_danger_zone": float(
                float(self.config.danger_start_minutes) <= elapsed_minutes <= float(self.config.danger_end_minutes)
            ),
            "late_tail": float(elapsed_minutes >= float(self.config.late_tail_minutes)),
        }

    @staticmethod
    def _open_r(positions: List[SpreePosition]) -> float:
        total = 0.0
        for pos in positions:
            stop_distance = max(abs(float(pos.entry_price) - float(pos.sl_price)), 1e-9)
            mark = float(getattr(pos, "last_mark_price", pos.entry_price))
            if int(pos.direction) == 1:
                total += (mark - float(pos.entry_price)) / stop_distance
            else:
                total += (float(pos.entry_price) - mark) / stop_distance
        return float(total)
