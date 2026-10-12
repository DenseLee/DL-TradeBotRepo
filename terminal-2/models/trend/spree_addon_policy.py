"""Signal-independent, risk-budgeted spree add-on policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from models.trend.trade_spree import SpreePosition


@dataclass
class SpreeAddonConfig:
    enabled: bool = False
    mode: str = "m15_confirmed"
    min_lead_mfe_r: float = 0.75
    min_total_r: float = 0.5
    min_m15_range_expansion: float = 0.75
    min_directional_momentum_r: Optional[float] = None
    cooldown_bars: int = 1
    base_size_multiplier: float = 0.5
    strong_mfe_r: float = 1.5
    strong_size_multiplier: float = 1.0
    micro_size_multiplier: float = 0.25
    max_size_multiplier: float = 1.0
    max_spree_risk_budget: float = 4.0
    protected_total_r: float = 5.0
    protected_add_every_bars: int = 4
    relax_range_after_total_r: Optional[float] = None
    # Relax momentum gate after spree maturity (mirrors range relaxation)
    relax_momentum_after_total_r: Optional[float] = None
    relaxed_min_directional_momentum_r: Optional[float] = None
    # Relax ALL add-on gates (range + momentum) after spree proves itself
    relax_addon_gates_after_total_r: Optional[float] = None
    max_positions_override: Optional[int] = None
    decay_after_total_r: Optional[float] = None
    decay_min_m15_range_expansion: Optional[float] = None
    decay_min_directional_momentum_r: Optional[float] = None
    decay_add_every_bars: Optional[int] = None
    decay_max_open_positions: Optional[int] = None
    giveback_throttle_after_peak_r: Optional[float] = None
    giveback_throttle_r: Optional[float] = None
    giveback_add_every_bars: Optional[int] = None
    giveback_max_open_positions: Optional[int] = None
    # Prove-first: block all add-ons until spree proves itself
    prove_first_min_total_r: float = 0.0
    prove_first_relax_after_total_r: Optional[float] = None
    prove_first_relaxed_min_total_r: Optional[float] = None
    # Regime-aware add-on policy
    regime_mode: str = "uniform"  # uniform | stress_only | stress_preferred
    block_open_r_min: Optional[float] = None
    block_open_r_max: Optional[float] = None
    block_peak_r_min: Optional[float] = None
    block_peak_r_max: Optional[float] = None
    block_giveback_r_min: Optional[float] = None
    block_giveback_r_max: Optional[float] = None
    block_lead_mfe_r_min: Optional[float] = None
    block_lead_mfe_r_max: Optional[float] = None
    block_m15_range_min: Optional[float] = None
    block_m15_range_max: Optional[float] = None
    block_h1_atr_pct_min: Optional[float] = None
    block_h1_atr_pct_max: Optional[float] = None
    scale_state_multiplier: Optional[float] = None
    scale_open_r_min: Optional[float] = None
    scale_open_r_max: Optional[float] = None
    scale_peak_r_min: Optional[float] = None
    scale_peak_r_max: Optional[float] = None
    scale_giveback_r_min: Optional[float] = None
    scale_giveback_r_max: Optional[float] = None
    scale_lead_mfe_r_min: Optional[float] = None
    scale_lead_mfe_r_max: Optional[float] = None
    scale_m15_range_min: Optional[float] = None
    scale_m15_range_max: Optional[float] = None
    scale_h1_atr_pct_min: Optional[float] = None
    scale_h1_atr_pct_max: Optional[float] = None


@dataclass
class SpreeAddonDecision:
    allow_add: bool
    size_multiplier: float
    max_positions_override: Optional[int]
    reason: str
    features: Dict[str, float]


class SpreeAddonPolicy:
    """Allows add-ons between raw bot signals using only decision-time state."""

    def __init__(self, config: SpreeAddonConfig | None = None):
        self.config = config or SpreeAddonConfig()

    def evaluate(
        self,
        positions: List[SpreePosition],
        closed_pnls: List[float],
        spree_r: float,
        sequence: int,
        bars_since_start: int,
        bars_since_last_add: int,
        m15_range_expansion: float,
        directional_momentum_r: float = 0.0,
        peak_spree_r: Optional[float] = None,
        h1_atr_percentile: Optional[float] = None,
        regime_label: str = "UNKNOWN",
    ) -> SpreeAddonDecision:
        if not self.config.enabled:
            return SpreeAddonDecision(False, 0.0, None, "addon_disabled", {})

        features = self._features(
            positions, closed_pnls, spree_r, bars_since_last_add, m15_range_expansion,
            directional_momentum_r,
            peak_spree_r,
            h1_atr_percentile,
        )
        mode = self.config.mode
        multiplier = self.config.base_size_multiplier

        if mode == "disabled":
            return self._decision(False, 0.0, "addon_disabled", features)

        # Regime-aware gating
        regime_allowed, regime_reason = self._check_regime(regime_label)
        features["regime_allowed"] = float(regime_allowed)
        if not regime_allowed:
            return self._decision(False, 0.0, regime_reason, features)

        # Prove-first: block add-ons until spree proves itself
        prove_ok, prove_reason = self._check_prove_first(features)
        features["prove_first_ok"] = float(prove_ok)
        if not prove_ok:
            return self._decision(False, 0.0, prove_reason, features)

        if mode == "m5_micro_confirmed":
            multiplier = self.config.micro_size_multiplier
        elif mode in {"m15_confirmed", "risk_budgeted"}:
            multiplier = self.config.base_size_multiplier
        else:
            return self._decision(False, 0.0, "unknown_addon_mode", features)

        if features["bars_since_last_add"] < self.config.cooldown_bars:
            return self._decision(False, multiplier, "addon_cooldown", features)
        realized_bridge = features["open_count"] == 0.0 and features["closed_count"] > 0.0 and features["realized_r"] > 0.0
        if features["lead_mfe_r"] < self.config.min_lead_mfe_r and not realized_bridge:
            return self._decision(False, multiplier, "lead_mfe_unconfirmed", features)
        if features["total_open_realized_r"] < self.config.min_total_r:
            return self._decision(False, multiplier, "spree_total_r_unconfirmed", features)
        # Gate relaxation: after spree proves itself, skip range + momentum checks
        gates_relaxed = (
            self.config.relax_addon_gates_after_total_r is not None
            and features["total_open_realized_r"] >= float(self.config.relax_addon_gates_after_total_r)
        )
        features["gates_relaxed"] = float(gates_relaxed)

        if not gates_relaxed:
            range_relaxed = (
                self.config.relax_range_after_total_r is not None
                and features["total_open_realized_r"] >= float(self.config.relax_range_after_total_r)
            )
            features["range_relaxed"] = float(range_relaxed)
            if features["m15_range_expansion"] < self.config.min_m15_range_expansion and not range_relaxed:
                return self._decision(False, multiplier, "range_expansion_too_low", features)

            # Momentum relaxation (mirrors range relaxation)
            momentum_relaxed = (
                self.config.relax_momentum_after_total_r is not None
                and self.config.relaxed_min_directional_momentum_r is not None
                and features["total_open_realized_r"] >= float(self.config.relax_momentum_after_total_r)
            )
            features["momentum_relaxed"] = float(momentum_relaxed)
            mom_threshold = (
                float(self.config.relaxed_min_directional_momentum_r) if momentum_relaxed
                else float(self.config.min_directional_momentum_r or -99.0)
            )
            if (
                self.config.min_directional_momentum_r is not None
                and features["directional_momentum_r"] < mom_threshold
            ):
                return self._decision(False, multiplier, "directional_momentum_too_low", features)

        if self._state_block_active(features):
            return self._decision(False, multiplier, "state_window_blocked", features)

        if features["lead_mfe_r"] >= self.config.strong_mfe_r:
            multiplier = self.config.strong_size_multiplier

        protected = features["total_open_realized_r"] >= self.config.protected_total_r
        if protected and self.config.protected_add_every_bars > 1:
            if bars_since_start % self.config.protected_add_every_bars != 0:
                return self._decision(False, multiplier, "protected_tail_throttle", features)

        if mode == "risk_budgeted":
            if features["worst_case_loss_after_next_r"] < -self.config.max_spree_risk_budget:
                return self._decision(False, multiplier, "spree_risk_budget_exhausted", features)

        state_scale_active = self._state_scale_active(features)
        features["state_scale_active"] = float(state_scale_active)
        if state_scale_active and self.config.scale_state_multiplier is not None:
            multiplier *= max(float(self.config.scale_state_multiplier), 0.0)

        max_positions_override = self.config.max_positions_override
        decay_active = (
            self.config.decay_after_total_r is not None
            and features["total_open_realized_r"] >= float(self.config.decay_after_total_r)
        )
        features["decay_active"] = float(decay_active)
        if decay_active:
            if (
                self.config.decay_min_m15_range_expansion is not None
                and features["m15_range_expansion"] < float(self.config.decay_min_m15_range_expansion)
            ):
                return self._decision(False, multiplier, "decay_range_too_low", features)
            if (
                self.config.decay_min_directional_momentum_r is not None
                and features["directional_momentum_r"] < float(self.config.decay_min_directional_momentum_r)
            ):
                return self._decision(False, multiplier, "decay_momentum_too_low", features)
            if (
                self.config.decay_add_every_bars is not None
                and int(self.config.decay_add_every_bars) > 1
                and features["bars_since_last_add"] < float(self.config.decay_add_every_bars)
            ):
                return self._decision(False, multiplier, "decay_add_cooldown", features)
            if self.config.decay_max_open_positions is not None:
                max_positions_override = int(self.config.decay_max_open_positions)

        giveback_throttle_active = self._giveback_throttle_active(features)
        features["giveback_throttle_active"] = float(giveback_throttle_active)
        if giveback_throttle_active:
            if (
                self.config.giveback_add_every_bars is not None
                and int(self.config.giveback_add_every_bars) > 1
                and features["bars_since_last_add"] < float(self.config.giveback_add_every_bars)
            ):
                return self._decision(False, multiplier, "giveback_add_cooldown", features)
            if self.config.giveback_max_open_positions is not None:
                max_positions_override = int(self.config.giveback_max_open_positions)

        multiplier = float(np.clip(multiplier, 0.0, self.config.max_size_multiplier))
        return self._decision(True, multiplier, f"{mode}_allowed", features, max_positions_override)

    def _decision(
        self, allow: bool, multiplier: float, reason: str, features: Dict[str, float],
        max_positions_override: Optional[int] = None,
    ) -> SpreeAddonDecision:
        return SpreeAddonDecision(
            allow_add=bool(allow),
            size_multiplier=float(np.clip(multiplier, 0.0, self.config.max_size_multiplier)),
            max_positions_override=max_positions_override if max_positions_override is not None else self.config.max_positions_override,
            reason=reason,
            features=features,
        )

    @staticmethod
    def _features(
        positions: List[SpreePosition],
        closed_pnls: List[float],
        spree_r: float,
        bars_since_last_add: int,
        m15_range_expansion: float,
        directional_momentum_r: float,
        peak_spree_r: Optional[float],
        h1_atr_percentile: Optional[float],
    ) -> Dict[str, float]:
        lead_mfe = max((float(getattr(pos, "mfe_r", 0.0)) for pos in positions), default=0.0)
        open_r = 0.0
        for pos in positions:
            sd = max(abs(float(pos.entry_price) - float(pos.sl_price)), 1e-9)
            mark = float(getattr(pos, "last_mark_price", pos.entry_price))
            if pos.direction == 1:
                open_r += (mark - float(pos.entry_price)) / sd
            else:
                open_r += (float(pos.entry_price) - mark) / sd
        worst_case_loss_now = float(spree_r) - float(len(positions))
        total_open_realized = float(spree_r + open_r)
        peak = max(float(peak_spree_r or 0.0), total_open_realized)
        return {
            "lead_mfe_r": float(lead_mfe),
            "realized_r": float(spree_r),
            "open_r": float(open_r),
            "total_open_realized_r": total_open_realized,
            "peak_spree_r": float(peak),
            "giveback_from_peak_r": float(max(0.0, peak - total_open_realized)),
            "closed_count": float(len(closed_pnls)),
            "open_count": float(len(positions)),
            "bars_since_last_add": float(bars_since_last_add),
            "m15_range_expansion": float(m15_range_expansion),
            "h1_atr_percentile": float(h1_atr_percentile or 0.0),
            "directional_momentum_r": float(directional_momentum_r),
            "worst_case_loss_now_r": worst_case_loss_now,
            "worst_case_loss_after_next_r": worst_case_loss_now - 1.0,
        }

    def _giveback_throttle_active(self, features: Dict[str, float]) -> bool:
        if (
            self.config.giveback_throttle_after_peak_r is None
            or self.config.giveback_throttle_r is None
        ):
            return False
        return (
            features["peak_spree_r"] >= float(self.config.giveback_throttle_after_peak_r)
            and features["giveback_from_peak_r"] >= float(self.config.giveback_throttle_r)
        )

    def _state_block_active(self, features: Dict[str, float]) -> bool:
        checks = [
            ("total_open_realized_r", self.config.block_open_r_min, self.config.block_open_r_max),
            ("peak_spree_r", self.config.block_peak_r_min, self.config.block_peak_r_max),
            ("giveback_from_peak_r", self.config.block_giveback_r_min, self.config.block_giveback_r_max),
            ("lead_mfe_r", self.config.block_lead_mfe_r_min, self.config.block_lead_mfe_r_max),
            ("m15_range_expansion", self.config.block_m15_range_min, self.config.block_m15_range_max),
            ("h1_atr_percentile", self.config.block_h1_atr_pct_min, self.config.block_h1_atr_pct_max),
        ]
        configured = False
        for feature, low, high in checks:
            if low is None and high is None:
                continue
            configured = True
            value = float(features[feature])
            if low is not None and value < float(low):
                return False
            if high is not None and value >= float(high):
                return False
        return configured

    def _state_scale_active(self, features: Dict[str, float]) -> bool:
        if self.config.scale_state_multiplier is None:
            return False
        checks = [
            ("total_open_realized_r", self.config.scale_open_r_min, self.config.scale_open_r_max),
            ("peak_spree_r", self.config.scale_peak_r_min, self.config.scale_peak_r_max),
            ("giveback_from_peak_r", self.config.scale_giveback_r_min, self.config.scale_giveback_r_max),
            ("lead_mfe_r", self.config.scale_lead_mfe_r_min, self.config.scale_lead_mfe_r_max),
            ("m15_range_expansion", self.config.scale_m15_range_min, self.config.scale_m15_range_max),
            ("h1_atr_percentile", self.config.scale_h1_atr_pct_min, self.config.scale_h1_atr_pct_max),
        ]
        configured = False
        for feature, low, high in checks:
            if low is None and high is None:
                continue
            configured = True
            value = float(features[feature])
            if low is not None and value < float(low):
                return False
            if high is not None and value >= float(high):
                return False
        return configured

    def _check_prove_first(self, features: Dict[str, float]) -> tuple:
        """Block add-ons until open+realized spree R proves the spree is viable."""
        if self.config.prove_first_min_total_r <= 0.0:
            return True, ""
        total_r = features["total_open_realized_r"]
        relaxed = (
            self.config.prove_first_relax_after_total_r is not None
            and self.config.prove_first_relaxed_min_total_r is not None
            and total_r >= float(self.config.prove_first_relax_after_total_r)
        )
        threshold = (
            float(self.config.prove_first_relaxed_min_total_r) if relaxed
            else self.config.prove_first_min_total_r
        )
        if total_r < threshold:
            return False, "prove_first_unmet"
        return True, ""

    def _check_regime(self, regime_label: str) -> tuple:
        """Regime-aware add-on gating. Returns (allowed, reason)."""
        mode = self.config.regime_mode
        if mode == "uniform":
            return True, ""
        regime = str(regime_label).upper()
        if mode == "stress_only":
            if regime == "STRESS":
                return True, ""
            return False, "regime_blocked"
        if mode == "stress_preferred":
            if regime == "STRESS":
                return True, ""
            if regime == "TRENDING":
                return True, "regime_trending_restricted"
            return False, "regime_choppy_blocked"
        return True, ""
