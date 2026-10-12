"""Live-safe spree expansion and leverage policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from models.trend.trade_spree import SpreePosition


@dataclass
class SpreeExpansionConfig:
    enabled: bool = False
    mode: str = "current_time_confirmation"
    min_lead_mfe_r: float = 1.0
    boost_lead_mfe_r: float = 1.5
    boost_size_multiplier: float = 1.5
    max_size_multiplier: float = 2.0
    top_quality_score: float = 0.75
    protected_realized_r: float = 3.0
    protected_total_r: float = 5.0
    protected_add_every_bars: int = 4
    max_positions_override: Optional[int] = None


@dataclass
class SpreeExpansionDecision:
    allow_add: bool
    size_multiplier: float
    max_positions_override: Optional[int]
    reason: str
    features: Dict[str, float]


class SpreeExpansionPolicy:
    """Controls pyramid additions using only state known at decision time."""

    def __init__(self, config: SpreeExpansionConfig | None = None):
        self.config = config or SpreeExpansionConfig()

    def evaluate(self, positions: List[SpreePosition], closed_pnls: List[float],
                 spree_r: float, sequence: int, bars_since_start: int,
                 entry_quality_score: float = 0.0) -> SpreeExpansionDecision:
        if not self.config.enabled:
            return SpreeExpansionDecision(True, 1.0, None, "", {})

        features = self._features(positions, closed_pnls, spree_r, entry_quality_score)
        mode = self.config.mode
        allow = True
        reason = mode
        multiplier = 1.0

        if mode == "current_time_confirmation":
            bridge_realized = not positions and bool(closed_pnls) and float(closed_pnls[0]) > 0
            allow = features["lead_mfe_r"] >= self.config.min_lead_mfe_r or bridge_realized
            reason = "lead_mfe_confirmed" if features["lead_mfe_r"] >= self.config.min_lead_mfe_r else (
                "first_trade_realized_bridge" if bridge_realized else "lead_mfe_unconfirmed"
            )
            if features["lead_mfe_r"] >= self.config.boost_lead_mfe_r or (bridge_realized and float(closed_pnls[0]) >= self.config.boost_lead_mfe_r):
                multiplier = self.config.boost_size_multiplier
        elif mode == "realized_confirmation":
            allow = bool(closed_pnls) and float(closed_pnls[0]) > 0
            reason = "first_trade_realized_win" if allow else "first_trade_not_realized_win"
        elif mode == "quality_score_confirmation":
            allow = entry_quality_score >= self.config.top_quality_score
            reason = "top_quality_score" if allow else "quality_score_too_low"
            if allow:
                multiplier = self.config.boost_size_multiplier
        elif mode == "protected_tail":
            protected = (
                features["realized_r"] >= self.config.protected_realized_r
                or features["total_open_realized_r"] >= self.config.protected_total_r
            )
            if protected and self.config.protected_add_every_bars > 1:
                allow = bars_since_start % self.config.protected_add_every_bars == 0
                reason = "protected_tail_throttled" if not allow else "protected_tail_allowed"
            else:
                allow = features["lead_mfe_r"] >= self.config.min_lead_mfe_r
                reason = "lead_mfe_confirmed" if allow else "lead_mfe_unconfirmed"
        else:
            allow = True
            reason = "unknown_mode_passthrough"

        multiplier = float(np.clip(multiplier, 0.0, self.config.max_size_multiplier))
        return SpreeExpansionDecision(
            allow_add=allow,
            size_multiplier=multiplier,
            max_positions_override=self.config.max_positions_override,
            reason=reason,
            features=features,
        )

    @staticmethod
    def _features(positions: List[SpreePosition], closed_pnls: List[float],
                  spree_r: float, entry_quality_score: float) -> Dict[str, float]:
        lead_mfe = max((float(getattr(pos, "mfe_r", 0.0)) for pos in positions), default=0.0)
        open_r = 0.0
        for pos in positions:
            sd = max(abs(float(pos.entry_price) - float(pos.sl_price)), 1e-9)
            mark = float(getattr(pos, "last_mark_price", pos.entry_price))
            if pos.direction == 1:
                open_r += (mark - pos.entry_price) / sd
            else:
                open_r += (pos.entry_price - mark) / sd
        return {
            "lead_mfe_r": float(lead_mfe),
            "realized_r": float(spree_r),
            "open_r": float(open_r),
            "total_open_realized_r": float(spree_r + open_r),
            "closed_count": float(len(closed_pnls)),
            "entry_quality_score": float(entry_quality_score),
        }
