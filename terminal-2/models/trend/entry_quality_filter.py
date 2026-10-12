"""Live-safe entry quality filter for voting/spree research."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class EntryQualityConfig:
    enabled: bool = False
    max_pre_entry_r: Optional[float] = 1.25
    max_pre_entry_block_r: Optional[float] = None
    min_pre_entry_h1_body_r: Optional[float] = None
    max_pre_entry_h1_body_r: Optional[float] = None
    min_pre_entry_m15_range_expansion: Optional[float] = None
    max_pre_entry_m15_range_expansion: Optional[float] = None
    min_h1_atr_percentile: Optional[float] = None
    max_h1_atr_percentile: Optional[float] = None
    h1_atr_percentile_lookback: int = 240
    max_h1_body_r: Optional[float] = 1.5
    require_m15_alignment: bool = True
    min_m15_range_expansion: Optional[float] = 0.75
    m15_range_lookback: int = 8
    m15_range_baseline: int = 32
    block_score_below: float = 0.0
    # Directional momentum gate — require minimum M15 directional momentum (in R) for first entries
    min_entry_directional_momentum_r: float = -99.0
    # M15 range expansion gate for first entries (separate from the scoring-based min_m15_range_expansion)
    min_entry_m15_range_expansion: float = 0.0
    active_months: Optional[Set[str]] = None
    active_quarters: Optional[Set[str]] = None


@dataclass
class EntryQualityDecision:
    allow_entry: bool
    quality_score: float
    block_reason: str
    features: Dict[str, float]


class EntryQualityFilter:
    """Scores only features available at the entry decision timestamp."""

    def __init__(self, config: EntryQualityConfig | None = None):
        self.config = config or EntryQualityConfig()

    def evaluate(self, h1: pd.DataFrame, m15: pd.DataFrame, h1_idx: int, m15_idx: int,
                 direction: int, price: float, stop_distance: float, ep: dict,
                 regime: str, event_allowed: bool) -> EntryQualityDecision:
        if not self.config.enabled:
            return EntryQualityDecision(True, 0.0, "", {})
        if not self._active_at(m15, m15_idx):
            return EntryQualityDecision(True, 0.0, "inactive_schedule", {})

        features = self._features(h1, m15, h1_idx, m15_idx, direction, price, stop_distance, ep, regime, event_allowed)
        reasons = []
        score = 0.0

        pre_entry_r = features["pre_entry_r"]
        h1_body_r = features["h1_body_r"]
        m15_alignment = features["m15_alignment"]
        range_expansion = features["m15_range_expansion"]
        h1_atr_pct = features["h1_atr_percentile"]

        if self.config.max_pre_entry_r is not None and pre_entry_r > self.config.max_pre_entry_r:
            pre_entry_gate_active = True
            if self.config.max_pre_entry_block_r is not None and pre_entry_r > self.config.max_pre_entry_block_r:
                pre_entry_gate_active = False
            if self.config.min_pre_entry_h1_body_r is not None and h1_body_r < self.config.min_pre_entry_h1_body_r:
                pre_entry_gate_active = False
            if self.config.max_pre_entry_h1_body_r is not None and h1_body_r > self.config.max_pre_entry_h1_body_r:
                pre_entry_gate_active = False
            if (
                self.config.min_pre_entry_m15_range_expansion is not None
                and range_expansion < self.config.min_pre_entry_m15_range_expansion
            ):
                pre_entry_gate_active = False
            if (
                self.config.max_pre_entry_m15_range_expansion is not None
                and range_expansion > self.config.max_pre_entry_m15_range_expansion
            ):
                pre_entry_gate_active = False
            if pre_entry_gate_active:
                reasons.append("pre_entry_exhausted")
                score -= 1.0
        elif pre_entry_r > 0:
            score += max(0.0, 1.0 - pre_entry_r / max(self.config.max_pre_entry_r or 1.0, 1e-9)) * 0.4

        if self.config.max_h1_body_r is not None and h1_body_r > self.config.max_h1_body_r:
            reasons.append("h1_body_extended")
            score -= 0.75
        elif h1_body_r > 0:
            score += max(0.0, 1.0 - h1_body_r / max(self.config.max_h1_body_r or 1.0, 1e-9)) * 0.25

        if self.config.require_m15_alignment and m15_alignment < 0:
            reasons.append("m15_conflict")
            score -= 1.0
        elif m15_alignment > 0:
            score += 0.35

        # Directional momentum gate (hard block) — require minimum directional momentum in R
        entry_momentum_r = features.get("entry_directional_momentum_r", -99.0)
        if (
            self.config.min_entry_directional_momentum_r > -98.0
            and entry_momentum_r < self.config.min_entry_directional_momentum_r
        ):
            reasons.append("low_entry_directional_momentum")
            score -= 1.0

        # M15 range expansion gate for first entries (hard block, separate threshold)
        if (
            self.config.min_entry_m15_range_expansion > 0.0
            and range_expansion < self.config.min_entry_m15_range_expansion
        ):
            reasons.append("low_entry_m15_range")
            score -= 1.0

        if self.config.min_m15_range_expansion is not None and range_expansion < self.config.min_m15_range_expansion:
            reasons.append("low_m15_range_expansion")
            score -= 0.75
        elif range_expansion >= 1.0:
            score += min(range_expansion, 2.0) * 0.15

        if self.config.min_h1_atr_percentile is not None and h1_atr_pct < self.config.min_h1_atr_percentile:
            reasons.append("low_h1_atr_percentile")
            score -= 0.75
        if self.config.max_h1_atr_percentile is not None and h1_atr_pct > self.config.max_h1_atr_percentile:
            reasons.append("high_h1_atr_percentile")
            score -= 0.75

        if features["vote_abs"] >= 3:
            score += 0.25
        if event_allowed:
            score += 0.05

        allow_entry = not reasons and score >= self.config.block_score_below
        return EntryQualityDecision(allow_entry, float(score), ",".join(reasons), features)

    def _active_at(self, m15: pd.DataFrame, m15_idx: int) -> bool:
        if self.config.active_months is None and self.config.active_quarters is None:
            return True
        if m15_idx < 0 or m15_idx >= len(m15):
            return False
        ts = pd.Timestamp(m15["timestamp"].iloc[m15_idx])
        if self.config.active_months is not None:
            month = f"{ts.year:04d}-{ts.month:02d}"
            if month in self.config.active_months:
                return True
        if self.config.active_quarters is not None:
            quarter = f"{ts.year:04d}Q{((ts.month - 1) // 3) + 1}"
            if quarter in self.config.active_quarters:
                return True
        return False

    def _features(self, h1: pd.DataFrame, m15: pd.DataFrame, h1_idx: int, m15_idx: int,
                  direction: int, price: float, stop_distance: float, ep: dict,
                  regime: str, event_allowed: bool) -> Dict[str, float]:
        sd = max(float(stop_distance), 1e-9)
        h1_open = float(h1["open"].iloc[h1_idx]) if 0 <= h1_idx < len(h1) else price
        h1_close = float(h1["close"].iloc[h1_idx]) if 0 <= h1_idx < len(h1) else price
        pre_entry_r = direction * (float(price) - h1_open) / sd
        h1_body_r = abs(h1_close - h1_open) / sd

        lookback = 3
        if m15_idx >= lookback:
            m15_open_now = float(m15["open"].iloc[m15_idx])
            m15_open_prev = float(m15["open"].iloc[m15_idx - lookback])
            m15_momentum = np.sign(m15_open_now - m15_open_prev)
        else:
            m15_momentum = 0.0
        m15_alignment = float(direction * m15_momentum)

        recent_start = max(0, m15_idx - self.config.m15_range_lookback)
        base_start = max(0, m15_idx - self.config.m15_range_baseline)
        recent = (m15["high"].iloc[recent_start:m15_idx] - m15["low"].iloc[recent_start:m15_idx]).astype(float)
        baseline = (m15["high"].iloc[base_start:m15_idx] - m15["low"].iloc[base_start:m15_idx]).astype(float)
        recent_mean = float(recent.mean()) if len(recent) else 0.0
        baseline_mean = float(baseline.mean()) if len(baseline) else 0.0
        range_expansion = recent_mean / max(baseline_mean, 1e-9)

        atr_pct = 0.5
        if "_entry_h1_atr_percentile" in h1.columns and 0 <= h1_idx < len(h1):
            atr_pct = float(h1["_entry_h1_atr_percentile"].iloc[h1_idx])
        elif 0 <= h1_idx < len(h1):
            h1_ranges = (h1["high"] - h1["low"]).astype(float)
            atr24 = h1_ranges.rolling(24, min_periods=1).mean()
            lookback = max(int(self.config.h1_atr_percentile_lookback), 1)
            start = max(0, h1_idx - lookback)
            hist = atr24.iloc[start:h1_idx].dropna().astype(float)
            if len(hist) >= 10:
                atr_pct = float((hist <= float(atr24.iloc[h1_idx])).mean())

        # Directional momentum in R-terms: current bar close vs prior bar close, in signal direction
        entry_momentum_r = -99.0
        if m15_idx >= 1:
            m15_close_now = float(m15["close"].iloc[m15_idx])
            m15_close_prev = float(m15["close"].iloc[m15_idx - 1])
            entry_momentum_r = direction * (m15_close_now - m15_close_prev) / sd

        votes = ep.get("votes", {})
        return {
            "pre_entry_r": float(pre_entry_r),
            "h1_body_r": float(h1_body_r),
            "m15_alignment": float(m15_alignment),
            "m15_range_expansion": float(range_expansion),
            "h1_atr_percentile": float(atr_pct),
            "entry_directional_momentum_r": float(entry_momentum_r),
            "vote_abs": float(abs(int(votes.get("sum", 0)))),
            "event_allowed": float(bool(event_allowed)),
            "is_stress": float(str(regime).upper() == "STRESS"),
        }
