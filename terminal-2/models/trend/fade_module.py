"""T9-Dormant Fade Module — auxiliary mean-reversion trades during T9 inactivity.

Activates only when T9 is dormant (no open spree, low entry count, non-STRESS regime).
Single-position fade trades with fixed R risk and per-dormancy-period loss cap.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.trend.fade_labels import FadeLabelConfig, FadeLabeler


@dataclass
class FadeModuleConfig:
    """Configuration for the fade trading module."""
    enabled: bool = True
    risk_per_trade_r: float = 0.5          # R risk per fade trade
    dormancy_cap_r: float = 3.0            # max fade loss per T9 dormancy period
    daily_cap_r: Optional[float] = None    # optional: max fade loss per day
    monthly_cap_r: Optional[float] = None  # optional: max fade loss per month
    cooldown_m5_bars: int = 48             # M5 bars to wait after a fade trade closes

    # Dormancy detection
    dormancy_entry_window_m15: int = 96    # M15 bars for low-entry-count check
    dormancy_entry_threshold: int = 2      # accepted T9 entries below this = dormant

    # Label / candidate detection config
    label_config: FadeLabelConfig = field(default_factory=FadeLabelConfig)


@dataclass
class FadePosition:
    """Open fade position."""
    direction: int                         # 1 = long, -1 = short
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: float
    tp_price: float
    max_hold_until: pd.Timestamp           # absolute time of max_hold expiry
    risk_distance: float                   # price distance from entry to SL
    risk_r: float                          # R risk allocated (e.g., 0.5)
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class FadeClose:
    """Closed fade trade."""
    direction: int
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str                       # "tp", "sl", "timeout"
    gross_r: float
    net_r: float                           # after costs
    risk_r: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class FadeDecision:
    """Output of fade module evaluation at a decision point."""
    enter: bool
    direction: int
    entry_price: float
    sl_price: float
    tp_price: float
    max_hold_bars: int
    risk_r: float
    features: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class FadeModule:
    """Manages fade trade lifecycle with dormancy budget tracking."""

    def __init__(self, config: Optional[FadeModuleConfig] = None):
        self.config = config or FadeModuleConfig()
        self.labeler = FadeLabeler(self.config.label_config)

        # State
        self._open_pos: Optional[FadePosition] = None
        self.dormancy_fade_pnl_r: float = 0.0       # running fade P&L this dormancy period
        self.daily_fade_pnl_r: float = 0.0
        self.monthly_fade_pnl_r: float = 0.0
        self.closed_trades: List[FadeClose] = []
        self.dormancy_active: bool = False

        # Dormancy tracking
        self._current_dormancy_start: Optional[pd.Timestamp] = None
        self._current_day: Optional[str] = None
        self._current_month: Optional[str] = None
        self._last_close_m5_idx: int = -10**9  # M5 bar index of last fade close

    # ── Dormancy lifecycle ────────────────────────────────────────────────

    def on_t9_activated(self):
        """Called when T9 opens a spree — reset dormancy budget."""
        self.dormancy_fade_pnl_r = 0.0
        self.dormancy_active = False
        self._current_dormancy_start = None

    def on_dormancy_start(self, ts: pd.Timestamp):
        """Called when T9 dormancy begins."""
        if not self.dormancy_active:
            self.dormancy_fade_pnl_r = 0.0
            self.dormancy_active = True
            self._current_dormancy_start = ts

    def is_dormancy_exhausted(self) -> bool:
        """True if fade dormancy budget is exhausted."""
        if not self.config.enabled:
            return True
        cap = self.config.dormancy_cap_r
        if self.dormancy_fade_pnl_r <= -cap:
            return True
        if self.config.daily_cap_r is not None and self.daily_fade_pnl_r <= -self.config.daily_cap_r:
            return True
        if self.config.monthly_cap_r is not None and self.monthly_fade_pnl_r <= -self.config.monthly_cap_r:
            return True
        return False

    def can_enter(self, m5_idx: int = 0) -> bool:
        """True if fade module can open a new position."""
        cooldown_ok = (m5_idx - self._last_close_m5_idx) >= self.config.cooldown_m5_bars
        return (
            self.config.enabled
            and self._open_pos is None
            and not self.is_dormancy_exhausted()
            and cooldown_ok
        )

    # ── Day/month tracking ────────────────────────────────────────────────

    def _update_calendar(self, ts: pd.Timestamp):
        """Reset daily/monthly counters on boundary crossing."""
        day_key = str(ts.date())
        month_key = f"{ts.year:04d}-{ts.month:02d}"
        if self._current_day != day_key:
            self.daily_fade_pnl_r = 0.0
            self._current_day = day_key
        if self._current_month != month_key:
            self.monthly_fade_pnl_r = 0.0
            self._current_month = month_key

    # ── Entry evaluation ──────────────────────────────────────────────────

    def evaluate(self, m5: pd.DataFrame, m5_idx: int,
                 entry_price: Optional[float] = None) -> FadeDecision:
        """Check for a fade candidate at M5 bar `m5_idx`.

        Args:
            m5: M5 OHLC DataFrame
            m5_idx: index of the completed M5 bar to evaluate
            entry_price: override entry price (if None, uses next M5 open)

        Returns:
            FadeDecision with enter=True if a valid fade setup is detected.
        """
        if not self.can_enter(m5_idx):
            return FadeDecision(enter=False, direction=0, entry_price=0.0,
                                sl_price=0.0, tp_price=0.0, max_hold_bars=0,
                                risk_r=0.0, reason="budget_exhausted_or_cooldown")

        label = self.labeler.label_m5_bar(m5, m5_idx)
        if label.good_fade < 0:
            return FadeDecision(enter=False, direction=0, entry_price=0.0,
                                sl_price=0.0, tp_price=0.0, max_hold_bars=0,
                                risk_r=0.0, reason="no_candidate")

        # Compute entry price first (before pricing calculations)
        if entry_price is None:
            next_idx = m5_idx + 1
            if next_idx >= len(m5):
                return FadeDecision(enter=False, direction=0, entry_price=0.0,
                                    sl_price=0.0, tp_price=0.0, max_hold_bars=0,
                                    risk_r=0.0, reason="no_next_bar")
            entry_price = float(m5["open"].iloc[next_idx])

        # We don't filter by good_fade=1/0 here — the structural candidate
        # exists regardless of forward outcome. The backtest will determine
        # whether the candidate leads to profitable trades.
        direction = label.direction
        features = label.features

        atr = max(features.get("atr", 0.0), 1e-9)
        cfg = self.config.label_config

        min_risk = cfg.min_risk_distance_atr * atr

        if direction == -1:  # Short fade
            sweep_high = features.get("sweep_high", features.get("high", 0))
            sl_price = sweep_high + cfg.sl_buffer_atr * atr
            risk_distance = max(sl_price - entry_price, min_risk)
            sl_price = entry_price + risk_distance
            tp_price = entry_price - cfg.tp_r * risk_distance
        else:  # Long fade
            sweep_low = features.get("sweep_low", features.get("low", 0))
            sl_price = sweep_low - cfg.sl_buffer_atr * atr
            risk_distance = max(entry_price - sl_price, min_risk)
            sl_price = entry_price - risk_distance
            tp_price = entry_price + cfg.tp_r * risk_distance

        if risk_distance <= 0:
            return FadeDecision(enter=False, direction=0, entry_price=0.0,
                                sl_price=0.0, tp_price=0.0, max_hold_bars=0,
                                risk_r=0.0, reason="invalid_risk")

        return FadeDecision(
            enter=True,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            max_hold_bars=cfg.max_hold_bars,
            risk_r=self.config.risk_per_trade_r,
            features=features,
            reason="fade_candidate",
        )

    # ── Position management ───────────────────────────────────────────────

    def open_position(self, decision: FadeDecision, ts: pd.Timestamp) -> FadePosition:
        """Open a fade position from a positive decision."""
        pos = FadePosition(
            direction=decision.direction,
            entry_price=decision.entry_price,
            entry_time=ts,
            sl_price=decision.sl_price,
            tp_price=decision.tp_price,
            max_hold_until=ts + pd.Timedelta(minutes=5 * decision.max_hold_bars),
            risk_distance=abs(decision.entry_price - decision.sl_price),
            risk_r=decision.risk_r,
            features=decision.features,
        )
        self._open_pos = pos
        if not self.dormancy_active:
            self.dormancy_active = True
            self._current_dormancy_start = ts
        return pos

    def check_exit(self, bar_high: float, bar_low: float, bar_close: float,
                   bar_ts: pd.Timestamp, cost_bps: float = 5.0,
                   m5_idx: int = -1) -> Optional[FadeClose]:
        """Check if an open fade position should close on this bar.

        Returns FadeClose if exit triggered, None otherwise.
        """
        pos = self._open_pos
        if pos is None:
            return None

        direction = pos.direction
        entry = pos.entry_price
        sl = pos.sl_price
        tp = pos.tp_price
        risk_dist = max(pos.risk_distance, 1e-9)

        exit_price = 0.0
        exit_reason = ""

        if direction == -1:  # Short
            if bar_low <= tp:
                exit_price = tp
                exit_reason = "tp"
            elif bar_high >= sl:
                exit_price = sl
                exit_reason = "sl"
        else:  # Long
            if bar_high >= tp:
                exit_price = tp
                exit_reason = "tp"
            elif bar_low <= sl:
                exit_price = sl
                exit_reason = "sl"

        # Timeout
        if not exit_reason and bar_ts >= pos.max_hold_until:
            exit_price = bar_close
            exit_reason = "timeout"

        if not exit_reason:
            return None

        # Compute R
        gross_r = (entry - exit_price) / risk_dist if direction == -1 else (exit_price - entry) / risk_dist
        entry_cost = entry * cost_bps / 10000.0
        exit_cost = exit_price * cost_bps / 10000.0
        cost_r = (entry_cost + exit_cost) / risk_dist
        net_r = gross_r - cost_r

        close = FadeClose(
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=bar_ts,
            exit_reason=exit_reason,
            gross_r=gross_r,
            net_r=net_r,
            risk_r=pos.risk_r,
            features=pos.features,
        )

        # Update budget
        self.dormancy_fade_pnl_r += net_r
        self.daily_fade_pnl_r += net_r
        self.monthly_fade_pnl_r += net_r
        self.closed_trades.append(close)
        self._open_pos = None
        self._last_close_m5_idx = m5_idx if m5_idx >= 0 else self._last_close_m5_idx

        return close

    # ── State queries ─────────────────────────────────────────────────────

    def has_open_position(self) -> bool:
        return self._open_pos is not None

    def get_state(self) -> Dict:
        return {
            "dormancy_active": self.dormancy_active,
            "dormancy_fade_pnl_r": round(self.dormancy_fade_pnl_r, 2),
            "open_position": self._open_pos is not None,
            "closed_trades": len(self.closed_trades),
            "dormancy_exhausted": self.is_dormancy_exhausted(),
        }
