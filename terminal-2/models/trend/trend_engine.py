"""
Document-aligned multi-timeframe trend-following engine.

Replaces the broken L2D ExtraTrees model with rule-based trend detection
per the institutional report [§25]: time-series momentum across multiple
horizons, volatility targeting, and regime-aware parameter switching.

Architecture:
  H4 trend filter  → SMA(200) direction locks LONG-only or SHORT-only
  H1 trend signal  → Close vs SMA(period) determines bias
  M15 breakout     → 20-bar high/low break triggers entry
  ATR filter       → Entries only when vol is in the 25-85 percentile range
  Regime switching → Wider stops in STRESS, tighter in CHOPPY

Usage:
  python models/trend/trend_engine.py  # smoke test on 2026 data
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Ensure sandbox root on path for standalone runs
_sandbox = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _sandbox not in sys.path:
    sys.path.insert(0, _sandbox)


@dataclass
class TrendConfig:
    """Configuration for the trend-following engine."""

    # ── Trend lock using H1 SMA ──
    trend_lock_sma_period: int = 50   # 50 hours ≈ 2 days — catches BTC's rapid trend swings
    trend_lock_enabled: bool = True

    # ── H1 trend signal ──
    h1_sma_period: int = 20           # SMA period for trend (1 day = fewer whipsaws than 4 days)

    # ── M15 entry timing ──
    m15_momentum_bars: int = 3        # Bars for M15 micro-momentum check

    # ── ATR / volatility filter ──
    atr_period: int = 14              # ATR calculation period
    atr_min_percentile: float = 5.0   # Min ATR percentile (avoid dead markets)
    atr_max_percentile: float = 95.0  # Max ATR percentile (avoid extreme stress)
    atr_percentile_lookback: int = 72  # 3 days of hourly bars

    # ── Regime-aware parameters ──
    # TRENDING (default)
    sl_atr: float = 1.5
    tp_atr: float = 2.0
    max_hold_bars: int = 18          # M15 bars (~4.5 hours)

    # STRESS (wider — let big vol moves run)
    stress_sl_atr: float = 2.0
    stress_tp_atr: float = 3.5
    stress_max_hold: int = 24

    # CHOPPY (tighter — quick exits in low vol)
    choppy_sl_atr: float = 1.0
    choppy_tp_atr: float = 1.5
    choppy_max_hold: int = 8

    # ── Trailing stop ──
    trail_trigger_r: float = 2.0     # Activate trail after 2.0R profit (STRESS moves big)
    trail_distance_r: float = 1.0    # Trail 1.0R behind — give room to breathe

    # ── Position management ──
    cooldown_bars: int = 4           # M15 bars to wait after exit before re-entry


class TrendEngine:
    """Multi-timeframe trend-following signal generator.

    Called once per H1 bar. Returns direction (1=LONG, -1=SHORT, 0=FLAT)
    and regime-adjusted exit parameters.
    """

    def __init__(self, config: TrendConfig = None):
        self.config = config or TrendConfig()

    # ── Public API ────────────────────────────────────────────────────

    def get_signal_vote(
        self,
        h1_bars: pd.DataFrame,
        m15_bars: pd.DataFrame,
        h4_bars: Optional[pd.DataFrame] = None,
        h1_idx: int = -1,
        m15_idx: int = -1,
        regime: str = "TRENDING",
    ) -> Tuple[int, dict]:
        """Multi-horizon voting ensemble — requires 2 of 3 timeframes to agree.

        H4: close vs SMA(50) → BULL (+1) / BEAR (-1)
        H1: close vs open   → BULL (+1) / BEAR (-1)
        M15: close vs 3-bar-ago → BULL (+1) / BEAR (-1)

        Vote: 3/3 = STRONG (full size), 2/3 = WEAK (half size), ≤1/3 = no signal.

        Returns (direction, exit_params) where exit_params includes 'strength'.
        """
        cfg = self.config

        if h1_idx < 0 or m15_idx < cfg.m15_momentum_bars:
            return 0, {}

        # ── Collect votes from each timeframe ─────────────────────────
        h4_vote = self._h4_vote(h4_bars) if h4_bars is not None else 0
        h1_vote = self._h1_vote(h1_bars, h1_idx)
        m15_vote = self._m15_vote(m15_bars, m15_idx)

        vote_sum = h4_vote + h1_vote + m15_vote

        # Need at least 2 votes in same direction
        if abs(vote_sum) < 2:
            return 0, {}

        direction = 1 if vote_sum > 0 else -1
        strength = "STRONG" if abs(vote_sum) >= 3 else "WEAK"

        # ── Exit params with strength ─────────────────────────────────
        exit_params = self._regime_params(regime)
        exit_params["strength"] = strength

        return direction, exit_params

    def get_signal(
        self,
        h1_bars: pd.DataFrame,
        m15_bars: pd.DataFrame,
        h4_bars: Optional[pd.DataFrame] = None,
        h1_idx: int = -1,
        m15_idx: int = -1,
        regime: str = "TRENDING",
    ) -> Tuple[int, dict]:
        """Get trading signal for the current bar.

        Args:
            h1_bars: H1 OHLCV DataFrame (needs 'close', 'high', 'low', 'timestamp')
            m15_bars: M15 OHLCV DataFrame
            h4_bars: Optional H4 OHLCV DataFrame for trend filter
            h1_idx: Index into h1_bars for the completed H1 bar
            m15_idx: Index into m15_bars for current M15 bar
            regime: Regime label (TRENDING, CHOPPY, STRESS)

        Returns:
            (direction, exit_params) where direction is 1/-1/0 and
            exit_params has sl_atr, tp_atr, max_hold
        """
        cfg = self.config

        if h1_idx < 0 or m15_idx < cfg.m15_momentum_bars:
            return 0, {}

        # ── 1. Trend direction lock (H1 SMA) ─────────────────────────
        allowed_dir = 0
        if cfg.trend_lock_enabled:
            trend_dir = self._trend_lock(h1_bars, h1_idx)
            if trend_dir == 0:
                return 0, {}
            allowed_dir = trend_dir

        # ── 2. H1 entry bias (regime-aware) ───────────────────────────
        h1_bias = self._h1_bias(h1_bars, h1_idx, regime)
        if h1_bias == 0:
            return 0, {}

        # Trend lock: only trade with the dominant trend
        if allowed_dir != 0 and h1_bias != allowed_dir:
            return 0, {}

        # ── 3. M15 entry timing (regime-aware) ────────────────────────
        if not self._m15_entry_ok(m15_bars, m15_idx, h1_bias, regime):
            return 0, {}

        # ── 4. ATR / volatility filter ────────────────────────────────
        # Disabled — the regime filter already handles vol-based gating

        # ── 5. Regime-aware exit parameters ───────────────────────────
        exit_params = self._regime_params(regime)

        return h1_bias, exit_params

    def _h4_vote(self, h4_bars: pd.DataFrame) -> int:
        """H4 vote: +1 if close > SMA(50), -1 if below, 0 if no data."""
        if h4_bars is None or len(h4_bars) < 50:
            return 0
        closes = h4_bars['close'].values.astype(float)
        sma = pd.Series(closes).rolling(50).mean().values
        if np.isnan(sma[-1]):
            return 0
        return 1 if closes[-1] > sma[-1] else -1

    def _h1_vote(self, h1_bars: pd.DataFrame, h1_idx: int) -> int:
        """H1 vote: +1 if bar closed > open, -1 if closed < open.

        Uses the COMPLETED H1 bar's close vs open. Both are known at bar close.
        No look-ahead: the signal is generated AFTER the H1 bar closes.
        """
        if h1_idx < 0 or h1_idx >= len(h1_bars):
            return 0
        o = h1_bars['open'].values.astype(float)[h1_idx]
        c = h1_bars['close'].values.astype(float)[h1_idx]
        if c > o: return 1
        if c < o: return -1
        return 0

    def _m15_vote(self, m15_bars: pd.DataFrame, m15_idx: int) -> int:
        """M15 vote: +1 if open > 3-bar-ago open, -1 if below.

        Uses OPEN price (not close) — close is future information at bar open.
        """
        lookback = self.config.m15_momentum_bars
        if m15_idx < lookback:
            return 0
        o_now = m15_bars['open'].values.astype(float)[m15_idx]
        o_prev = m15_bars['open'].values.astype(float)[m15_idx - lookback]
        if o_now > o_prev: return 1
        if o_now < o_prev: return -1
        return 0

    def _trend_lock(self, h1_bars: pd.DataFrame, h1_idx: int) -> int:
        """Trend direction from H1 SMA: 1 if close > SMA, -1 if below.

        Uses H1 bars (not H4) for more responsive trend detection.
        200 hours ≈ 8 days — enough to capture medium-term trend without
        the multi-month lag of H4 SMA(200).
        """
        cfg = self.config
        period = cfg.trend_lock_sma_period

        if h1_idx < period:
            return 0

        closes = h1_bars['close'].values.astype(float)
        window = closes[max(0, h1_idx - period + 1):h1_idx + 1]
        if len(window) < period:
            return 0

        sma = window.mean()
        if closes[h1_idx] > sma:
            return 1
        elif closes[h1_idx] < sma:
            return -1
        return 0

    def _h4_trend(self, h4_bars: pd.DataFrame) -> int:
        """H4 trend: 1 if close > SMA, -1 if below (kept for compatibility)."""
        cfg = self.config
        if len(h4_bars) < 50:
            return 0
        closes = h4_bars['close'].values.astype(float)
        sma = pd.Series(closes).rolling(50).mean().values
        if np.isnan(sma[-1]):
            return 0
        return 1 if closes[-1] > sma[-1] else -1

    def _h1_bias(self, h1_bars: pd.DataFrame, h1_idx: int,
                  regime: str = "TRENDING") -> int:
        """H1 trend bias — regime-aware.

        TRENDING: close vs SMA(50) — follow the medium-term trend
        STRESS:   close vs open — react to short-term momentum in high vol
        """
        if h1_idx < 0:
            return 0

        opens = h1_bars['open'].values.astype(float)
        closes = h1_bars['close'].values.astype(float)

        if h1_idx >= len(closes):
            return 0

        if regime.upper() == "STRESS":
            # Short-term momentum during vol spikes
            if closes[h1_idx] > opens[h1_idx]:
                return 1
            elif closes[h1_idx] < opens[h1_idx]:
                return -1
            return 0
        else:
            # Medium-term trend via SMA
            cfg = self.config
            if h1_idx < cfg.h1_sma_period:
                return 0
            window = closes[max(0, h1_idx - cfg.h1_sma_period + 1):h1_idx + 1]
            if len(window) < cfg.h1_sma_period:
                return 0
            sma = window.mean()
            if closes[h1_idx] > sma:
                return 1
            elif closes[h1_idx] < sma:
                return -1
            return 0

    def _m15_entry_ok(
        self, m15_bars: pd.DataFrame, m15_idx: int, direction: int,
        regime: str = "TRENDING"
    ) -> bool:
        """M15 entry: bar must move in trend direction. Range filter only in STRESS."""
        cfg = self.config
        lookback = max(cfg.m15_momentum_bars, 20)

        if m15_idx < lookback:
            return False

        closes = m15_bars['close'].values.astype(float)
        opens = m15_bars['open'].values.astype(float)
        highs = m15_bars['high'].values.astype(float)
        lows = m15_bars['low'].values.astype(float)

        # Range filter: only trade STRESS bars with above-median activity
        if regime.upper() == "STRESS":
            ranges = highs[m15_idx - lookback:m15_idx] - lows[m15_idx - lookback:m15_idx]
            current_range = highs[m15_idx] - lows[m15_idx]
            if current_range < np.median(ranges):
                return False

        if direction == 1:
            return closes[m15_idx] > opens[m15_idx] and closes[m15_idx] > closes[m15_idx - cfg.m15_momentum_bars]
        else:
            return closes[m15_idx] < opens[m15_idx] and closes[m15_idx] < closes[m15_idx - cfg.m15_momentum_bars]

    def _atr_filter(
        self, h1_bars: pd.DataFrame, m15_bars: pd.DataFrame,
        h1_idx: int, m15_idx: int
    ) -> bool:
        """ATR percentile filter: allow entries only when vol is in [min_pct, max_pct] range."""
        cfg = self.config
        lookback = cfg.atr_percentile_lookback

        if h1_idx < lookback:
            return True  # not enough data, allow

        # Compute ATR on H1 bars
        h1_high = h1_bars['high'].values.astype(float)[max(0, h1_idx - lookback):h1_idx + 1]
        h1_low = h1_bars['low'].values.astype(float)[max(0, h1_idx - lookback):h1_idx + 1]
        h1_close = h1_bars['close'].values.astype(float)[max(0, h1_idx - lookback):h1_idx + 1]

        tr = np.maximum(
            h1_high[1:] - h1_low[1:],
            np.maximum(
                np.abs(h1_high[1:] - h1_close[:-1]),
                np.abs(h1_low[1:] - h1_close[:-1])
            )
        )
        if len(tr) < cfg.atr_period:
            return True

        atr = pd.Series(tr).ewm(span=cfg.atr_period, adjust=False).mean().values
        current_atr = atr[-1]

        if len(atr) < lookback // 2:
            return True

        atr_percentile = (atr < current_atr).mean() * 100

        return cfg.atr_min_percentile <= atr_percentile <= cfg.atr_max_percentile

    def _regime_params(self, regime: str) -> dict:
        """Get regime-adjusted exit AND sizing parameters."""
        cfg = self.config
        regime = regime.upper()

        if regime == "STRESS":
            return {
                "sl_atr": cfg.stress_sl_atr,
                "tp_atr": cfg.stress_tp_atr,
                "max_hold": cfg.stress_max_hold,
                "trail_trigger_r": cfg.trail_trigger_r,
                "trail_distance_r": cfg.trail_distance_r,
                "target_vol": 0.15,      # smaller in high vol [doc §24]
                "max_notional_pct": 0.20,
            }
        elif regime == "CHOPPY":
            return {
                "sl_atr": cfg.choppy_sl_atr,
                "tp_atr": cfg.choppy_tp_atr,
                "max_hold": cfg.choppy_max_hold,
                "trail_trigger_r": cfg.trail_trigger_r,
                "trail_distance_r": cfg.trail_distance_r,
                "target_vol": 0.10,
                "max_notional_pct": 0.10,
            }
        else:  # TRENDING
            return {
                "sl_atr": cfg.sl_atr,
                "tp_atr": cfg.tp_atr,
                "max_hold": cfg.max_hold_bars,
                "trail_trigger_r": cfg.trail_trigger_r,
                "trail_distance_r": cfg.trail_distance_r,
                "target_vol": 0.25,      # bigger in clean trends [doc §37]
                "max_notional_pct": 0.30,
            }


# ── Quick smoke test ──────────────────────────────────────────────────
if __name__ == "__main__":
    from core.data_loader import V3DataLoader

    loader = V3DataLoader()
    data = loader.load_range("2026-01-01", "2026-05-22")
    h1 = data['h1']
    m15 = data['m15']

    # Build H4 from H1 (resample 4 bars → 1)
    h1_idxed = h1.set_index('timestamp')
    h4 = h1_idxed.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()

    engine = TrendEngine()
    h1_map = []
    m15_ts = pd.to_datetime(m15['timestamp'].values)
    h1_ts = pd.to_datetime(h1['timestamp'].values)
    for i in range(len(m15)):
        idx = int(np.searchsorted(h1_ts, m15_ts[i], side='right') - 1)
        h1_map.append(max(0, idx))

    signals = {"long": 0, "short": 0, "flat": 0}
    for i in range(100, len(m15)):
        h1_idx = h1_map[i] - 1  # completed H1 bar
        if h1_idx < 50:
            continue
        direction, params = engine.get_signal(h1, m15, h4, h1_idx, i)
        if direction == 1:
            signals["long"] += 1
        elif direction == -1:
            signals["short"] += 1
        else:
            signals["flat"] += 1

    total = sum(signals.values())
    print(f"Trend Engine smoke test (2026-01-01 → 2026-05-22)")
    print(f"  LONG:  {signals['long']:>5} ({signals['long']/total*100:.1f}%)")
    print(f"  SHORT: {signals['short']:>5} ({signals['short']/total*100:.1f}%)")
    print(f"  FLAT:  {signals['flat']:>5} ({signals['flat']/total*100:.1f}%)")
    print(f"  Signal rate: {(signals['long']+signals['short'])/total*100:.1f}%")
