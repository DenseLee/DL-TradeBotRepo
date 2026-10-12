"""
Volatility-targeted regime filter for BTC trend-following pipeline.

Replaces the broken L1B/L1C ExtraTrees models (AUC ~0.55) with rule-based
regime detection that combines OHLCV volatility with on-chain/derivatives
regime-state data (funding rates, OI, liquidations).

Per the institutional report [§14]:
  "BTCUSD should be treated as a regime-switching market by default.
   Models that do not explicitly condition on volatility, spread,
   liquidation, or funding-state regimes are likely to degrade sharply."

Regime labels:
  TRENDING  — normal volatility, neutral funding, safe to trade direction
  CHOPPY    — low volatility, mean-reverting, avoid trend-following
  STRESS    — high volatility, extreme funding/OI moves, liquidation cascade
  UNKNOWN   — insufficient data to classify

Usage:
  from models.trend.regime_filter import RegimeFilter
  rf = RegimeFilter()
  regimes = rf.classify(h1_bars, regime_state_df)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Regime constants ──────────────────────────────────────────────────────
REGIME_TRENDING = "TRENDING"
REGIME_CHOPPY = "CHOPPY"
REGIME_STRESS = "STRESS"
REGIME_UNKNOWN = "UNKNOWN"

ALL_REGIMES = [REGIME_TRENDING, REGIME_CHOPPY, REGIME_STRESS, REGIME_UNKNOWN]


@dataclass
class RegimeFilterConfig:
    """Configuration for the volatility-targeted regime filter."""

    # Volatility thresholds (percentile of lookback realized vol)
    vol_lookback_hours: int = 168        # 1 week
    vol_choppy_max_pct: float = 25.0     # below 25th pctile = CHOPPY
    vol_stress_min_pct: float = 85.0     # above 85th pctile = STRESS

    # Funding rate extremes (z-score)
    funding_stress_zscore: float = 2.5   # |z| > 2.5 = extreme funding

    # Liquidation extremes (z-score)
    liquidation_stress_zscore: float = 3.0  # z > 3.0 = cascade

    # OI delta extremes (z-score of 24h delta)
    oi_delta_stress_zscore: float = 2.5

    # Minimum bars needed for a valid classification
    min_lookback_bars: int = 48           # 2 days of hourly data

    # Persistence: require N consecutive bars in same regime to switch
    persistence_bars: int = 3

    # Allow trend-following during STRESS? (dangerous but sometimes highest alpha)
    allow_trading_in_stress: bool = False


class RegimeFilter:
    """Rule-based regime classifier for BTC H1 bars.

    Combines:
      1. Realized volatility percentile vs. 1-week lookback
      2. Funding rate extreme detection
      3. Liquidation cascade detection
      4. OI delta (position build-up or unwind)
    """

    def __init__(self, config: RegimeFilterConfig = None):
        self.config = config or RegimeFilterConfig()
        self._last_regime = REGIME_UNKNOWN
        self._persistence_counter = 0

    # ── Public API ────────────────────────────────────────────────────────

    def classify(
        self,
        h1_bars: pd.DataFrame,
        regime_state: pd.DataFrame = None,
        precompute_vol: bool = True,
    ) -> pd.Series:
        """Classify each H1 bar into TRENDING / CHOPPY / STRESS.

        Args:
            h1_bars: DataFrame with H1 OHLCV (must have 'close' column;
                     uses 'timestamp' column if available for datetime alignment)
            regime_state: Optional DataFrame from RegimeStateLoader with funding/OI/liq data
            precompute_vol: If True, compute realized vol from h1_bars

        Returns:
            pd.Series with regime labels, same index as h1_bars
        """
        # Normalize: use timestamp column as index if available, strip timezone
        if 'timestamp' in h1_bars.columns:
            h1_bars = h1_bars.set_index('timestamp')
            if hasattr(h1_bars.index, 'tz') and h1_bars.index.tz is not None:
                h1_bars.index = h1_bars.index.tz_localize(None)

        n = len(h1_bars)
        labels = np.full(n, REGIME_UNKNOWN, dtype=object)

        if n < self.config.min_lookback_bars:
            return pd.Series(labels, index=h1_bars.index)

        # ── 1. Volatility percentile ──────────────────────────────────
        if precompute_vol:
            vol = self._realized_volatility(h1_bars)
            vol_pct = self._rolling_percentile(vol, self.config.vol_lookback_hours)
        else:
            vol = pd.Series(0.0, index=h1_bars.index)
            vol_pct = pd.Series(50.0, index=h1_bars.index)

        # ── 2. Funding extreme ────────────────────────────────────────
        funding_z = None
        if regime_state is not None and "funding_zscore" in regime_state.columns:
            funding_z = self._align_series(regime_state["funding_zscore"], h1_bars.index)

        # ── 3. Liquidation cascade ────────────────────────────────────
        liq_z = None
        if regime_state is not None and "liquidation_zscore" in regime_state.columns:
            liq_z = self._align_series(regime_state["liquidation_zscore"], h1_bars.index)

        # ── 4. OI delta extreme ──────────────────────────────────────
        oi_z = None
        if regime_state is not None and "oi_zscore" in regime_state.columns:
            oi_z = self._align_series(regime_state["oi_zscore"], h1_bars.index)

        # ── Classify each bar ─────────────────────────────────────────
        cfg = self.config

        for i in range(cfg.min_lookback_bars, n):
            vp = vol_pct.iloc[i]

            # Stress check: high vol + extreme funding/liquidation/OI
            is_stress = vp >= cfg.vol_stress_min_pct

            if funding_z is not None and i < len(funding_z):
                fz = funding_z.iloc[i]
                if abs(fz) >= cfg.funding_stress_zscore:
                    is_stress = True

            if liq_z is not None and i < len(liq_z):
                lz = liq_z.iloc[i]
                if lz >= cfg.liquidation_stress_zscore:
                    is_stress = True

            if oi_z is not None and i < len(oi_z):
                oz = oi_z.iloc[i]
                if abs(oz) >= cfg.oi_delta_stress_zscore:
                    is_stress = True

            # Choppy check: low vol, no stress signals
            is_choppy = (vp <= cfg.vol_choppy_max_pct) and not is_stress

            # Trending: normal vol, no stress signals
            is_trending = not is_stress and not is_choppy

            if is_stress:
                labels[i] = REGIME_STRESS
            elif is_choppy:
                labels[i] = REGIME_CHOPPY
            elif is_trending:
                labels[i] = REGIME_TRENDING

        # ── Apply persistence filter ──────────────────────────────────
        labels = self._apply_persistence(labels)

        return pd.Series(labels, index=h1_bars.index)

    def trading_allowed(self, regime: str) -> bool:
        """Check if trading is allowed in this regime."""
        if regime == REGIME_UNKNOWN:
            return False
        if regime == REGIME_CHOPPY:
            return False  # trend-following doesn't work in chop
        if regime == REGIME_STRESS:
            return self.config.allow_trading_in_stress
        return True  # TRENDING

    def classify_single(self, bar: dict, recent_vol_pct: float,
                        funding_z: float = None, liq_z: float = None,
                        oi_z: float = None) -> str:
        """Classify a single bar (for live/streaming use)."""
        cfg = self.config
        is_stress = recent_vol_pct >= cfg.vol_stress_min_pct

        if funding_z is not None and abs(funding_z) >= cfg.funding_stress_zscore:
            is_stress = True
        if liq_z is not None and liq_z >= cfg.liquidation_stress_zscore:
            is_stress = True
        if oi_z is not None and abs(oi_z) >= cfg.oi_delta_stress_zscore:
            is_stress = True

        if is_stress:
            return REGIME_STRESS
        if recent_vol_pct <= cfg.vol_choppy_max_pct:
            return REGIME_CHOPPY
        return REGIME_TRENDING

    # ── Metrics / diagnostics ─────────────────────────────────────────────

    def regime_distribution(self, labels: pd.Series) -> Dict[str, float]:
        """Return percentage distribution of each regime."""
        counts = labels.value_counts()
        total = len(labels)
        return {r: round(counts.get(r, 0) / total * 100, 1) for r in ALL_REGIMES}

    def regime_quality_report(
        self, labels: pd.Series, h1_bars: pd.DataFrame
    ) -> dict:
        """Report on regime discrimination quality.

        Should show that TRENDING bars have higher forward |return|
        than CHOPPY bars, and STRESS bars have highest vol.
        """
        df = h1_bars.copy()
        df["regime"] = labels.values
        df["return_1h"] = df["close"].pct_change().shift(-1)
        df["return_4h"] = df["close"].pct_change(4).shift(-4)
        df["return_24h"] = df["close"].pct_change(24).shift(-24)

        report = {"distribution": self.regime_distribution(labels)}
        for r in [REGIME_TRENDING, REGIME_CHOPPY, REGIME_STRESS]:
            mask = df["regime"] == r
            if mask.sum() < 10:
                continue
            subset = df[mask]
            report[f"{r}_count"] = int(mask.sum())
            report[f"{r}_mean_abs_ret_1h"] = round(
                float(subset["return_1h"].abs().mean()) * 100, 4)
            report[f"{r}_mean_abs_ret_4h"] = round(
                float(subset["return_4h"].abs().mean()) * 100, 4)
            report[f"{r}_mean_abs_ret_24h"] = round(
                float(subset["return_24h"].abs().mean()) * 100, 4)

        return report

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _realized_volatility(h1_bars: pd.DataFrame, window: int = 24) -> pd.Series:
        """Compute annualized realized volatility from hourly returns."""
        returns = h1_bars["close"].pct_change()
        vol = returns.rolling(window).std() * np.sqrt(365 * 24)
        return vol.fillna(0.0)

    @staticmethod
    def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling percentile of each value within its lookback window."""
        result = pd.Series(0.0, index=series.index)
        values = series.values
        for i in range(window, len(values)):
            window_vals = values[i - window : i + 1]
            result.iloc[i] = (np.sum(window_vals <= values[i]) / len(window_vals)) * 100
        return result

    @staticmethod
    def _align_series(source: pd.Series, target_index: pd.Index) -> pd.Series:
        """Align a series to a target index using forward fill.

        Uses reindex with ffill for DatetimeIndex alignment, or
        simple index-based alignment for other index types.
        """
        if not isinstance(target_index, pd.DatetimeIndex):
            return pd.Series(0.0, index=target_index)

        # Ensure source has DatetimeIndex
        if not isinstance(source.index, pd.DatetimeIndex):
            return pd.Series(0.0, index=target_index)

        # Normalize timezones: strip tz from both
        src_idx = source.index.tz_localize(None) if hasattr(source.index, 'tz') and source.index.tz is not None else source.index
        tgt_idx = target_index.tz_localize(None) if hasattr(target_index, 'tz') and target_index.tz is not None else target_index

        src = source.copy()
        src.index = src_idx
        aligned = src.reindex(tgt_idx, method="ffill")
        aligned.index = target_index
        return aligned.fillna(0.0)

    def _apply_persistence(self, labels: np.ndarray) -> np.ndarray:
        """Require N consecutive bars in same regime before switching."""
        cfg = self.config
        if cfg.persistence_bars <= 1:
            return labels

        result = labels.copy()
        current = REGIME_UNKNOWN
        count = 0

        for i in range(len(labels)):
            if labels[i] == current:
                count += 1
            else:
                if count >= cfg.persistence_bars:
                    current = labels[i - 1]  # confirmed switch
                count = 1

            if count >= cfg.persistence_bars:
                result[i] = labels[i]
            else:
                result[i] = current if current != REGIME_UNKNOWN else labels[i]

        return result


# ── Quick smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _root)
    from data.sources.regime_state import RegimeStateLoader

    # Generate synthetic H1 bars
    idx = pd.date_range("2026-01-01", "2026-05-22", freq="1h")
    n = len(idx)
    np.random.seed(99)
    close = 90000 + np.cumsum(np.random.normal(0, 200, n))
    close = np.maximum(close, 70000)

    h1 = pd.DataFrame({
        "open": close - np.random.normal(0, 50, n),
        "high": close + np.abs(np.random.normal(0, 100, n)),
        "low": close - np.abs(np.random.normal(0, 100, n)),
        "close": close,
        "volume": np.random.exponential(100, n),
    }, index=idx)

    # Load regime state data (synthetic)
    loader = RegimeStateLoader()
    state = loader.load_range("2026-01-01", "2026-05-22")

    # Run regime filter
    rf = RegimeFilter()
    labels = rf.classify(h1, state)

    # Report
    print("=== Regime Distribution ===")
    for regime, pct in rf.regime_distribution(labels).items():
        print(f"  {regime}: {pct}%")

    report = rf.regime_quality_report(labels, h1)
    print("\n=== Regime Quality ===")
    for k, v in report.items():
        if "count" in k or "distribution" in k:
            continue
        print(f"  {k}: {v}")

    # Check: TRENDING should have higher abs return than CHOPPY
    if "TRENDING_mean_abs_ret_4h" in report and "CHOPPY_mean_abs_ret_4h" in report:
        t_ret = report["TRENDING_mean_abs_ret_4h"]
        c_ret = report["CHOPPY_mean_abs_ret_4h"]
        print(f"\n  TRENDING abs ret (4h): {t_ret}%")
        print(f"  CHOPPY abs ret (4h):   {c_ret}%")
        if t_ret > c_ret:
            print("  ✓ TRENDING has higher directional movement than CHOPPY")
        else:
            print("  ✗ Regime filter not discriminating properly")
