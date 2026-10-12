"""Fade-pattern label generator — B+A hybrid (structural trigger + forward-outcome oracle).

Produces labels for failed-breakout / liquidity-sweep fade setups from M5 OHLC data.
All feature computation uses only data available at bar `t` (decision time).
Forward simulation uses bars t+1 onward — never trains on future data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FadeLabelConfig:
    """Parameters for structural fade-candidate detection and forward-outcome labeling."""
    # Candidate detection
    lookback_n: int = 24              # M5 bars for prior high/low window
    sweep_buffer_atr: float = 0.18   # ATR multiplier for sweep threshold
    wick_threshold: float = 0.45     # min wick / candle range for rejection
    extension_atr: float = 1.00      # ATR multiplier for extension from EMA
    ema_period: int = 50             # M5 bars for EMA
    atr_period: int = 24             # M5 bars for ATR

    # Forward-outcome simulation
    tp_r: float = 1.0                # take-profit in R
    sl_r: float = 1.0                # stop-loss in R
    max_hold_bars: int = 24          # max M5 bars to hold before timeout
    sl_buffer_atr: float = 0.50      # ATR multiplier for SL beyond sweep extreme
    min_risk_distance_atr: float = 0.5  # minimum risk distance in ATR
    cost_bps: float = 5.0            # spread cost in bps (one-way)

    # Acceptance
    min_label_r: float = 0.5         # minimum R for good_fade classification


@dataclass
class FadeLabel:
    """Single fade label produced at M5 bar `t`."""
    signal_bar: int                  # M5 bar index of candidate signal
    direction: int                   # 1 = long fade, -1 = short fade
    good_fade: int                   # 1 = good, 0 = bad, -1 = no candidate
    entry_price: float = 0.0
    exit_price: float = 0.0
    exit_bar: int = 0
    exit_reason: str = ""            # "tp", "sl", "timeout"
    gross_r: float = 0.0
    net_r: float = 0.0               # after costs
    features: Dict[str, float] = field(default_factory=dict)


class FadeLabeler:
    """Computes fade labels from M5 OHLC data.

    All feature computation at bar `t` uses data ≤ bar `t`.
    Forward simulation uses bars > `t` only.
    """

    def __init__(self, config: Optional[FadeLabelConfig] = None):
        self.config = config or FadeLabelConfig()

    def label_m5_bar(self, m5: pd.DataFrame, idx: int) -> FadeLabel:
        """Produce a fade label for completed M5 bar at `idx`.

        Returns FadeLabel with good_fade=-1 if no candidate detected.
        """
        n = self.config.lookback_n
        if idx < n + 1:
            return FadeLabel(signal_bar=idx, direction=0, good_fade=-1)

        features = self._compute_features(m5, idx)
        if features is None:
            return FadeLabel(signal_bar=idx, direction=0, good_fade=-1)

        # Check short and long candidates
        short_candidate = self._is_short_fade(features)
        long_candidate = self._is_long_fade(features)

        if not short_candidate and not long_candidate:
            return FadeLabel(signal_bar=idx, direction=0, good_fade=-1,
                             features=features)

        direction = -1 if short_candidate else 1
        sweep_high = features["sweep_high"] if short_candidate else None
        sweep_low = features["sweep_low"] if long_candidate else None

        # Forward-outcome simulation
        entry_price, exit_price, exit_bar, exit_reason, gross_r = self._simulate_outcome(
            m5, idx, direction, sweep_high, sweep_low, features
        )

        # Apply costs
        entry_cost = entry_price * self.config.cost_bps / 10000.0
        exit_cost = exit_price * self.config.cost_bps / 10000.0
        stop_distance = abs(entry_price - (sweep_high if direction == -1 else sweep_low)) if (sweep_high or sweep_low) else abs(entry_price * 0.01)
        cost_r = (entry_cost + exit_cost) / max(stop_distance, 1e-9)
        net_r = gross_r - cost_r

        good_fade = 1 if net_r >= self.config.min_label_r else 0

        return FadeLabel(
            signal_bar=idx,
            direction=direction,
            good_fade=good_fade,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_bar=exit_bar,
            exit_reason=exit_reason,
            gross_r=gross_r,
            net_r=net_r,
            features=features,
        )

    def label_range(self, m5: pd.DataFrame, start_idx: int = 0,
                    end_idx: Optional[int] = None) -> List[FadeLabel]:
        """Label a range of M5 bars. Returns list of FadeLabels (includes no_sample=-1)."""
        end = end_idx if end_idx is not None else len(m5)
        labels = []
        for idx in range(start_idx, end):
            label = self.label_m5_bar(m5, idx)
            labels.append(label)
        return labels

    # ── Feature computation ──────────────────────────────────────────────

    def _compute_features(self, m5: pd.DataFrame, idx: int) -> Optional[Dict[str, float]]:
        """Compute all derived features from data ≤ bar `idx`."""
        n = self.config.lookback_n
        atr_period = self.config.atr_period
        ema_period = self.config.ema_period

        if idx < max(n, atr_period, ema_period):
            return None

        high = m5["high"].values
        low = m5["low"].values
        open_ = m5["open"].values
        close = m5["close"].values

        # Prior N-bar high/low (EXCLUDES current bar `idx`)
        prior_high_n = float(np.max(high[idx - n : idx]))
        prior_low_n = float(np.min(low[idx - n : idx]))

        # ATR (rolling mean of high-low, includes current bar)
        ranges = np.abs(high[max(0, idx - atr_period + 1) : idx + 1] -
                        low[max(0, idx - atr_period + 1) : idx + 1])
        atr = float(np.mean(ranges)) if len(ranges) > 0 else 0.0

        # EMA(close) — standard EMA, includes current bar
        close_series = pd.Series(close[:idx + 1])
        ema = float(close_series.ewm(span=ema_period, adjust=False).mean().iloc[-1])

        # Candle anatomy
        candle_range = high[idx] - low[idx]
        upper_wick = high[idx] - max(open_[idx], close[idx])
        lower_wick = min(open_[idx], close[idx]) - low[idx]
        body = abs(close[idx] - open_[idx])

        return {
            "prior_high_n": prior_high_n,
            "prior_low_n": prior_low_n,
            "atr": atr,
            "ema": ema,
            "candle_range": float(candle_range),
            "upper_wick": float(upper_wick),
            "lower_wick": float(lower_wick),
            "body": float(body),
            "close": float(close[idx]),
            "high": float(high[idx]),
            "low": float(low[idx]),
            "open": float(open_[idx]),
            "idx": float(idx),
        }

    # ── Candidate detection ──────────────────────────────────────────────

    def _is_short_fade(self, f: Dict[str, float]) -> bool:
        """Short fade: price swept above prior high, then closed back inside with wick rejection."""
        cfg = self.config
        atr = max(f["atr"], 1e-9)
        candle_range = max(f["candle_range"], 1e-9)

        upside_sweep = f["high"] > f["prior_high_n"] + cfg.sweep_buffer_atr * atr
        rejection = f["close"] < f["prior_high_n"]
        wick = (f["upper_wick"] / candle_range) >= cfg.wick_threshold
        extension = f["close"] > f["ema"] + cfg.extension_atr * atr

        # Store sweep reference for SL placement
        f["sweep_high"] = f["high"]

        return upside_sweep and rejection and wick and extension

    def _is_long_fade(self, f: Dict[str, float]) -> bool:
        """Long fade: price swept below prior low, then closed back inside with wick rejection."""
        cfg = self.config
        atr = max(f["atr"], 1e-9)
        candle_range = max(f["candle_range"], 1e-9)

        downside_sweep = f["low"] < f["prior_low_n"] - cfg.sweep_buffer_atr * atr
        rejection = f["close"] > f["prior_low_n"]
        wick = (f["lower_wick"] / candle_range) >= cfg.wick_threshold
        extension = f["close"] < f["ema"] - cfg.extension_atr * atr

        f["sweep_low"] = f["low"]

        return downside_sweep and rejection and wick and extension

    # ── Forward-outcome simulation ───────────────────────────────────────

    def _simulate_outcome(
        self, m5: pd.DataFrame, idx: int, direction: int,
        sweep_high: Optional[float], sweep_low: Optional[float],
        features: Dict[str, float],
    ) -> Tuple[float, float, int, str, float]:
        """Simulate a fade trade entered at open[idx+1], tracking TP/SL bar-by-bar.

        Returns: (entry_price, exit_price, exit_bar, exit_reason, gross_r)
        """
        cfg = self.config
        entry_bar = idx + 1
        if entry_bar >= len(m5):
            return 0.0, 0.0, idx, "no_data", 0.0

        entry_price = float(m5["open"].iloc[entry_bar])
        atr = max(features["atr"], 1e-9)

        min_risk = cfg.min_risk_distance_atr * atr

        if direction == -1:  # Short fade
            sl_price = (sweep_high or features["high"]) + cfg.sl_buffer_atr * atr
            risk_distance = max(sl_price - entry_price, min_risk)
            sl_price = entry_price + risk_distance  # enforce minimum
            tp_price = entry_price - cfg.tp_r * risk_distance
        else:  # Long fade
            sl_price = (sweep_low or features["low"]) - cfg.sl_buffer_atr * atr
            risk_distance = max(entry_price - sl_price, min_risk)
            sl_price = entry_price - risk_distance  # enforce minimum
            tp_price = entry_price + cfg.tp_r * risk_distance

        if risk_distance <= 0:
            return entry_price, entry_price, entry_bar, "invalid_risk", 0.0

        max_bar = min(entry_bar + cfg.max_hold_bars, len(m5) - 1)

        for bar in range(entry_bar, max_bar + 1):
            bar_high = float(m5["high"].iloc[bar])
            bar_low = float(m5["low"].iloc[bar])

            if direction == -1:
                # Short: TP hit if low <= tp_price, SL hit if high >= sl_price
                if bar_low <= tp_price:
                    gross_r = (entry_price - tp_price) / risk_distance
                    return entry_price, tp_price, bar, "tp", gross_r
                if bar_high >= sl_price:
                    gross_r = (entry_price - sl_price) / risk_distance
                    return entry_price, sl_price, bar, "sl", gross_r
            else:
                # Long: TP hit if high >= tp_price, SL hit if low <= sl_price
                if bar_high >= tp_price:
                    gross_r = (tp_price - entry_price) / risk_distance
                    return entry_price, tp_price, bar, "tp", gross_r
                if bar_low <= sl_price:
                    gross_r = (sl_price - entry_price) / risk_distance
                    return entry_price, sl_price, bar, "sl", gross_r

        # Timeout — exit at close of max_hold bar
        exit_price = float(m5["close"].iloc[max_bar])
        gross_r = (entry_price - exit_price) / risk_distance if direction == -1 else (exit_price - entry_price) / risk_distance
        return entry_price, exit_price, max_bar, "timeout", gross_r


# ── Label statistics ─────────────────────────────────────────────────────

def label_summary(labels: List[FadeLabel]) -> Dict:
    """Compute aggregate statistics for a list of fade labels."""
    candidates = [l for l in labels if l.good_fade >= 0]
    goods = [l for l in candidates if l.good_fade == 1]
    bads = [l for l in candidates if l.good_fade == 0]

    total = len(candidates)
    good_count = len(goods)
    bad_count = len(bads)

    avg_r = np.mean([l.net_r for l in candidates]) if candidates else 0.0
    good_avg_r = np.mean([l.net_r for l in goods]) if goods else 0.0
    bad_avg_r = np.mean([l.net_r for l in bads]) if bads else 0.0

    gross_r_sum = sum(l.gross_r for l in candidates)
    net_r_sum = sum(l.net_r for l in candidates)

    tp_count = sum(1 for l in candidates if l.exit_reason == "tp")
    sl_count = sum(1 for l in candidates if l.exit_reason == "sl")
    timeout_count = sum(1 for l in candidates if l.exit_reason == "timeout")

    return {
        "total_candidates": total,
        "good_fades": good_count,
        "bad_fades": bad_count,
        "good_fade_rate": good_count / max(total, 1),
        "avg_net_r": round(avg_r, 4),
        "good_avg_net_r": round(good_avg_r, 4),
        "bad_avg_net_r": round(bad_avg_r, 4),
        "gross_r_sum": round(gross_r_sum, 2),
        "net_r_sum": round(net_r_sum, 2),
        "profit_factor": abs(sum(l.net_r for l in goods) / min(sum(l.net_r for l in bads), -1e-9)) if bads else 0.0,
        "tp_exits": tp_count,
        "sl_exits": sl_count,
        "timeout_exits": timeout_count,
    }
