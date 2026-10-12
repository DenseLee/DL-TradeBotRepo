"""
V3 5-Minute Data Loader

Reads the 5-min OHLCV CSV and provides:
  - Raw 5-min bars (used as intrabar data for oracle SL/TP simulation)
  - Resampled M15, M30, M60 bars for multi-resolution model input
  - Resampled H1 bars for expansion gate
  - Feature computation at each resolution

Data: btc_usd_dataset_London-Strategic-Edge.csv
Range: 2020-01-01 → 2026-06-09 (~677K bars, 288/day)
"""
import os
import numpy as np
import pandas as pd
from typing import Optional


class V3DataLoader:
    """
    Loads and caches the 5-min CSV, resamples to all required timeframes.

    Usage:
        loader = V3DataLoader()
        h1, m15, m5 = loader.load_range('2022-01-01', '2025-12-31')
        # h1: H1 OHLCV DataFrame
        # m15: M15 OHLCV DataFrame (aggregated from 5-min)
        # m5: 5-min OHLCV DataFrame (for intrabar oracle sim)
    """

    def __init__(self, csv_path: str = None):
        if csv_path is None:
            # core/data_loader.py → go up one level to sandbox root
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'V3_TrainingData',
                'btc_usd_dataset_London-Strategic-Edge (4).csv')
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None  # cached full 5-min data

    # ── Public API ──────────────────────────────────────────────────

    def load_range(self, start: str, end: str,
                   lookahead_m5_bars: int = 0) -> dict:
        """
        Load all timeframes for a date range.

        Returns dict with keys:
          - h1: H1 OHLCV DataFrame
          - m15: M15 OHLCV DataFrame
          - m30: M30 OHLCV DataFrame (aggregated 2×15min = 6×5min)
          - m60: M60 OHLCV DataFrame (aggregated 4×15min = 12×5min)
          - m5: 5-min OHLCV DataFrame, optionally with simulation lookahead
          - m5_features: 5-min OHLCV DataFrame clipped exactly to the range
        """
        since = self._as_utc(start)
        feature_until = self._as_utc(end) + pd.Timedelta(days=1)
        lookahead_m5_bars = max(0, int(lookahead_m5_bars))
        sim_until = feature_until + pd.Timedelta(minutes=5 * lookahead_m5_bars)

        # ── Load and slice 5-min data ──
        full = self._load_full()
        m5_features = full[
            (full['timestamp'] >= since) & (full['timestamp'] < feature_until)
        ].reset_index(drop=True)
        m5 = full[
            (full['timestamp'] >= since) & (full['timestamp'] < sim_until)
        ].reset_index(drop=True)

        if len(m5_features) == 0:
            raise ValueError(f"No 5-min data in range {start} → {end}")

        # ── Resample to higher timeframes ──
        h1 = self._resample(m5_features, '1h')
        m15 = self._resample(m5_features, '15min')
        m30 = self._resample(m5_features, '30min')
        m60 = self._resample(m5_features, '1h')  # same as H1 but can also do 60min

        return {
            'h1': h1,
            'm15': m15,
            'm30': m30,
            'm60': m60,
            'm5': m5,
            'm5_features': m5_features,
            'start': start,
            'end': end,
            'feature_end_exclusive': feature_until,
            'simulation_end_exclusive': sim_until,
            'lookahead_m5_bars': lookahead_m5_bars,
            'n_m5': len(m5),
            'n_m5_features': len(m5_features),
            'n_m15': len(m15),
            'n_h1': len(h1),
            'data_audit': self.audit_m5(m5_features),
            'simulation_data_audit': self.audit_m5(m5),
        }

    # ── Internal ────────────────────────────────────────────────────

    def _load_full(self) -> pd.DataFrame:
        """Load the full 5-min CSV (cached after first load)."""
        if self._df is not None:
            return self._df

        print(f"  Loading 5-min data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=5)
        self._df = df
        print(f"    {len(df):,} bars loaded ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
        return df

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample 5-min OHLCV to a higher timeframe."""
        resampled = df.set_index('timestamp').resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna().reset_index()
        resampled['close_time'] = resampled['timestamp'] + pd.Timedelta(rule)
        return resampled

    # ── Convenience ─────────────────────────────────────────────────

    def get_m5_slice(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Get raw 5-min bars between two timestamps (for intrabar simulation)."""
        df = self._load_full()
        start_ts = self._as_utc(start_ts)
        end_ts = self._as_utc(end_ts)
        mask = (df['close_time'] > start_ts) & (df['close_time'] <= end_ts)
        return df[mask].reset_index(drop=True)

    @staticmethod
    def _as_utc(value) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize('UTC')
        return ts.tz_convert('UTC')

    @staticmethod
    def audit_m5(df: pd.DataFrame) -> dict:
        """Return Layer 0 data-integrity checks for raw M5 candles."""
        if len(df) == 0:
            return {'rows': 0}

        ts = pd.to_datetime(df['timestamp'], utc=True)
        diffs = ts.diff().dropna()
        expected_step = pd.Timedelta(minutes=5)
        duplicate_count = int(ts.duplicated().sum())
        non_monotonic_count = int((diffs <= pd.Timedelta(0)).sum())
        gap_diffs = diffs[diffs > expected_step]
        short_diffs = diffs[(diffs > pd.Timedelta(0)) & (diffs < expected_step)]
        missing_bars = int(
            np.sum((gap_diffs / expected_step).astype(int) - 1)
        ) if len(gap_diffs) else 0

        o = df['open'].to_numpy(dtype=np.float64)
        h = df['high'].to_numpy(dtype=np.float64)
        l = df['low'].to_numpy(dtype=np.float64)
        c = df['close'].to_numpy(dtype=np.float64)
        v = df['volume'].to_numpy(dtype=np.float64)
        finite_ohlcv = (
            np.isfinite(o) & np.isfinite(h) & np.isfinite(l)
            & np.isfinite(c) & np.isfinite(v)
        )
        bad_ohlc = (
            (h < np.maximum(o, c))
            | (l > np.minimum(o, c))
            | (h < l)
        )

        expected_rows = int(((ts.iloc[-1] - ts.iloc[0]) / expected_step) + 1)
        return {
            'rows': int(len(df)),
            'first_timestamp': str(ts.iloc[0]),
            'last_timestamp': str(ts.iloc[-1]),
            'expected_rows_from_first_to_last': expected_rows,
            'monotonic_increasing': bool(ts.is_monotonic_increasing),
            'duplicate_timestamps': duplicate_count,
            'non_monotonic_steps': non_monotonic_count,
            'gap_count': int(len(gap_diffs)),
            'missing_5m_bars': missing_bars,
            'short_step_count': int(len(short_diffs)),
            'max_gap_minutes': round(
                float(gap_diffs.max() / pd.Timedelta(minutes=1)), 2
            ) if len(gap_diffs) else 0.0,
            'gap_examples': [
                {
                    'from': str(ts.iloc[int(i) - 1]),
                    'to': str(ts.iloc[int(i)]),
                    'minutes': round(float(d / pd.Timedelta(minutes=1)), 2),
                }
                for i, d in list(gap_diffs.head(5).items())
            ],
            'non_finite_ohlcv_rows': int((~finite_ohlcv).sum()),
            'bad_ohlc_rows': int(bad_ohlc.sum()),
            'negative_volume_rows': int((v < 0).sum()),
        }
