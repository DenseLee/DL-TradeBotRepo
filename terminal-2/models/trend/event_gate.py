"""
Macro/event calendar gating for BTC trading.

Per the institutional report [§10, §20]: BTC bots should gate trading
around high-impact macroeconomic events (FOMC, CPI, NFP) because models
trained on "normal" conditions degrade sharply during these windows.

This module provides:
  - A built-in calendar of known high-impact events (2020-2026)
  - Gate logic to suppress trading ±N minutes around events
  - Forward-looking event schedule for live trading

Usage:
  from models.trend.event_gate import EventGate
  gate = EventGate()
  if gate.trading_allowed(timestamp):
      # proceed with trade
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd


# ── High-impact US economic events ────────────────────────────────────────
# Format: (date, time_utc, event_name, pre_window_min, post_window_min)
# pre_window: minutes BEFORE event to suppress trading
# post_window: minutes AFTER event to suppress trading

HIGH_IMPACT_EVENTS = [
    # ── 2024 ──
    ("2024-01-31", "19:00", "FOMC Rate Decision", 60, 120),
    ("2024-03-20", "18:00", "FOMC Rate Decision", 60, 120),
    ("2024-05-01", "18:00", "FOMC Rate Decision", 60, 120),
    ("2024-06-12", "18:00", "FOMC Rate Decision", 60, 120),
    ("2024-07-31", "18:00", "FOMC Rate Decision", 60, 120),
    ("2024-09-18", "18:00", "FOMC Rate Decision", 60, 120),
    ("2024-11-07", "19:00", "FOMC Rate Decision", 60, 120),
    ("2024-12-18", "19:00", "FOMC Rate Decision", 60, 120),
    # CPI releases (monthly, ~12:30 UTC)
    ("2024-01-11", "13:30", "CPI", 30, 60),
    ("2024-02-13", "13:30", "CPI", 30, 60),
    ("2024-03-12", "12:30", "CPI", 30, 60),
    ("2024-04-10", "12:30", "CPI", 30, 60),
    ("2024-05-15", "12:30", "CPI", 30, 60),
    ("2024-06-12", "12:30", "CPI", 30, 60),
    ("2024-07-11", "12:30", "CPI", 30, 60),
    ("2024-08-14", "12:30", "CPI", 30, 60),
    ("2024-09-11", "12:30", "CPI", 30, 60),
    ("2024-10-10", "12:30", "CPI", 30, 60),
    ("2024-11-13", "13:30", "CPI", 30, 60),
    ("2024-12-11", "13:30", "CPI", 30, 60),
    # NFP (monthly, first Friday, ~12:30 UTC)
    ("2024-01-05", "13:30", "NFP", 45, 90),
    ("2024-02-02", "13:30", "NFP", 45, 90),
    ("2024-03-08", "13:30", "NFP", 45, 90),
    ("2024-04-05", "12:30", "NFP", 45, 90),
    ("2024-05-03", "12:30", "NFP", 45, 90),
    ("2024-06-07", "12:30", "NFP", 45, 90),
    ("2024-07-05", "12:30", "NFP", 45, 90),
    ("2024-08-02", "12:30", "NFP", 45, 90),
    ("2024-09-06", "12:30", "NFP", 45, 90),
    ("2024-10-04", "12:30", "NFP", 45, 90),
    ("2024-11-01", "12:30", "NFP", 45, 90),
    ("2024-12-06", "13:30", "NFP", 45, 90),

    # ── 2025 ──
    ("2025-01-29", "19:00", "FOMC", 60, 120),
    ("2025-03-19", "18:00", "FOMC", 60, 120),
    ("2025-05-07", "18:00", "FOMC", 60, 120),
    ("2025-06-18", "18:00", "FOMC", 60, 120),
    ("2025-07-30", "18:00", "FOMC", 60, 120),
    ("2025-09-17", "18:00", "FOMC", 60, 120),
    ("2025-11-06", "19:00", "FOMC", 60, 120),
    ("2025-12-17", "19:00", "FOMC", 60, 120),
    # CPI 2025
    ("2025-01-15", "13:30", "CPI", 30, 60),
    ("2025-02-12", "13:30", "CPI", 30, 60),
    ("2025-03-12", "12:30", "CPI", 30, 60),
    ("2025-04-10", "12:30", "CPI", 30, 60),
    ("2025-05-14", "12:30", "CPI", 30, 60),
    ("2025-06-11", "12:30", "CPI", 30, 60),
    ("2025-07-15", "12:30", "CPI", 30, 60),
    ("2025-08-13", "12:30", "CPI", 30, 60),
    ("2025-09-11", "12:30", "CPI", 30, 60),
    ("2025-10-15", "12:30", "CPI", 30, 60),
    ("2025-11-13", "13:30", "CPI", 30, 60),
    ("2025-12-11", "13:30", "CPI", 30, 60),
    # NFP 2025
    ("2025-01-10", "13:30", "NFP", 45, 90),
    ("2025-02-07", "13:30", "NFP", 45, 90),
    ("2025-03-07", "13:30", "NFP", 45, 90),
    ("2025-04-04", "12:30", "NFP", 45, 90),
    ("2025-05-02", "12:30", "NFP", 45, 90),
    ("2025-06-06", "12:30", "NFP", 45, 90),
    ("2025-07-03", "12:30", "NFP", 45, 90),
    ("2025-08-01", "12:30", "NFP", 45, 90),
    ("2025-09-05", "12:30", "NFP", 45, 90),
    ("2025-10-03", "12:30", "NFP", 45, 90),
    ("2025-11-07", "13:30", "NFP", 45, 90),
    ("2025-12-05", "13:30", "NFP", 45, 90),

    # ── 2026 ──
    ("2026-01-28", "19:00", "FOMC", 60, 120),
    ("2026-03-18", "18:00", "FOMC", 60, 120),
    ("2026-04-29", "18:00", "FOMC", 60, 120),
    ("2026-06-17", "18:00", "FOMC", 60, 120),
    ("2026-07-29", "18:00", "FOMC", 60, 120),
    ("2026-09-16", "18:00", "FOMC", 60, 120),
    ("2026-10-28", "18:00", "FOMC", 60, 120),
    ("2026-12-09", "19:00", "FOMC", 60, 120),
    # CPI 2026
    ("2026-01-13", "13:30", "CPI", 30, 60),
    ("2026-02-13", "13:30", "CPI", 30, 60),
    ("2026-03-11", "12:30", "CPI", 30, 60),
    ("2026-04-10", "12:30", "CPI", 30, 60),
    ("2026-05-12", "12:30", "CPI", 30, 60),
    ("2026-06-10", "12:30", "CPI", 30, 60),
    ("2026-07-14", "12:30", "CPI", 30, 60),
    ("2026-08-12", "12:30", "CPI", 30, 60),
    ("2026-09-11", "12:30", "CPI", 30, 60),
    ("2026-10-14", "12:30", "CPI", 30, 60),
    ("2026-11-10", "13:30", "CPI", 30, 60),
    ("2026-12-10", "13:30", "CPI", 30, 60),
    # NFP 2026
    ("2026-01-09", "13:30", "NFP", 45, 90),
    ("2026-02-11", "13:30", "NFP", 45, 90),
    ("2026-03-06", "13:30", "NFP", 45, 90),
    ("2026-04-03", "12:30", "NFP", 45, 90),
    ("2026-05-08", "12:30", "NFP", 45, 90),
    ("2026-06-05", "12:30", "NFP", 45, 90),
    ("2026-07-02", "12:30", "NFP", 45, 90),
    ("2026-08-07", "12:30", "NFP", 45, 90),
    ("2026-09-04", "12:30", "NFP", 45, 90),
    ("2026-10-02", "12:30", "NFP", 45, 90),
    ("2026-11-06", "13:30", "NFP", 45, 90),
    ("2026-12-04", "13:30", "NFP", 45, 90),
]


@dataclass
class EventGateConfig:
    """Configuration for event gating."""

    # Enable/disable event gating
    enabled: bool = True

    # Default window around events (minutes before, minutes after)
    default_pre_minutes: int = 30
    default_post_minutes: int = 60

    # Which event types to gate
    gate_fomc: bool = True
    gate_cpi: bool = True
    gate_nfp: bool = True

    # Path to custom event calendar CSV (optional)
    custom_calendar_path: str = ""


class EventGate:
    """Pre-trade filter: suppresses trading around macroeconomic events.

    Usage:
        gate = EventGate()
        # In backtest loop:
        if not gate.trading_allowed(bar_timestamp):
            continue  # skip this bar
    """

    def __init__(self, config: EventGateConfig = None):
        self.config = config or EventGateConfig()
        self._events = self._build_event_index()
        self._gated_count = 0
        self._total_checks = 0

    def trading_allowed(self, timestamp: pd.Timestamp) -> bool:
        """Check if trading is allowed at this timestamp.

        Returns False if timestamp falls within a gated event window.
        """
        self._total_checks += 1

        if not self.config.enabled:
            return True

        if timestamp is None:
            return True

        # Check against all events
        for event_start, event_end, event_name in self._events:
            if event_start <= timestamp <= event_end:
                self._gated_count += 1
                return False

        return True

    def get_active_events(self, timestamp: pd.Timestamp) -> List[dict]:
        """Return list of events that are currently gating this timestamp."""
        active = []
        for event_start, event_end, event_name in self._events:
            if event_start <= timestamp <= event_end:
                active.append({
                    "name": event_name,
                    "start": event_start,
                    "end": event_end,
                })
        return active

    def gated_fraction(self) -> float:
        """Fraction of checks that were gated."""
        if self._total_checks == 0:
            return 0.0
        return self._gated_count / self._total_checks

    def event_count(self, start: str, end: str) -> int:
        """Count events in a date range."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        count = 0
        for es, ee, _ in self._events:
            if es <= end_ts and ee >= start_ts:
                count += 1
        return count

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_event_index(self) -> List[tuple]:
        """Build list of (start, end, name) tuples for all gated events."""
        events = []
        cfg = self.config

        for date_str, time_str, event_name, pre_min, post_min in HIGH_IMPACT_EVENTS:
            # Filter by event type
            if event_name.startswith("FOMC") and not cfg.gate_fomc:
                continue
            if event_name == "CPI" and not cfg.gate_cpi:
                continue
            if event_name == "NFP" and not cfg.gate_nfp:
                continue

            try:
                event_time = pd.Timestamp(f"{date_str}T{time_str}:00")
                start = event_time - pd.Timedelta(minutes=pre_min)
                end = event_time + pd.Timedelta(minutes=post_min)
                events.append((start, end, event_name))
            except ValueError:
                continue

        # Load custom calendar if provided
        if cfg.custom_calendar_path and os.path.exists(cfg.custom_calendar_path):
            custom = pd.read_csv(cfg.custom_calendar_path)
            for _, row in custom.iterrows():
                event_time = pd.Timestamp(row["timestamp"])
                pre = int(row.get("pre_minutes", cfg.default_pre_minutes))
                post = int(row.get("post_minutes", cfg.default_post_minutes))
                start = event_time - pd.Timedelta(minutes=pre)
                end = event_time + pd.Timedelta(minutes=post)
                events.append((start, end, row.get("event", "CUSTOM")))

        return sorted(events, key=lambda x: x[0])


# ── Quick smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    gate = EventGate()

    # Test: check a known FOMC day
    fomc_time = pd.Timestamp("2025-01-29T19:30:00")  # During FOMC window
    print(f"FOMC during window: {gate.trading_allowed(fomc_time)} (should be False)")

    normal_time = pd.Timestamp("2025-01-30T10:00:00")  # Day after
    print(f"Day after FOMC:    {gate.trading_allowed(normal_time)} (should be True)")

    nfp_time = pd.Timestamp("2025-02-07T14:00:00")
    print(f"NFP during window: {gate.trading_allowed(nfp_time)} (should be False)")

    print(f"\nTotal events loaded: {len(gate._events)}")
    print(f"Events in 2025: {gate.event_count('2025-01-01', '2025-12-31')}")
    print(f"Events in 2026 H1: {gate.event_count('2026-01-01', '2026-06-30')}")
