"""C1-SEQ causal sequence dataset and executable-path labels.

The builder consumes the append-only live probe files without importing T9 or
any execution code. Binance events are made available by ``local_receive_time``
and FBS labels use the first executable quote at or after each UTC minute
decision. July 2026 is rehearsal data only.

Examples:
  python .\models\trend\c1_sequence_dataset.py smoke
  python .\models\trend\c1_sequence_dataset.py build-day --date 2026-07-28
  python .\models\trend\c1_sequence_dataset.py replay --start 2026-07-20 --end 2026-07-31 --run-id july_rehearsal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "live_probe"
DEFAULT_BINANCE_DIR = RAW_ROOT / "binance_futures"
DEFAULT_SYNC_DIR = RAW_ROOT / "sync"
DEFAULT_DECISIONS = ROOT / "logs" / "shadow_decisions.jsonl"
DEFAULT_SHARD_DIR = ROOT / "data" / "features" / "microstructure" / "c1_sequence"
DEFAULT_FEATURE_SHARD_DIR = DEFAULT_SHARD_DIR / "features"
DEFAULT_LABEL_SHARD_DIR = DEFAULT_SHARD_DIR / "labels"
DEFAULT_REPORT_DIR = ROOT / "research" / "validation_runs"

WINDOW_SECONDS = (5, 15, 30, 60, 300, 900)
WINDOW_NAMES = {5: "5s", 15: "15s", 30: "30s", 60: "60s", 300: "5m", 900: "15m"}
REHEARSAL_START = pd.Timestamp("2026-07-20", tz="UTC")
REHEARSAL_END = pd.Timestamp("2026-07-31", tz="UTC")
STOP_WIDTH_BPS = 100.0
TARGET_R = 1.5
HORIZON_HOURS = 6.0
COST_BPS = (5.0, 10.0)
MAX_ENTRY_DELAY_SECONDS = 2.0
MAX_LABEL_GAP_SECONDS = 5.0
MAX_STATE_AGE_SECONDS = 2.0
MIN_WINDOW_COVERAGE = 0.95
RANDOM_SEED = 20260730


@dataclass(frozen=True)
class SequenceBuildConfig:
    stop_width_bps: float = STOP_WIDTH_BPS
    target_r: float = TARGET_R
    horizon_hours: float = HORIZON_HOURS
    cost_bps: tuple[float, ...] = COST_BPS
    max_entry_delay_seconds: float = MAX_ENTRY_DELAY_SECONDS
    max_label_gap_seconds: float = MAX_LABEL_GAP_SECONDS
    max_state_age_seconds: float = MAX_STATE_AGE_SECONDS
    min_window_coverage: float = MIN_WINDOW_COVERAGE


def utc_ts(value: Any) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    try:
        value = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def iter_jsonl(path: Path) -> Iterator[tuple[Optional[dict[str, Any]], bool]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield None, True
                continue
            yield row, False


def depth_summary(
    payload: dict[str, Any],
    previous: Optional[dict[str, dict[float, float]]] = None,
) -> tuple[dict[str, float], dict[str, dict[float, float]]]:
    bids = [(float(price), float(qty)) for price, qty in payload.get("b", [])[:10]]
    asks = [(float(price), float(qty)) for price, qty in payload.get("a", [])[:10]]
    bid_map = {price: qty for price, qty in bids if price > 0 and qty >= 0}
    ask_map = {price: qty for price, qty in asks if price > 0 and qty >= 0}
    bid_notional = sum(price * qty for price, qty in bid_map.items())
    ask_notional = sum(price * qty for price, qty in ask_map.items())
    total = bid_notional + ask_notional
    best_bid = max(bid_map, default=np.nan)
    best_ask = min(ask_map, default=np.nan)
    mid = (best_bid + best_ask) / 2.0 if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan

    def imbalance(levels: int) -> float:
        bid = sum(price * qty for price, qty in bids[:levels])
        ask = sum(price * qty for price, qty in asks[:levels])
        return (bid - ask) / (bid + ask) if bid + ask > 0 else np.nan

    def wall_distance(rows: list[tuple[float, float]], side: str) -> float:
        if not rows or not np.isfinite(mid) or mid <= 0:
            return np.nan
        wall_price, _ = max(rows, key=lambda item: item[0] * item[1])
        return ((mid - wall_price) if side == "bid" else (wall_price - mid)) / mid * 10000.0

    def book_delta(side: str) -> tuple[float, float]:
        old = (previous or {}).get(side, {})
        new = bid_map if side == "bid" else ask_map
        prices = set(old) | set(new)
        cancellation = 0.0
        replenishment = 0.0
        for price in prices:
            delta = (new.get(price, 0.0) - old.get(price, 0.0)) * price
            if delta < 0:
                cancellation += -delta
            else:
                replenishment += delta
        return cancellation, replenishment

    bid_cancel, bid_replenish = book_delta("bid")
    ask_cancel, ask_replenish = book_delta("ask")
    microprice = np.nan
    if bids and asks:
        bid_qty = bids[0][1]
        ask_qty = asks[0][1]
        if bid_qty + ask_qty > 0 and mid > 0:
            microprice = (asks[0][0] * bid_qty + bids[0][0] * ask_qty) / (bid_qty + ask_qty)
    return (
        {
            "binance_bid": best_bid,
            "binance_ask": best_ask,
            "binance_mid": mid,
            "binance_spread_bps": (
                (best_ask - best_bid) / mid * 10000.0
                if np.isfinite(best_bid) and np.isfinite(best_ask) and mid > 0
                else np.nan
            ),
            "book_imbalance_1": imbalance(1),
            "book_imbalance_5": imbalance(5),
            "book_imbalance_10": imbalance(10),
            "top_depth_usd": total,
            "bid_depth_usd": bid_notional,
            "ask_depth_usd": ask_notional,
            "bid_wall_distance_bps": wall_distance(bids, "bid"),
            "ask_wall_distance_bps": wall_distance(asks, "ask"),
            "microprice_deviation_bps": (
                (microprice - mid) / mid * 10000.0
                if np.isfinite(microprice) and np.isfinite(mid) and mid > 0
                else np.nan
            ),
            "bid_cancellation_usd": bid_cancel,
            "ask_cancellation_usd": ask_cancel,
            "bid_replenishment_usd": bid_replenish,
            "ask_replenishment_usd": ask_replenish,
        },
        {"bid": bid_map, "ask": ask_map},
    )


def read_binance_seconds(
    paths: Iterable[Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream raw Binance events into causal one-second state rows."""
    rows: list[dict[str, Any]] = []
    malformed = 0
    impossible_clock = 0
    event_counts: Counter[str] = Counter()
    state: dict[str, Any] = {}
    previous_depth: Optional[dict[str, dict[float, float]]] = None
    current_second: Optional[pd.Timestamp] = None
    acc: dict[str, Any] = {}

    def reset_acc() -> dict[str, Any]:
        return {
            "has_depth_event": 0.0,
            "has_book_event": 0.0,
            "has_trade_event": 0.0,
            "depth_event_count": 0.0,
            "book_flip_count": 0.0,
            "trade_count": 0.0,
            "buy_trade_notional": 0.0,
            "sell_trade_notional": 0.0,
            "trade_price_notional": 0.0,
            "trade_notional": 0.0,
            "long_liq_count": 0.0,
            "short_liq_count": 0.0,
            "long_liq_notional": 0.0,
            "short_liq_notional": 0.0,
            "bid_cancellation_usd": 0.0,
            "ask_cancellation_usd": 0.0,
            "bid_replenishment_usd": 0.0,
            "ask_replenishment_usd": 0.0,
        }

    def flush() -> None:
        if current_second is None:
            return
        row = {"second": current_second, **state, **acc}
        total_trade = float(row.get("buy_trade_notional", 0.0)) + float(row.get("sell_trade_notional", 0.0))
        row["signed_trade_notional"] = float(row.get("buy_trade_notional", 0.0)) - float(row.get("sell_trade_notional", 0.0))
        row["agg_trade_imbalance"] = row["signed_trade_notional"] / total_trade if total_trade > 0 else 0.0
        row["agg_trade_price"] = (
            float(row.get("trade_price_notional", 0.0)) / float(row.get("trade_notional", 0.0))
            if float(row.get("trade_notional", 0.0)) > 0
            else np.nan
        )
        liq_total = float(row.get("long_liq_notional", 0.0)) + float(row.get("short_liq_notional", 0.0))
        row["liq_imbalance"] = (
            (float(row.get("short_liq_notional", 0.0)) - float(row.get("long_liq_notional", 0.0))) / liq_total
            if liq_total > 0
            else 0.0
        )
        rows.append(row)

    acc = reset_acc()
    for path in sorted(paths):
        if not path.exists():
            continue
        for raw, bad in iter_jsonl(path):
            if bad:
                malformed += 1
                continue
            assert raw is not None
            received = utc_ts(raw.get("local_receive_time"))
            if received is None or received < start or received >= end:
                continue
            # New probe logs distinguish raw machine-clock skew from a
            # corrected impossible source timestamp.  Legacy rows retain the
            # original conservative flag.
            source_after_receive = raw.get(
                "source_after_receive_corrected", raw.get("source_after_receive")
            )
            if bool(source_after_receive):
                impossible_clock += 1
                continue
            second = received.floor("s")
            if current_second is None:
                current_second = second
            elif second != current_second:
                flush()
                current_second = second
                acc = reset_acc()
            event_type = str(raw.get("event_type") or "")
            payload = raw.get("payload") or {}
            event_counts[event_type] += 1
            if event_type == "depth":
                metrics, previous_depth = depth_summary(payload, previous_depth)
                previous_imbalance = state.get("book_imbalance_5")
                for key, value in metrics.items():
                    if key.endswith(("_cancellation_usd", "_replenishment_usd")):
                        acc[key] += float(value)
                    else:
                        state[key] = value
                current_imbalance = state.get("book_imbalance_5")
                if finite(previous_imbalance) is not None and finite(current_imbalance) is not None:
                    if float(previous_imbalance) * float(current_imbalance) < 0:
                        acc["book_flip_count"] += 1.0
                acc["has_depth_event"] = 1.0
                acc["depth_event_count"] += 1.0
            elif event_type == "book_ticker":
                bid = finite(payload.get("b"))
                ask = finite(payload.get("a"))
                if bid and ask and bid > 0 and ask > 0:
                    state["binance_bid"] = bid
                    state["binance_ask"] = ask
                    state["binance_mid"] = (bid + ask) / 2.0
                    state["binance_spread_bps"] = (ask - bid) / state["binance_mid"] * 10000.0
                    state["binance_bid_qty"] = finite(payload.get("B"))
                    state["binance_ask_qty"] = finite(payload.get("A"))
                acc["has_book_event"] = 1.0
            elif event_type == "agg_trade":
                price = finite(payload.get("p")) or 0.0
                qty = finite(payload.get("q")) or 0.0
                notional = price * qty
                if bool(payload.get("m")):
                    acc["sell_trade_notional"] += notional
                else:
                    acc["buy_trade_notional"] += notional
                acc["trade_count"] += 1.0
                acc["trade_notional"] += notional
                acc["trade_price_notional"] += price * notional
                acc["has_trade_event"] = 1.0
            elif event_type == "liquidation":
                order = payload.get("o") or {}
                price = finite(order.get("ap") or order.get("p")) or 0.0
                qty = finite(order.get("q")) or 0.0
                notional = price * qty
                # A forced SELL liquidates long exposure; BUY liquidates short.
                if str(order.get("S") or "").upper() == "SELL":
                    acc["long_liq_count"] += 1.0
                    acc["long_liq_notional"] += notional
                else:
                    acc["short_liq_count"] += 1.0
                    acc["short_liq_notional"] += notional
            elif event_type == "mark_price":
                state["mark_price"] = finite(payload.get("p"))
                state["funding_rate"] = finite(payload.get("r"))
    flush()
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("second").drop_duplicates("second", keep="last").reset_index(drop=True)
    return frame, {
        "raw_files": [str(path) for path in sorted(paths) if path.exists()],
        "malformed_rows": malformed,
        "source_after_receive_excluded": impossible_clock,
        "event_counts": dict(event_counts),
        "second_rows": int(len(frame)),
    }


def read_sync_quotes(
    paths: Iterable[Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    for path in sorted(paths):
        if not path.exists():
            continue
        for raw, bad in iter_jsonl(path):
            if bad:
                malformed += 1
                continue
            assert raw is not None
            received = utc_ts(raw.get("local_receive_time"))
            if received is None or received < start or received >= end:
                continue
            bid = finite(raw.get("fbs_bid"))
            ask = finite(raw.get("fbs_ask"))
            has_tick = bool(raw.get("has_fbs_tick")) and bid is not None and ask is not None and bid > 0 and ask > 0
            rows.append(
                {
                    "quote_time": received,
                    "fbs_bid": bid,
                    "fbs_ask": ask,
                    "fbs_mid": (bid + ask) / 2.0 if has_tick else np.nan,
                    "fbs_spread_bps": finite(raw.get("fbs_spread_bps")),
                    "binance_fbs_mid_divergence_bps": finite(raw.get("fbs_vs_binance_mid_divergence_bps")),
                    "sync_binance_mid": finite(raw.get("binance_mid")),
                    "has_fbs_tick": float(has_tick),
                    "sync_incomplete": float(bool(raw.get("incomplete"))),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("quote_time").drop_duplicates("quote_time", keep="last").reset_index(drop=True)
    return frame, {
        "raw_files": [str(path) for path in sorted(paths) if path.exists()],
        "malformed_rows": malformed,
        "quote_rows": int(len(frame)),
        "valid_tick_rate": float(frame["has_fbs_tick"].mean()) if len(frame) else 0.0,
    }


def _age_seconds(valid: pd.Series) -> pd.Series:
    index_series = pd.Series(valid.index, index=valid.index)
    last = index_series.where(valid.astype(bool)).ffill()
    return (index_series - last).dt.total_seconds()


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    return (series - series.shift(window - 1)) / max(window - 1, 1)


def _rolling_flips(series: pd.Series, window: int) -> pd.Series:
    signs = np.sign(series)
    flips = ((signs * signs.shift(1)) < 0).astype(float)
    return flips.rolling(window, min_periods=window).sum()


def _trailing_z(series: pd.Series, window: int = 1440, min_periods: int = 60) -> pd.Series:
    known = series.shift(1)
    mean = known.rolling(window, min_periods=min_periods).mean()
    std = known.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    return (series - mean) / std


def build_minute_features(
    binance_seconds: pd.DataFrame,
    quotes: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: SequenceBuildConfig,
) -> pd.DataFrame:
    """Build one causal feature row at each completed UTC minute."""
    if binance_seconds.empty or quotes.empty:
        return pd.DataFrame(
            {
                "feature_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "feature_max_receive_time": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        )
    grid_start = min(binance_seconds["second"].min(), quotes["quote_time"].min().floor("s"))
    grid_end = max(binance_seconds["second"].max(), quotes["quote_time"].max().floor("s"))
    index = pd.date_range(grid_start, grid_end, freq="1s", tz="UTC")
    sec = binance_seconds.set_index("second").reindex(index)
    quote_seconds = quotes.copy()
    quote_seconds["second"] = quote_seconds["quote_time"].dt.floor("s")
    quote_seconds = quote_seconds.sort_values("quote_time").groupby("second", as_index=False).last().set_index("second")
    sec = sec.join(quote_seconds.drop(columns=["quote_time"], errors="ignore"), how="left")

    depth_cols = [
        "book_imbalance_1", "book_imbalance_5", "book_imbalance_10",
        "top_depth_usd", "bid_depth_usd", "ask_depth_usd",
        "bid_wall_distance_bps", "ask_wall_distance_bps",
        "microprice_deviation_bps",
    ]
    market_cols = ["binance_bid", "binance_ask", "binance_mid", "binance_spread_bps", "mark_price", "funding_rate"]
    fbs_cols = ["fbs_bid", "fbs_ask", "fbs_mid", "fbs_spread_bps", "binance_fbs_mid_divergence_bps", "sync_binance_mid"]
    depth_valid = sec.get("has_depth_event", pd.Series(0.0, index=index)).fillna(0.0) > 0
    fbs_valid = sec.get("has_fbs_tick", pd.Series(0.0, index=index)).fillna(0.0) > 0
    depth_age = _age_seconds(depth_valid)
    fbs_age = _age_seconds(fbs_valid)
    for col in depth_cols + market_cols:
        if col in sec:
            sec[col] = sec[col].ffill().where(depth_age <= config.max_state_age_seconds)
    for col in fbs_cols:
        if col in sec:
            sec[col] = sec[col].ffill().where(fbs_age <= config.max_state_age_seconds)
    sec["depth_available"] = (depth_age <= config.max_state_age_seconds).astype(float)
    sec["flow_available"] = sec["depth_available"]
    sec["liquidation_available"] = sec["depth_available"]
    sec["execution_available"] = (fbs_age <= config.max_state_age_seconds).astype(float)

    flow_zero_cols = [
        "trade_count", "buy_trade_notional", "sell_trade_notional",
        "signed_trade_notional", "trade_notional", "book_flip_count",
        "bid_cancellation_usd", "ask_cancellation_usd",
        "bid_replenishment_usd", "ask_replenishment_usd",
        "long_liq_count", "short_liq_count",
        "long_liq_notional", "short_liq_notional",
    ]
    for col in flow_zero_cols:
        values = pd.to_numeric(sec.get(col, np.nan), errors="coerce")
        sec[col] = values.fillna(0.0).where(sec["depth_available"] > 0)
    sec["binance_return_1s"] = np.log(pd.to_numeric(sec["binance_mid"], errors="coerce")).diff()
    sec["fbs_return_1s"] = np.log(pd.to_numeric(sec["fbs_mid"], errors="coerce")).diff()

    output_columns: dict[str, pd.Series] = {}
    group_columns: dict[str, list[str]] = {"depth": [], "flow": [], "liquidation": [], "execution": []}
    for window in WINDOW_SECONDS:
        suffix = WINDOW_NAMES[window]
        min_periods = max(1, int(math.ceil(window * config.min_window_coverage)))
        depth_coverage = sec["depth_available"].rolling(window, min_periods=window).mean()
        flow_coverage = sec["flow_available"].rolling(window, min_periods=window).mean()
        liq_coverage = sec["liquidation_available"].rolling(window, min_periods=window).mean()
        exec_coverage = sec["execution_available"].rolling(window, min_periods=window).mean()
        output_columns[f"depth_coverage_{suffix}"] = depth_coverage
        output_columns[f"flow_coverage_{suffix}"] = flow_coverage
        output_columns[f"liquidation_coverage_{suffix}"] = liq_coverage
        output_columns[f"execution_coverage_{suffix}"] = exec_coverage

        for base in ("book_imbalance_1", "book_imbalance_5", "book_imbalance_10"):
            values = pd.to_numeric(sec[base], errors="coerce")
            cols = {
                f"{base}_mean_{suffix}": values.rolling(window, min_periods=min_periods).mean(),
                f"{base}_slope_{suffix}": _rolling_slope(values, window),
                f"{base}_persistence_{suffix}": np.sign(values).rolling(window, min_periods=min_periods).mean(),
            }
            for name, series in cols.items():
                output_columns[name] = series.where(depth_coverage >= config.min_window_coverage)
                group_columns["depth"].append(name)
        flip_name = f"book_flip_count_{suffix}"
        output_columns[flip_name] = _rolling_flips(sec["book_imbalance_5"], window).where(depth_coverage >= config.min_window_coverage)
        group_columns["depth"].append(flip_name)
        for base in (
            "bid_cancellation_usd", "ask_cancellation_usd",
            "bid_replenishment_usd", "ask_replenishment_usd",
        ):
            name = f"{base}_{suffix}"
            output_columns[name] = sec[base].rolling(window, min_periods=min_periods).sum().where(depth_coverage >= config.min_window_coverage)
            group_columns["depth"].append(name)
        for base in ("bid_wall_distance_bps", "ask_wall_distance_bps", "top_depth_usd", "microprice_deviation_bps"):
            values = pd.to_numeric(sec[base], errors="coerce")
            name = f"{base}_change_{suffix}"
            output_columns[name] = values.diff(window).where(depth_coverage >= config.min_window_coverage)
            group_columns["depth"].append(name)

        buy = sec["buy_trade_notional"].rolling(window, min_periods=min_periods).sum()
        sell = sec["sell_trade_notional"].rolling(window, min_periods=min_periods).sum()
        total = (buy + sell).replace(0.0, np.nan)
        signed = buy - sell
        flow_features = {
            f"signed_trade_notional_{suffix}": signed,
            f"agg_trade_imbalance_{suffix}": signed / total,
            f"trade_intensity_{suffix}": sec["trade_count"].rolling(window, min_periods=min_periods).sum(),
            f"trade_burstiness_{suffix}": (
                sec["trade_count"].rolling(window, min_periods=min_periods).std()
                / sec["trade_count"].rolling(window, min_periods=min_periods).mean().replace(0.0, np.nan)
            ),
            f"cvd_slope_{suffix}": signed / float(window),
            f"cvd_accel_{suffix}": signed - sec["signed_trade_notional"].shift(window).rolling(window, min_periods=min_periods).sum(),
            f"price_response_to_flow_{suffix}": sec["binance_return_1s"].rolling(window, min_periods=min_periods).sum() * np.sign(signed),
        }
        for name, series in flow_features.items():
            output_columns[name] = series.where(flow_coverage >= config.min_window_coverage)
            group_columns["flow"].append(name)

        long_liq = sec["long_liq_notional"].rolling(window, min_periods=min_periods).sum()
        short_liq = sec["short_liq_notional"].rolling(window, min_periods=min_periods).sum()
        liq_total = (long_liq + short_liq).replace(0.0, np.nan)
        liq_features = {
            f"long_liq_notional_{suffix}": long_liq,
            f"short_liq_notional_{suffix}": short_liq,
            f"liq_count_{suffix}": (
                sec["long_liq_count"] + sec["short_liq_count"]
            ).rolling(window, min_periods=min_periods).sum(),
            f"liq_imbalance_{suffix}": (short_liq - long_liq) / liq_total,
            f"liq_absorption_{suffix}": (
                sec["binance_return_1s"].rolling(window, min_periods=min_periods).sum()
                * np.sign(short_liq - long_liq)
            ),
        }
        for name, series in liq_features.items():
            output_columns[name] = series.where(liq_coverage >= config.min_window_coverage)
            group_columns["liquidation"].append(name)

        exec_features = {
            f"fbs_spread_bps_mean_{suffix}": sec["fbs_spread_bps"].rolling(window, min_periods=min_periods).mean(),
            f"fbs_spread_bps_max_{suffix}": sec["fbs_spread_bps"].rolling(window, min_periods=min_periods).max(),
            f"broker_divergence_mean_{suffix}": sec["binance_fbs_mid_divergence_bps"].rolling(window, min_periods=min_periods).mean(),
            f"broker_divergence_slope_{suffix}": _rolling_slope(sec["binance_fbs_mid_divergence_bps"], window),
            f"binance_return_{suffix}": sec["binance_return_1s"].rolling(window, min_periods=min_periods).sum(),
            f"fbs_return_{suffix}": sec["fbs_return_1s"].rolling(window, min_periods=min_periods).sum(),
            f"binance_volatility_{suffix}": sec["binance_return_1s"].rolling(window, min_periods=min_periods).std() * math.sqrt(window),
            f"fbs_volatility_{suffix}": sec["fbs_return_1s"].rolling(window, min_periods=min_periods).std() * math.sqrt(window),
            f"binance_lead_proxy_{suffix}": sec["binance_return_1s"].shift(1).rolling(window, min_periods=min_periods).corr(sec["fbs_return_1s"]),
            f"fbs_lead_proxy_{suffix}": sec["fbs_return_1s"].shift(1).rolling(window, min_periods=min_periods).corr(sec["binance_return_1s"]),
        }
        for name, series in exec_features.items():
            output_columns[name] = series.where(exec_coverage >= config.min_window_coverage)
            group_columns["execution"].append(name)

    # A decision at HH:MM sees all events received through HH:MM-1 second.
    output = pd.DataFrame(output_columns, index=index)
    minute = output.loc[output.index.second == 59].copy()
    minute.index = minute.index + pd.Timedelta(seconds=1)
    minute["feature_timestamp"] = minute.index
    minute["feature_max_receive_time"] = minute.index - pd.Timedelta(seconds=1)
    minute["utc_hour"] = minute.index.hour
    minute["day_of_week"] = minute.index.dayofweek
    minute["is_weekend"] = (minute.index.dayofweek >= 5).astype(float)
    for group, cols in group_columns.items():
        coverage_col = f"{group}_coverage_15m"
        minute[f"has_{group}_sequence"] = (minute[coverage_col] >= config.min_window_coverage).astype(float)
        minute[f"{group}_sequence_feature_count"] = minute[cols].notna().sum(axis=1).astype(float)

    normalise_cols = [
        "signed_trade_notional_60s", "trade_intensity_60s",
        "long_liq_notional_5m", "short_liq_notional_5m",
        "fbs_spread_bps_mean_60s", "top_depth_usd_change_60s",
    ]
    for col in normalise_cols:
        if col in minute:
            minute[f"{col}_trailing_z"] = _trailing_z(pd.to_numeric(minute[col], errors="coerce"))
    minute = minute[(minute.index >= start) & (minute.index < end)].reset_index(drop=True)
    return minute


def load_t9_state(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(columns=["state_time", "t9_open_count", "t9_action", "t9_reason", "t9_regime"])
    for raw, bad in iter_jsonl(path):
        if bad or raw is None:
            continue
        state_time = utc_ts(raw.get("probe_join_time") or raw.get("local_time"))
        if state_time is None:
            continue
        rows.append(
            {
                "state_time": state_time,
                "t9_open_count": finite(raw.get("spree_open_count")),
                "t9_action": raw.get("action"),
                "t9_reason": raw.get("reason"),
                "t9_regime": raw.get("regime"),
                "t9_direction": finite(raw.get("direction")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["state_time", "t9_open_count", "t9_action", "t9_reason", "t9_regime"])
    return pd.DataFrame(rows).sort_values("state_time").drop_duplicates("state_time", keep="last")


def join_t9_state(features: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if out.empty:
        if "feature_timestamp" not in out:
            out["feature_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        out["t9_state_known"] = pd.Series(dtype=float)
        out["t9_flat"] = pd.Series(dtype=float)
        out["t9_open_count"] = pd.Series(dtype=float)
        return out
    if state.empty:
        out["t9_state_known"] = 0.0
        out["t9_flat"] = np.nan
        out["t9_open_count"] = np.nan
        return out
    out = pd.merge_asof(
        out.sort_values("feature_timestamp"),
        state.sort_values("state_time"),
        left_on="feature_timestamp",
        right_on="state_time",
        direction="backward",
    )
    out["t9_state_known"] = out["t9_open_count"].notna().astype(float)
    out["t9_flat"] = (pd.to_numeric(out["t9_open_count"], errors="coerce") <= 0).astype(float)
    out["t9_state_age_seconds"] = (
        out["feature_timestamp"] - out["state_time"]
    ).dt.total_seconds()
    out["t9_decision_fresh"] = (
        out["t9_state_age_seconds"].between(0.0, 60.0, inclusive="both")
    ).astype(float)
    return out


def first_executable_quote_index(
    quote_times_ns: np.ndarray,
    decision: pd.Timestamp,
    max_delay_seconds: float,
) -> Optional[int]:
    idx = int(np.searchsorted(quote_times_ns, decision.value, side="left"))
    if idx >= len(quote_times_ns):
        return None
    delay = (int(quote_times_ns[idx]) - decision.value) / 1e9
    return idx if 0.0 <= delay <= max_delay_seconds else None


def _first_true(mask: np.ndarray) -> Optional[int]:
    positions = np.flatnonzero(mask)
    return int(positions[0]) if len(positions) else None


def label_side(
    path: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    side: int,
    config: SequenceBuildConfig,
) -> dict[str, Any]:
    stop_distance = entry_price * config.stop_width_bps / 10000.0
    stop_price = entry_price - side * stop_distance
    target_price = entry_price + side * config.target_r * stop_distance
    one_r_price = entry_price + side * stop_distance
    executable = pd.to_numeric(path["fbs_bid" if side == 1 else "fbs_ask"], errors="coerce").to_numpy(float)
    if side == 1:
        stop_mask = executable <= stop_price
        target_mask = executable >= target_price
        one_r_mask = executable >= one_r_price
        favorable = (executable - entry_price) / stop_distance
    else:
        stop_mask = executable >= stop_price
        target_mask = executable <= target_price
        one_r_mask = executable <= one_r_price
        favorable = (entry_price - executable) / stop_distance
    stop_idx = _first_true(stop_mask)
    target_idx = _first_true(target_mask)
    one_r_idx = _first_true(one_r_mask)
    same_observation = stop_idx is not None and target_idx is not None and stop_idx == target_idx
    if stop_idx is not None and (target_idx is None or stop_idx <= target_idx):
        exit_idx = stop_idx
        exit_reason = "sl_hit"
        exit_price = stop_price
    elif target_idx is not None:
        exit_idx = target_idx
        exit_reason = "tp_hit"
        exit_price = target_price
    else:
        exit_idx = len(path) - 1
        exit_reason = "time_exit"
        exit_price = float(executable[exit_idx])
    gross_r = side * (exit_price - entry_price) / stop_distance
    result = {
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_time": pd.Timestamp(path.iloc[exit_idx]["quote_time"]),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_r": gross_r,
        "hit_1r_before_stop": bool(one_r_idx is not None and (stop_idx is None or one_r_idx < stop_idx)),
        "hit_1_5r_before_stop": bool(target_idx is not None and (stop_idx is None or target_idx < stop_idx)),
        "full_stop_before_1r": bool(stop_idx is not None and (one_r_idx is None or stop_idx <= one_r_idx)),
        "mfe_r": float(np.nanmax(favorable)),
        "mae_r": float(np.nanmin(favorable)),
        "time_to_1r_minutes": (
            (pd.Timestamp(path.iloc[one_r_idx]["quote_time"]) - entry_time).total_seconds() / 60.0
            if one_r_idx is not None else np.nan
        ),
        "time_to_target_minutes": (
            (pd.Timestamp(path.iloc[target_idx]["quote_time"]) - entry_time).total_seconds() / 60.0
            if target_idx is not None else np.nan
        ),
        "time_to_stop_minutes": (
            (pd.Timestamp(path.iloc[stop_idx]["quote_time"]) - entry_time).total_seconds() / 60.0
            if stop_idx is not None else np.nan
        ),
        "same_observation_sl_tp_sl_first": bool(same_observation),
    }
    for cost in config.cost_bps:
        result[f"net_r_{cost_suffix(cost)}"] = gross_r - cost / config.stop_width_bps
    return result


def cost_suffix(cost: float) -> str:
    return f"{int(cost)}bps" if float(cost).is_integer() else f"{str(cost).replace('.', 'p')}bps"


def add_labels(
    features: pd.DataFrame,
    quotes: pd.DataFrame,
    config: SequenceBuildConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if features.empty or quotes.empty:
        return features.copy(), {"labeled_rows": 0, "skip_counts": {"empty_input": int(len(features))}}
    quote_frame = quotes[quotes["has_fbs_tick"] > 0].sort_values("quote_time").reset_index(drop=True)
    quote_times_ns = (
        pd.to_datetime(quote_frame["quote_time"], utc=True)
        .dt.as_unit("ns")
        .astype("int64")
        .to_numpy(np.int64)
    )
    records: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    for feature in features.to_dict("records"):
        decision = utc_ts(feature.get("feature_timestamp"))
        if decision is None:
            skips["missing_decision"] += 1
            continue
        entry_idx = first_executable_quote_index(quote_times_ns, decision, config.max_entry_delay_seconds)
        if entry_idx is None:
            skips["no_executable_quote_within_2s"] += 1
            continue
        horizon_end = decision + pd.Timedelta(hours=config.horizon_hours)
        end_idx = int(np.searchsorted(quote_times_ns, horizon_end.value, side="right"))
        path = quote_frame.iloc[entry_idx:end_idx].copy()
        if len(path) < 2:
            skips["insufficient_future_path"] += 1
            continue
        gaps = path["quote_time"].diff().dt.total_seconds().dropna()
        if len(gaps) and float(gaps.max()) > config.max_label_gap_seconds:
            skips["future_fbs_gap"] += 1
            continue
        entry = path.iloc[0]
        record = dict(feature)
        record["decision_time"] = decision
        record["entry_time"] = pd.Timestamp(entry["quote_time"])
        record["entry_delay_seconds"] = (record["entry_time"] - decision).total_seconds()
        record["label_horizon_end"] = horizon_end
        record["label_path_end"] = pd.Timestamp(path.iloc[-1]["quote_time"])
        record["label_path_rows"] = int(len(path))
        record["label_max_gap_seconds"] = float(gaps.max()) if len(gaps) else 0.0
        record["embedded_entry_spread_bps"] = finite(entry.get("fbs_spread_bps"))
        for side_name, side, price_col in (("long", 1, "fbs_ask"), ("short", -1, "fbs_bid")):
            entry_price = float(entry[price_col])
            labels = label_side(path, record["entry_time"], entry_price, side, config)
            for key, value in labels.items():
                record[f"{side_name}_{key}"] = value
        path_complete = (
            pd.Timestamp(path.iloc[-1]["quote_time"])
            >= horizon_end - pd.Timedelta(seconds=config.max_label_gap_seconds)
        )
        if not path_complete and any(
            record[f"{side}_exit_reason"] == "time_exit"
            for side in ("long", "short")
        ):
            skips["incomplete_six_hour_horizon"] += 1
            continue
        record["direction_agnostic_expansion"] = bool(
            record["long_hit_1_5r_before_stop"] or record["short_hit_1_5r_before_stop"]
        )
        record["feature_after_decision"] = bool(
            pd.Timestamp(record["feature_max_receive_time"]) > decision
        )
        records.append(record)
    labeled = pd.DataFrame(records)
    if not labeled.empty:
        labeled = assign_expansion_episodes(labeled, config)
    monotonic = True
    if not labeled.empty:
        for side in ("long", "short"):
            monotonic = monotonic and bool(
                (
                    pd.to_numeric(labeled[f"{side}_net_r_10bps"], errors="coerce")
                    <= pd.to_numeric(labeled[f"{side}_net_r_5bps"], errors="coerce") + 1e-12
                ).all()
            )
    return labeled, {
        "labeled_rows": int(len(labeled)),
        "skip_counts": dict(skips),
        "feature_after_decision": int(labeled.get("feature_after_decision", pd.Series(dtype=bool)).sum()),
        "cost_monotonic": monotonic,
        "direction_agnostic_expansion_rate": (
            float(labeled["direction_agnostic_expansion"].mean()) if len(labeled) else None
        ),
    }


def assign_expansion_episodes(frame: pd.DataFrame, config: SequenceBuildConfig) -> pd.DataFrame:
    out = frame.copy()
    out["long_episode_id"] = None
    out["short_episode_id"] = None
    out["long_episode_anchor"] = False
    out["short_episode_anchor"] = False
    out["long_early_usable_capture"] = False
    out["short_early_usable_capture"] = False
    horizon = pd.Timedelta(hours=config.horizon_hours)
    for side in ("long", "short"):
        winners = out[out[f"{side}_hit_1_5r_before_stop"]].sort_values("decision_time")
        episode_index = 0
        active_end: Optional[pd.Timestamp] = None
        active_id: Optional[str] = None
        onset_price: Optional[float] = None
        direction = 1 if side == "long" else -1
        for idx, row in winners.iterrows():
            decision = pd.Timestamp(row["decision_time"])
            if active_end is None or decision >= active_end:
                episode_index += 1
                active_id = f"{side}-{decision.strftime('%Y%m%dT%H%M%S')}-{episode_index:04d}"
                active_end = decision + horizon
                onset_price = float(row[f"{side}_entry_price"])
                out.at[idx, f"{side}_episode_anchor"] = True
            assert active_id is not None and onset_price is not None
            current_price = float(row[f"{side}_entry_price"])
            consumed_r = direction * (current_price - onset_price) / (onset_price * config.stop_width_bps / 10000.0)
            remaining_mfe = float(row[f"{side}_mfe_r"])
            out.at[idx, f"{side}_episode_id"] = active_id
            out.at[idx, f"{side}_consumed_r_from_episode_onset"] = consumed_r
            out.at[idx, f"{side}_remaining_mfe_r"] = remaining_mfe
            out.at[idx, f"{side}_early_usable_capture"] = bool(consumed_r <= 0.5 and remaining_mfe >= 1.0)
    return out


def day_files(directory: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = pd.date_range(start.floor("D"), (end - pd.Timedelta(microseconds=1)).floor("D"), freq="1D", tz="UTC")
    return [directory / f"{day.strftime('%Y-%m-%d')}.jsonl" for day in dates]


def frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    stable = frame.copy()
    stable = stable.reindex(sorted(stable.columns), axis=1)
    for col in stable.columns:
        if pd.api.types.is_datetime64_any_dtype(stable[col]):
            stable[col] = stable[col].astype(str)
    payload = stable.to_json(orient="records", date_format="iso", double_precision=12)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    os.replace(temp, path)


def build_day(
    day: pd.Timestamp,
    binance_dir: Path,
    sync_dir: Path,
    decisions_path: Path,
    output_dir: Path,
    config: SequenceBuildConfig,
    with_labels: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    day = day.floor("D")
    feature_path = output_dir / "features" / f"{day.strftime('%Y-%m-%d')}.parquet"
    label_path = output_dir / "labels" / f"{day.strftime('%Y-%m-%d')}.parquet"
    report_path = output_dir / f"{day.strftime('%Y-%m-%d')}.json"
    required_output = feature_path.exists() and (label_path.exists() or not with_labels)
    if required_output and report_path.exists() and not force:
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        return cached
    warm_start = day - pd.Timedelta(minutes=max(WINDOW_SECONDS) / 60)
    feature_end = day + pd.Timedelta(days=1)
    source_end = feature_end + (pd.Timedelta(hours=config.horizon_hours) if with_labels else pd.Timedelta(0))
    binance, binance_report = read_binance_seconds(
        day_files(binance_dir, warm_start, feature_end),
        warm_start,
        feature_end,
    )
    quotes, quote_report = read_sync_quotes(
        day_files(sync_dir, warm_start, source_end),
        warm_start,
        source_end,
    )
    features = build_minute_features(binance, quotes, day, feature_end, config)
    features = join_t9_state(features, load_t9_state(decisions_path))
    t9_flat_rows = int(features["t9_flat"].eq(1.0).sum()) if "t9_flat" in features else 0
    dataset, label_report = add_labels(features, quotes, config) if with_labels else (features, {})
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    if with_labels:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(label_path, index=False)
    report = {
        "dataset_version": "c1_seq_v1",
        "day": day,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0),
        "rehearsal_only": bool(REHEARSAL_START <= day < REHEARSAL_END),
        "config": asdict(config),
        "source": {"binance": binance_report, "sync": quote_report},
        "feature_rows": int(len(features)),
        "t9_flat_rows": t9_flat_rows,
        "independent_c1_rows": int(len(features)),
        "output_rows": int(len(dataset)),
        "feature_hash": frame_hash(dataset),
        "feature_timestamp_after_decision": int(
            (
                pd.to_datetime(dataset.get("feature_max_receive_time"), utc=True, errors="coerce")
                > pd.to_datetime(dataset.get("decision_time"), utc=True, errors="coerce")
            ).sum()
        ) if len(dataset) else 0,
        "label_report": label_report,
        "outputs": {
            "features": str(feature_path),
            "labels": str(label_path) if with_labels else None,
        },
        "cache_hit": False,
    }
    atomic_write_json(report_path, report)
    return report


def replay_report(shards: list[Path], run_id: str, output_dir: Path) -> dict[str, Any]:
    frames = [pd.read_parquet(path) for path in shards if path.exists()]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        frame = assign_expansion_episodes(
            frame.sort_values("decision_time").reset_index(drop=True),
            SequenceBuildConfig(),
        )
    episode_rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        episode_col = f"{side}_episode_id"
        if episode_col not in frame:
            continue
        winners = frame[frame[episode_col].notna()].copy()
        for episode_id, group in winners.groupby(episode_col, sort=True):
            group = group.sort_values("decision_time")
            anchor = group.iloc[0]
            eligible = group[group[f"{side}_early_usable_capture"].astype(bool)]
            earliest = eligible.iloc[0] if len(eligible) else None
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "side": side,
                    "onset": anchor.get("decision_time"),
                    "earliest_eligible_decision": (
                        earliest.get("decision_time") if earliest is not None else None
                    ),
                    "signal_lag_minutes": (
                        (
                            pd.Timestamp(earliest.get("decision_time"))
                            - pd.Timestamp(anchor.get("decision_time"))
                        ).total_seconds() / 60.0
                        if earliest is not None else None
                    ),
                    "consumed_r": (
                        earliest.get(f"{side}_consumed_r_from_episode_onset")
                        if earliest is not None else None
                    ),
                    "remaining_mfe_r": (
                        earliest.get(f"{side}_remaining_mfe_r")
                        if earliest is not None else float(group[f"{side}_remaining_mfe_r"].max())
                    ),
                    "rejection_reason": None if earliest is not None else "early_capture_geometry_failed",
                }
            )
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0),
        "corpus_role": "rehearsal_case_test_only",
        "fit_permitted": False,
        "rows": int(len(frame)),
        "valid_episode_count": len(episode_rows),
        "episodes": episode_rows,
        "feature_after_decision": int(frame.get("feature_after_decision", pd.Series(dtype=bool)).sum()),
        "source_shards": [str(path) for path in shards if path.exists()],
        "dataset_hash": frame_hash(frame),
        "decision": {
            "status": "rehearsal_complete" if episode_rows else "rehearsal_incomplete",
            "reason": "July cases are replay diagnostics and cannot alter fitted parameters.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"c1_sequence_replay_{run_id}.json"
    md_path = output_dir / f"c1_sequence_replay_{run_id}.md"
    atomic_write_json(json_path, payload)
    lines = [
        "# C1-SEQ Rehearsal Replay",
        "",
        f"Run ID: `{run_id}`",
        f"Status: `{payload['decision']['status']}`",
        f"Rows: `{payload['rows']}`",
        f"Independent expansion episodes: `{payload['valid_episode_count']}`",
        f"Feature-after-decision violations: `{payload['feature_after_decision']}`",
        "",
        "July is case-test material only. It is excluded from fitting, threshold selection, and promotion evidence.",
        "",
        "## Episodes",
        "",
        "| Side | Onset | Early Capture | Remaining MFE | Rejection |",
        "|---|---|---:|---:|---|",
    ]
    for row in episode_rows:
        lines.append(
            f"| {row['side']} | {row['onset']} | {row['earliest_eligible_decision'] or ''} | "
            f"{float(row['remaining_mfe_r'] or 0.0):.3f} | {row['rejection_reason'] or ''} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def synthetic_inputs(root: Path) -> tuple[Path, Path, Path]:
    binance_dir = root / "binance"
    sync_dir = root / "sync"
    binance_dir.mkdir(parents=True)
    sync_dir.mkdir(parents=True)
    day = pd.Timestamp("2026-07-20", tz="UTC")
    binance_path = binance_dir / "2026-07-20.jsonl"
    sync_path = sync_dir / "2026-07-20.jsonl"
    decision_path = root / "decisions.jsonl"
    binance_rows = []
    sync_rows = []
    for second in range(0, 60 * 20):
        ts = day + pd.Timedelta(seconds=second, milliseconds=100)
        price = 10000.0 + second * 0.25
        bid = price - 1.0
        ask = price + 1.0
        depth = {
            "event_type": "depth",
            "local_receive_time": ts.isoformat(),
            "source_after_receive": False,
            "payload": {
                "b": [[str(bid - i), str(2 + i + (second % 3))] for i in range(10)],
                "a": [[str(ask + i), str(8 - min(i, 6))] for i in range(10)],
            },
        }
        trade = {
            "event_type": "agg_trade",
            "local_receive_time": (ts + pd.Timedelta(milliseconds=100)).isoformat(),
            "source_after_receive": False,
            "payload": {"p": str(price), "q": "0.5", "m": False},
        }
        binance_rows.extend([depth, trade])
        if second == 300:
            binance_rows.append(
                {
                    "event_type": "liquidation",
                    "local_receive_time": (ts + pd.Timedelta(milliseconds=200)).isoformat(),
                    "source_after_receive": False,
                    "payload": {"o": {"S": "SELL", "ap": str(price), "q": "3"}},
                }
            )
        sync_rows.append(
            {
                "local_receive_time": ts.isoformat(),
                "fbs_bid": price - 2.0,
                "fbs_ask": price + 2.0,
                "fbs_spread_bps": 4.0,
                "fbs_vs_binance_mid_divergence_bps": 0.0,
                "binance_mid": price,
                "has_fbs_tick": 1,
                "incomplete": False,
            }
        )
    binance_path.write_text("\n".join(json.dumps(row) for row in binance_rows) + "\n", encoding="utf-8")
    sync_path.write_text("\n".join(json.dumps(row) for row in sync_rows) + "\n", encoding="utf-8")
    decision_path.write_text(
        json.dumps(
            {
                "action": "h1_evaluation",
                "probe_join_time": day.isoformat(),
                "spree_open_count": 0,
                "reason": "synthetic_flat",
                "regime": "CHOPPY",
            }
        ) + "\n",
        encoding="utf-8",
    )
    return binance_dir, sync_dir, decision_path


def smoke(_: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        binance_dir, sync_dir, decisions = synthetic_inputs(root)
        config = SequenceBuildConfig(horizon_hours=0.05, max_label_gap_seconds=2.0)
        start = pd.Timestamp("2026-07-20", tz="UTC")
        end = start + pd.Timedelta(minutes=20)
        binance, _ = read_binance_seconds([binance_dir / "2026-07-20.jsonl"], start, end)
        quotes, _ = read_sync_quotes([sync_dir / "2026-07-20.jsonl"], start, end)
        features = join_t9_state(build_minute_features(binance, quotes, start, end, config), load_t9_state(decisions))
        labeled, label_report = add_labels(features, quotes, config)
        sequence_ready = labeled[
            (labeled["has_depth_sequence"] == 1.0)
            & (labeled["has_flow_sequence"] == 1.0)
        ]
        first = sequence_ready.iloc[0]
        active_features = features.copy()
        active_time = pd.Timestamp(first["decision_time"])
        active_features.loc[
            pd.to_datetime(active_features["feature_timestamp"], utc=True).eq(active_time),
            "t9_flat",
        ] = 0.0
        active_labeled, _ = add_labels(active_features, quotes, config)
        causal = bool(pd.Timestamp(first["feature_max_receive_time"]) <= pd.Timestamp(first["decision_time"]))
        entry_causal = bool(pd.Timestamp(first["entry_time"]) >= pd.Timestamp(first["decision_time"]))
        cost_monotonic = bool(
            first["long_net_r_10bps"] <= first["long_net_r_5bps"] <= first["long_gross_r"]
        )

        # A six-hour horizon is timestamp based even when rows are missing.
        decision = pd.Timestamp("2026-07-20T00:01:00Z")
        times = np.array(
            [decision.value - 1, decision.value + int(6 * 3600 * 1e9)],
            dtype=np.int64,
        )
        wall_clock_search = int(np.searchsorted(times, (decision + pd.Timedelta(hours=6)).value, side="right")) == 2

        # Gap injection must invalidate labels rather than bridge the path.
        gapped = quotes.drop(quotes.index[(quotes["quote_time"] >= start + pd.Timedelta(minutes=5)) & (quotes["quote_time"] < start + pd.Timedelta(minutes=6))])
        _, gap_report = add_labels(features, gapped, config)
        checks = {
            "minute_rows_created": len(features) >= 4 and len(sequence_ready) > 0,
            "depth_sequence_present": float(first["has_depth_sequence"]) == 1.0,
            "flow_sequence_present": float(first["has_flow_sequence"]) == 1.0,
            "liquidation_feature_present": "long_liq_notional_5m" in labeled,
            "direction_symmetric_columns": all(f"{side}_gross_r" in labeled for side in ("long", "short")),
            "feature_receive_time_causal": causal,
            "entry_quote_at_or_after_decision": entry_causal,
            "cost_monotonic": cost_monotonic and label_report["cost_monotonic"],
            "wall_clock_horizon": wall_clock_search,
            "gap_invalidates_label": gap_report["skip_counts"].get("future_fbs_gap", 0) > 0,
            "t9_context_preserved_without_signal_filter": "t9_flat" in labeled,
            "t9_active_rows_are_labeled": bool(active_labeled["t9_flat"].eq(0.0).any()),
        }
        if not all(checks.values()):
            raise AssertionError(checks)
        first_hash = frame_hash(labeled)
        second_hash = frame_hash(add_labels(features, quotes, config)[0])
        checks["deterministic_hash"] = first_hash == second_hash
        if not checks["deterministic_hash"]:
            raise AssertionError(checks)
    print(json.dumps({"status": "ok", "checks": checks}, indent=2))
    return 0


def build_day_command(args: argparse.Namespace) -> int:
    report = build_day(
        utc_ts(args.date) or pd.Timestamp(args.date, tz="UTC"),
        Path(args.binance_dir),
        Path(args.sync_dir),
        Path(args.decisions),
        Path(args.output_dir),
        SequenceBuildConfig(),
        with_labels=not args.features_only,
        force=args.force,
    )
    print(json.dumps(json_ready(report), indent=2))
    return 0


def build_range_command(args: argparse.Namespace) -> int:
    start = utc_ts(args.start)
    end = utc_ts(args.end)
    if start is None or end is None or end <= start:
        raise ValueError("start/end must define an increasing UTC range")
    reports = []
    for day in pd.date_range(start.floor("D"), end.floor("D"), freq="1D", inclusive="left", tz="UTC"):
        reports.append(
            build_day(
                day,
                Path(args.binance_dir),
                Path(args.sync_dir),
                Path(args.decisions),
                Path(args.output_dir),
                SequenceBuildConfig(),
                with_labels=not args.features_only,
                force=args.force,
            )
        )
    print(json.dumps(json_ready({"days": len(reports), "reports": reports}), indent=2))
    return 0


def replay_command(args: argparse.Namespace) -> int:
    start = utc_ts(args.start)
    end = utc_ts(args.end)
    if start is None or end is None or end <= start:
        raise ValueError("start/end must define an increasing UTC range")
    shard_dir = Path(args.shard_dir)
    shards = [
        shard_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
        for day in pd.date_range(start.floor("D"), end.floor("D"), freq="1D", inclusive="left", tz="UTC")
    ]
    payload = replay_report(shards, args.run_id, Path(args.output_dir))
    print(json.dumps(json_ready(payload["decision"]), indent=2))
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binance-dir", default=str(DEFAULT_BINANCE_DIR))
    parser.add_argument("--sync-dir", default=str(DEFAULT_SYNC_DIR))
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_SHARD_DIR))
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--force", action="store_true")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.set_defaults(func=smoke)
    day_parser = sub.add_parser("build-day")
    day_parser.add_argument("--date", required=True)
    add_common(day_parser)
    day_parser.set_defaults(func=build_day_command)
    range_parser = sub.add_parser("build-range")
    range_parser.add_argument("--start", required=True)
    range_parser.add_argument("--end", required=True)
    add_common(range_parser)
    range_parser.set_defaults(func=build_range_command)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--start", default="2026-07-20")
    replay_parser.add_argument("--end", default="2026-07-31")
    replay_parser.add_argument("--shard-dir", default=str(DEFAULT_LABEL_SHARD_DIR))
    replay_parser.add_argument("--run-id", default="c1_seq_rehearsal")
    replay_parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    replay_parser.set_defaults(func=replay_command)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
