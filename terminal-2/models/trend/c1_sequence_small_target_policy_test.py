"""Exploratory small-target policy test on C1-SEQ rehearsal features.

Signals are symmetric rolling-tail anomalies fitted only from the preceding six
wall-clock hours. The test is diagnostic and does not freeze or promote C1.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.trend.c1_sequence_dataset import (
    MAX_ENTRY_DELAY_SECONDS,
    MAX_LABEL_GAP_SECONDS,
    STOP_WIDTH_BPS,
    atomic_write_json,
    day_files,
    first_executable_quote_index,
    json_ready,
    read_sync_quotes,
    utc_ts,
)


DEFAULT_FEATURE_DIR = ROOT / "data" / "features" / "microstructure" / "c1_sequence" / "features"
DEFAULT_SYNC_DIR = ROOT / "data" / "raw" / "live_probe" / "sync"
DEFAULT_REPORT_DIR = ROOT / "research" / "validation_runs"
SIGNAL_COLUMNS = (
    "agg_trade_imbalance_60s",
    "signed_trade_notional_60s",
    "fbs_return_60s",
)
TARGETS_R = (0.25, 0.50, 0.75, 1.0, 1.5)
EXTRA_COSTS_BPS = (0.0, 2.0, 5.0, 10.0)


def load_features(directory: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for day in pd.date_range(start.floor("D"), (end - pd.Timedelta(microseconds=1)).floor("D"), freq="1D", tz="UTC"):
        path = directory / f"{day.strftime('%Y-%m-%d')}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["feature_timestamp"] = pd.to_datetime(frame["feature_timestamp"], utc=True)
    return frame[
        (frame["feature_timestamp"] >= start)
        & (frame["feature_timestamp"] < end)
    ].sort_values("feature_timestamp").drop_duplicates("feature_timestamp", keep="last").reset_index(drop=True)


def build_candidate_state(
    features: pd.DataFrame,
    reference_hours: float,
    quantile: float,
    require_t9_flat: bool = True,
) -> pd.DataFrame:
    required = ["feature_timestamp", *SIGNAL_COLUMNS]
    if require_t9_flat:
        required.append("t9_flat")
    missing = [column for column in required if column not in features]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    work = features.copy().set_index("feature_timestamp")
    lower_hits = []
    upper_hits = []
    window = f"{reference_hours:g}h"
    for column in SIGNAL_COLUMNS:
        values = pd.to_numeric(work[column], errors="coerce")
        rolling = values.rolling(window, closed="left", min_periods=180)
        lower = rolling.quantile(quantile)
        upper = rolling.quantile(1.0 - quantile)
        lower_hits.append(values <= lower)
        upper_hits.append(values >= upper)
    lower_score = pd.concat(lower_hits, axis=1).sum(axis=1)
    upper_score = pd.concat(upper_hits, axis=1).sum(axis=1)
    short_raw = lower_score.ge(2)
    long_raw = upper_score.ge(2)
    consecutive = work.index.to_series().diff().eq(pd.Timedelta(minutes=1))
    short_persistent = short_raw & short_raw.shift(1, fill_value=False) & consecutive
    long_persistent = long_raw & long_raw.shift(1, fill_value=False) & consecutive
    conflict = short_persistent & long_persistent
    direction = pd.Series(0, index=work.index, dtype=int)
    direction.loc[long_persistent & ~conflict] = 1
    direction.loc[short_persistent & ~conflict] = -1
    new_event = direction.ne(0) & direction.ne(direction.shift(1, fill_value=0))
    eligible = new_event
    if require_t9_flat:
        eligible = eligible & work["t9_flat"].eq(1.0)
    state = work.copy()
    state["feature_timestamp"] = state.index
    state["decision_time"] = state.index
    state["lower_score"] = lower_score
    state["upper_score"] = upper_score
    state["short_raw"] = short_raw
    state["long_raw"] = long_raw
    state["short_persistent"] = short_persistent
    state["long_persistent"] = long_persistent
    state["conflict"] = conflict
    state["direction"] = direction
    state["new_event"] = new_event
    state["eligible"] = eligible
    state["anomaly_score"] = np.where(
        state["direction"].eq(1),
        state["upper_score"],
        state["lower_score"],
    )
    return state.reset_index(drop=True)


def build_candidates(
    features: pd.DataFrame,
    reference_hours: float,
    quantile: float,
    require_t9_flat: bool = True,
) -> pd.DataFrame:
    state = build_candidate_state(features, reference_hours, quantile, require_t9_flat)
    return state.loc[state["eligible"]].reset_index(drop=True)


def evaluate_trade(
    quote_frame: pd.DataFrame,
    quote_times_ns: np.ndarray,
    decision: pd.Timestamp,
    side: int,
    target_r: float,
    horizon_hours: float,
) -> dict[str, Any]:
    entry_idx = first_executable_quote_index(quote_times_ns, decision, MAX_ENTRY_DELAY_SECONDS)
    if entry_idx is None:
        return {"valid": False, "reason": "no_executable_quote"}
    horizon_end = decision + pd.Timedelta(hours=horizon_hours)
    end_idx = int(np.searchsorted(quote_times_ns, horizon_end.value, side="right"))
    path = quote_frame.iloc[entry_idx:end_idx].copy()
    if len(path) < 2:
        return {"valid": False, "reason": "insufficient_path"}
    entry = path.iloc[0]
    entry_price = float(entry["fbs_ask" if side == 1 else "fbs_bid"])
    executable = pd.to_numeric(path["fbs_bid" if side == 1 else "fbs_ask"], errors="coerce").to_numpy(float)
    stop_distance = entry_price * STOP_WIDTH_BPS / 10000.0
    stop_price = entry_price - side * stop_distance
    target_price = entry_price + side * target_r * stop_distance
    if side == 1:
        stop_mask = executable <= stop_price
        target_mask = executable >= target_price
    else:
        stop_mask = executable >= stop_price
        target_mask = executable <= target_price
    stop_hits = np.flatnonzero(stop_mask)
    target_hits = np.flatnonzero(target_mask)
    stop_idx = int(stop_hits[0]) if len(stop_hits) else None
    target_idx = int(target_hits[0]) if len(target_hits) else None
    if stop_idx is not None and (target_idx is None or stop_idx <= target_idx):
        exit_idx, reason, gross_r = stop_idx, "sl_hit", -1.0
    elif target_idx is not None:
        exit_idx, reason, gross_r = target_idx, "tp_hit", target_r
    else:
        path_end = pd.Timestamp(path.iloc[-1]["quote_time"])
        if path_end < horizon_end - pd.Timedelta(seconds=MAX_LABEL_GAP_SECONDS):
            return {"valid": False, "reason": "incomplete_horizon"}
        exit_idx = len(path) - 1
        exit_price = float(executable[exit_idx])
        gross_r = side * (exit_price - entry_price) / stop_distance
        reason = "time_exit"
    observed = path.iloc[: exit_idx + 1]
    gaps = observed["quote_time"].diff().dt.total_seconds().dropna()
    max_gap = float(gaps.max()) if len(gaps) else 0.0
    if max_gap > MAX_LABEL_GAP_SECONDS:
        return {"valid": False, "reason": "broker_quote_gap", "max_gap_seconds": max_gap}
    path_r = side * (executable[: exit_idx + 1] - entry_price) / stop_distance
    mfe_index = int(np.argmax(path_r))
    mae_index = int(np.argmin(path_r))
    return {
        "valid": True,
        "reason": reason,
        "decision_time": decision,
        "entry_time": pd.Timestamp(entry["quote_time"]),
        "exit_time": pd.Timestamp(path.iloc[exit_idx]["quote_time"]),
        "side": side,
        "gross_r": float(gross_r),
        "mfe_r": float(path_r[mfe_index]),
        "mae_r": float(path_r[mae_index]),
        "time_to_mfe_minutes": float(
            (pd.Timestamp(observed.iloc[mfe_index]["quote_time"]) - pd.Timestamp(entry["quote_time"])).total_seconds() / 60.0
        ),
        "time_to_mae_minutes": float(
            (pd.Timestamp(observed.iloc[mae_index]["quote_time"]) - pd.Timestamp(entry["quote_time"])).total_seconds() / 60.0
        ),
        "entry_spread_bps": float(entry.get("fbs_spread_bps", np.nan)),
        "max_gap_seconds": max_gap,
    }


def simulate(
    candidates: pd.DataFrame,
    quotes: pd.DataFrame,
    target_r: float,
    extra_cost_bps: float,
    horizon_hours: float,
) -> dict[str, Any]:
    quote_frame = quotes[quotes["has_fbs_tick"] > 0].sort_values("quote_time").reset_index(drop=True)
    quote_times_ns = pd.to_datetime(quote_frame["quote_time"], utc=True).dt.as_unit("ns").astype("int64").to_numpy(np.int64)
    trades = []
    skips: Counter[str] = Counter()
    occupied_until: pd.Timestamp | None = None
    for row in candidates.to_dict("records"):
        decision = pd.Timestamp(row["decision_time"])
        if occupied_until is not None and decision < occupied_until:
            skips["occupied"] += 1
            continue
        trade = evaluate_trade(quote_frame, quote_times_ns, decision, int(row["direction"]), target_r, horizon_hours)
        if not trade.get("valid"):
            skips[str(trade.get("reason"))] += 1
            continue
        trade["net_r"] = float(trade["gross_r"] - extra_cost_bps / STOP_WIDTH_BPS)
        trades.append(trade)
        occupied_until = pd.Timestamp(trade["exit_time"])
    net = np.array([trade["net_r"] for trade in trades], dtype=float)
    gross = np.array([trade["gross_r"] for trade in trades], dtype=float)
    wins = net[net > 0]
    losses = net[net < 0]
    equity = np.cumsum(net) if len(net) else np.array([], dtype=float)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = peaks[1:] - equity if len(equity) else np.array([], dtype=float)
    return {
        "trades": int(len(trades)),
        "net_r": float(net.sum()) if len(net) else 0.0,
        "gross_r": float(gross.sum()) if len(gross) else 0.0,
        "avg_r": float(net.mean()) if len(net) else None,
        "win_rate": float((net > 0).mean()) if len(net) else None,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None,
        "max_drawdown_r": float(drawdown.max()) if len(drawdown) else 0.0,
        "exit_reasons": dict(Counter(trade["reason"] for trade in trades)),
        "median_entry_spread_bps": float(np.nanmedian([trade["entry_spread_bps"] for trade in trades])) if trades else None,
        "skips": dict(skips),
        "trade_details": trades,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = utc_ts(args.start)
    end = utc_ts(args.end)
    if start is None or end is None or end <= start:
        raise ValueError("start/end must define a valid UTC range")
    features = load_features(Path(args.feature_dir), start, end)
    feature_after_decision = 0
    if "feature_max_receive_time" in features:
        feature_after_decision = int(
            (
                pd.to_datetime(features["feature_max_receive_time"], utc=True, errors="coerce")
                > features["feature_timestamp"]
            ).sum()
        )
    if feature_after_decision:
        raise ValueError(f"feature-after-decision violations: {feature_after_decision}")
    candidates = build_candidates(features, args.reference_hours, args.quantile)
    quote_end = end + pd.Timedelta(hours=args.horizon_hours)
    quotes, quote_report = read_sync_quotes(
        day_files(Path(args.sync_dir), start, quote_end),
        start,
        quote_end,
    )
    results = {}
    for target in TARGETS_R:
        results[f"{target:g}R"] = {
            f"{cost:g}bps": simulate(candidates, quotes, target, cost, args.horizon_hours)
            for cost in EXTRA_COSTS_BPS
        }
    cost_monotonic = all(
        all(
            profiles[f"{higher:g}bps"]["net_r"] <= profiles[f"{lower:g}bps"]["net_r"] + 1e-12
            for lower, higher in zip(EXTRA_COSTS_BPS, EXTRA_COSTS_BPS[1:])
        )
        for profiles in results.values()
    )
    payload = {
        "run_id": args.run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0),
        "status": "exploratory_rehearsal_only",
        "period": {"start": start, "end_exclusive": end},
        "signal": {
            "rule": "persistent_symmetric_two_of_three_rolling_tail_anomaly",
            "reference_hours": args.reference_hours,
            "quantile": args.quantile,
            "candidate_count": int(len(candidates)),
            "long_candidates": int(candidates["direction"].eq(1).sum()) if len(candidates) else 0,
            "short_candidates": int(candidates["direction"].eq(-1).sum()) if len(candidates) else 0,
        },
        "economics": {
            "stop_width_bps": STOP_WIDTH_BPS,
            "horizon_hours": args.horizon_hours,
            "spread": "embedded in FBS executable bid/ask",
            "extra_cost_bps": list(EXTRA_COSTS_BPS),
            "results": results,
        },
        "data": {
            "feature_rows": int(len(features)),
            "feature_after_decision": feature_after_decision,
            "cost_monotonic": cost_monotonic,
            "quote_report": quote_report,
        },
        "limitations": [
            "July/August FBS-era observations are rehearsal data already seen during development.",
            "The rolling anomaly rule is a diagnostic proxy, not a trained or frozen C1-SEQ model.",
            "Tickmill transfer requires forward confirmation with Tickmill executable quotes.",
        ],
    }
    report_dir = Path(args.report_dir)
    json_path = report_dir / f"c1_sequence_small_target_policy_{args.run_id}.json"
    md_path = report_dir / f"c1_sequence_small_target_policy_{args.run_id}.md"
    atomic_write_json(json_path, payload)
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    payload["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# C1-SEQ Small-Target Policy: {payload['run_id']}",
        "",
        f"Status: `{payload['status']}`",
        f"Candidates: `{payload['signal']['candidate_count']}`",
        "",
        "| Target | Extra cost | Trades | Net R | Avg R | PF | WR | Max DD R |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target, profiles in payload["economics"]["results"].items():
        for cost, metrics in profiles.items():
            lines.append(
                f"| {target} | {cost} | {metrics['trades']} | {metrics['net_r']:.3f} | "
                f"{metrics['avg_r'] if metrics['avg_r'] is not None else 'null'} | "
                f"{metrics['profit_factor'] if metrics['profit_factor'] is not None else 'null'} | "
                f"{metrics['win_rate'] if metrics['win_rate'] is not None else 'null'} | {metrics['max_drawdown_r']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Spread is embedded in executable quotes; cost profiles are additional fee/slippage stress.",
            "This report is exploratory and cannot promote C1-SEQ.",
        ]
    )
    return "\n".join(lines) + "\n"


def smoke() -> dict[str, Any]:
    decision = pd.Timestamp("2026-01-01T00:00:00Z")
    times = [decision + pd.Timedelta(milliseconds=500), decision + pd.Timedelta(seconds=1, milliseconds=500)]
    long_quotes = pd.DataFrame(
        {
            "quote_time": times,
            "fbs_bid": [99.99, 100.51],
            "fbs_ask": [100.0, 100.52],
            "fbs_spread_bps": [1.0, 1.0],
            "has_fbs_tick": [1.0, 1.0],
        }
    )
    short_quotes = pd.DataFrame(
        {
            "quote_time": times,
            "fbs_bid": [100.0, 99.48],
            "fbs_ask": [100.01, 99.49],
            "fbs_spread_bps": [1.0, 1.0],
            "has_fbs_tick": [1.0, 1.0],
        }
    )
    outcomes = []
    for side, frame in ((1, long_quotes), (-1, short_quotes)):
        times_ns = pd.to_datetime(frame["quote_time"], utc=True).dt.as_unit("ns").astype("int64").to_numpy(np.int64)
        outcome = evaluate_trade(frame, times_ns, decision, side, 0.5, 6.0)
        assert outcome["valid"] and outcome["reason"] == "tp_hit"
        assert abs(float(outcome["gross_r"]) - 0.5) < 1e-12
        outcomes.append(outcome)
    return {"status": "smoke_passed", "long_short_symmetric": True, "outcomes": outcomes}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-07-20T00:00:00Z")
    parser.add_argument("--end", default="2026-07-30T00:00:00Z")
    parser.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--sync-dir", default=str(DEFAULT_SYNC_DIR))
    parser.add_argument("--reference-hours", type=float, default=6.0)
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--horizon-hours", type=float, default=6.0)
    parser.add_argument("--run-id", default="july_rehearsal_small_targets")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = smoke() if args.smoke else run(args)
    print(json.dumps(json_ready(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
