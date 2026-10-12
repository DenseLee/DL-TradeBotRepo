"""Attribute C1 losses to its individual rolling-agreement components.

This is a read-only diagnostic.  It replays the current C1 baseline lifecycle
on Tickmill executable quotes, keeping each component's candidate timestamps
and comparing its chosen direction with deterministic 50/50 random sides at
the identical candidate opportunities.  It is not a model-selection or live
promotion tool because the available Tickmill interval has already been seen.
"""

from __future__ import annotations

import argparse
import json
import math
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
    STOP_WIDTH_BPS,
    atomic_write_json,
    day_files,
    read_sync_quotes,
    utc_ts,
)
from models.trend.c1_sequence_small_target_policy_test import (
    SIGNAL_COLUMNS,
    evaluate_trade,
    load_features,
)


DEFAULT_FEATURE_DIR = ROOT / "data" / "features" / "microstructure" / "c1_sequence_tickmill_v1" / "features"
DEFAULT_SYNC_DIR = ROOT / "data" / "raw" / "live_probe_tickmill" / "sync"
DEFAULT_REPORT_DIR = ROOT / "research" / "validation_runs"
REFERENCE_HOURS = 6.0
QUANTILE = 0.10
PERSISTENCE_MINUTES = 2
TARGET_R = 0.50
HORIZON_HOURS = 2.0
COIN_REPLICATIONS = 500
RANDOM_SEED = 20260912

VARIANTS: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("flow_only", ("agg_trade_imbalance_60s",), 1),
    ("notional_only", ("signed_trade_notional_60s",), 1),
    ("price_only", ("fbs_return_60s",), 1),
    ("flow_notional", ("agg_trade_imbalance_60s", "signed_trade_notional_60s"), 2),
    ("flow_price", ("agg_trade_imbalance_60s", "fbs_return_60s"), 2),
    ("notional_price", ("signed_trade_notional_60s", "fbs_return_60s"), 2),
    ("current_two_of_three", SIGNAL_COLUMNS, 2),
    ("all_three", SIGNAL_COLUMNS, 3),
)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def candidate_state(
    features: pd.DataFrame,
    columns: tuple[str, ...],
    minimum_agreement: int,
) -> pd.DataFrame:
    """Build the live C1 state with a restricted agreement set.

    Quantiles use only the preceding six wall-clock hours.  The event and
    persistence mechanics deliberately match ``build_candidate_state``.
    """
    work = features[["feature_timestamp", *columns]].copy().set_index("feature_timestamp")
    upper_hits: list[pd.Series] = []
    lower_hits: list[pd.Series] = []
    for column in columns:
        values = pd.to_numeric(work[column], errors="coerce")
        rolling = values.rolling(f"{REFERENCE_HOURS:g}h", closed="left", min_periods=180)
        lower_hits.append(values.le(rolling.quantile(QUANTILE)))
        upper_hits.append(values.ge(rolling.quantile(1.0 - QUANTILE)))
    lower_score = pd.concat(lower_hits, axis=1).sum(axis=1)
    upper_score = pd.concat(upper_hits, axis=1).sum(axis=1)
    short_raw = lower_score.ge(minimum_agreement)
    long_raw = upper_score.ge(minimum_agreement)
    consecutive = work.index.to_series().diff().eq(pd.Timedelta(minutes=1))
    short_persistent = short_raw & short_raw.shift(1, fill_value=False) & consecutive
    long_persistent = long_raw & long_raw.shift(1, fill_value=False) & consecutive
    conflict = short_persistent & long_persistent
    direction = pd.Series(0, index=work.index, dtype=int)
    direction.loc[long_persistent & ~conflict] = 1
    direction.loc[short_persistent & ~conflict] = -1
    new_event = direction.ne(0) & direction.ne(direction.shift(1, fill_value=0))
    directional_hits = pd.DataFrame(
        {
            column: (direction.eq(1) & upper_hits[index]) | (direction.eq(-1) & lower_hits[index])
            for index, column in enumerate(columns)
        }
    )
    agreement_components = directional_hits.apply(
        lambda row: "+".join(column for column, hit in row.items() if bool(hit)), axis=1
    )
    return pd.DataFrame(
        {
            "decision_time": work.index,
            "direction": direction.to_numpy(),
            "new_event": new_event.to_numpy(),
            "upper_score": upper_score.to_numpy(),
            "lower_score": lower_score.to_numpy(),
            "agreement_components": agreement_components.to_numpy(),
        }
    )


def build_outcome_cache(
    decisions: set[pd.Timestamp],
    quote_frame: pd.DataFrame,
    quote_times_ns: np.ndarray,
) -> dict[tuple[int, int], dict[str, Any]]:
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    for decision in sorted(decisions):
        for side in (-1, 1):
            cache[(int(decision.value), side)] = evaluate_trade(
                quote_frame,
                quote_times_ns,
                decision,
                side,
                TARGET_R,
                HORIZON_HOURS,
            )
    return cache


def replay(
    candidates: pd.DataFrame,
    outcomes: dict[tuple[int, int], dict[str, Any]],
    extra_cost_bps: float,
    random_sides: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the exact single-position policy using cached executable outcomes."""
    occupied_until: pd.Timestamp | None = None
    trades: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    rows = candidates.sort_values("decision_time").reset_index(drop=True)
    if random_sides is not None and len(random_sides) != len(rows):
        raise ValueError("random_sides must align with candidate rows")
    for index, row in rows.iterrows():
        decision = pd.Timestamp(row["decision_time"])
        if occupied_until is not None and decision < occupied_until:
            skips["occupied"] += 1
            continue
        side = int(random_sides[index]) if random_sides is not None else int(row["direction"])
        outcome = outcomes[(int(decision.value), side)]
        if not outcome.get("valid"):
            skips[str(outcome.get("reason", "invalid_path"))] += 1
            continue
        trade = dict(outcome)
        trade["net_r"] = float(trade["gross_r"] - extra_cost_bps / STOP_WIDTH_BPS)
        trade["agreement_components"] = str(row.get("agreement_components", ""))
        trades.append(trade)
        occupied_until = pd.Timestamp(trade["exit_time"])
    gross = np.array([float(row["gross_r"]) for row in trades], dtype=float)
    net = np.array([float(row["net_r"]) for row in trades], dtype=float)
    full_stops = int(sum(row["reason"] == "sl_hit" for row in trades))
    losses = gross[gross < 0.0]
    winners = gross[gross > 0.0]
    holding_minutes = np.array(
        [
            (pd.Timestamp(row["exit_time"]) - pd.Timestamp(row["entry_time"])).total_seconds() / 60.0
            for row in trades
        ],
        dtype=float,
    )

    def duration_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        values = np.array(
            [
                (pd.Timestamp(row["exit_time"]) - pd.Timestamp(row["entry_time"])).total_seconds() / 60.0
                for row in rows
            ],
            dtype=float,
        )
        if not len(values):
            return {"mean": None, "median": None, "p25": None, "p75": None}
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
        }

    return {
        "trades": int(len(trades)),
        "gross_r": float(gross.sum()) if len(gross) else 0.0,
        "net_r": float(net.sum()) if len(net) else 0.0,
        "avg_gross_r": float(gross.mean()) if len(gross) else None,
        "win_rate": float((gross > 0.0).mean()) if len(gross) else None,
        "loss_rate": float((gross < 0.0).mean()) if len(gross) else None,
        "full_stop_rate": float(full_stops / len(trades)) if trades else None,
        "full_stops": full_stops,
        "profit_factor_gross": float(winners.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else None,
        "exit_reasons": dict(Counter(row["reason"] for row in trades)),
        "holding_minutes": duration_summary(trades),
        "holding_minutes_by_exit_reason": {
            reason: duration_summary([row for row in trades if row["reason"] == reason])
            for reason in sorted({str(row["reason"]) for row in trades})
        },
        "agreement_component_attribution": component_attribution(trades),
        "skips": dict(skips),
    }


def component_attribution(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Describe which directional agreement signatures produced the losses."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        signature = str(trade.get("agreement_components") or "unavailable")
        grouped.setdefault(signature, []).append(trade)
    result: dict[str, dict[str, Any]] = {}
    for signature, rows in grouped.items():
        gross = np.array([float(row["gross_r"]) for row in rows], dtype=float)
        result[signature] = {
            "trades": len(rows),
            "gross_r": float(gross.sum()),
            "gross_loss_r": float(gross[gross < 0.0].sum()),
            "full_stops": int(sum(row["reason"] == "sl_hit" for row in rows)),
            "full_stop_rate": float(sum(row["reason"] == "sl_hit" for row in rows) / len(rows)),
        }
    return result


def coin_distribution(
    candidates: pd.DataFrame,
    outcomes: dict[tuple[int, int], dict[str, Any]],
    extra_cost_bps: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    net_values: list[float] = []
    win_rates: list[float] = []
    stop_rates: list[float] = []
    trade_counts: list[int] = []
    for _ in range(COIN_REPLICATIONS):
        sides = rng.choice(np.array([-1, 1], dtype=int), size=len(candidates))
        result = replay(candidates, outcomes, extra_cost_bps, sides)
        net_values.append(float(result["net_r"]))
        trade_counts.append(int(result["trades"]))
        if result["win_rate"] is not None:
            win_rates.append(float(result["win_rate"]))
        if result["full_stop_rate"] is not None:
            stop_rates.append(float(result["full_stop_rate"]))

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "p05": None, "p95": None}
        array = np.array(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p05": float(np.quantile(array, 0.05)),
            "p95": float(np.quantile(array, 0.95)),
        }

    return {
        "replications": COIN_REPLICATIONS,
        "net_r": summary(net_values),
        "win_rate": summary(win_rates),
        "full_stop_rate": summary(stop_rates),
        "trades": summary([float(value) for value in trade_counts]),
        "net_r_samples": net_values,
        "win_rate_samples": win_rates,
        "full_stop_rate_samples": stop_rates,
    }


def comparison(actual: dict[str, Any], coin: dict[str, Any]) -> dict[str, Any]:
    net_samples = np.array(coin.pop("net_r_samples"), dtype=float)
    win_samples = np.array(coin.pop("win_rate_samples"), dtype=float)
    stop_samples = np.array(coin.pop("full_stop_rate_samples"), dtype=float)
    return {
        "actual": actual,
        "coin": coin,
        "net_r_vs_coin_mean": float(actual["net_r"] - float(coin["net_r"]["mean"])),
        "win_rate_vs_coin_mean": (
            float(actual["win_rate"] - float(coin["win_rate"]["mean"]))
            if actual["win_rate"] is not None and coin["win_rate"]["mean"] is not None
            else None
        ),
        "full_stop_rate_vs_coin_mean": (
            float(actual["full_stop_rate"] - float(coin["full_stop_rate"]["mean"]))
            if actual["full_stop_rate"] is not None and coin["full_stop_rate"]["mean"] is not None
            else None
        ),
        "coin_net_r_one_sided_p": float((net_samples >= actual["net_r"]).mean()),
        "coin_win_rate_one_sided_p": (
            float((win_samples >= actual["win_rate"]).mean()) if actual["win_rate"] is not None and len(win_samples) else None
        ),
        "coin_full_stop_lower_one_sided_p": (
            float((stop_samples <= actual["full_stop_rate"]).mean())
            if actual["full_stop_rate"] is not None and len(stop_samples)
            else None
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# C1 Agreement Component Audit: {payload['run_id']}",
        "",
        "Status: `observed_tickmill_diagnostic_only`.",
        "",
        "| Variant | Signals | Trades | Gross R | Net R 10bps | WR | Full-stop rate | Net vs 50/50 mean | Coin p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in payload["variants"].items():
        actual = result["at_10bps"]["actual"]
        delta = result["at_10bps"]["net_r_vs_coin_mean"]
        lines.append(
            f"| {name} | {result['signals']} | {actual['trades']} | {actual['gross_r']:.3f} | "
            f"{actual['net_r']:.3f} | {actual['win_rate'] if actual['win_rate'] is not None else 'null'} | "
            f"{actual['full_stop_rate'] if actual['full_stop_rate'] is not None else 'null'} | "
            f"{delta:.3f} | {result['at_10bps']['coin_net_r_one_sided_p']:.3f} |"
        )
    current = payload["variants"].get("current_two_of_three")
    if current:
        lines.extend(
            [
                "",
                "## Current Two-Of-Three Loss Attribution",
                "",
                "| Directional agreement members | Trades | Gross R | Gross loss R | Full stops |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        attribution = current["at_10bps"]["actual"]["agreement_component_attribution"]
        for signature, values in sorted(attribution.items(), key=lambda item: item[1]["gross_loss_r"]):
            lines.append(
                f"| {signature} | {values['trades']} | {values['gross_r']:.3f} | "
                f"{values['gross_loss_r']:.3f} | {values['full_stops']} |"
            )
    lines.extend(
        [
            "",
            "Coin paths preserve each variant's candidate timestamps and one-position occupancy, "
            "but choose long/short independently at 50/50 for 500 deterministic replications.",
            "A low one-sided coin p means the variant beat most coin draws; this observed data is "
            "not eligible for live-selection or promotion.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_start = utc_ts(args.start)
    requested_end = utc_ts(args.end)
    feature_dir = Path(args.feature_dir)
    all_features = load_features(feature_dir, pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2030-01-01", tz="UTC"))
    if all_features.empty:
        raise FileNotFoundError(f"No feature shards found in {feature_dir}")
    start = requested_start or pd.Timestamp(all_features["feature_timestamp"].min())
    latest = pd.Timestamp(all_features["feature_timestamp"].max())
    end = requested_end or (latest - pd.Timedelta(hours=HORIZON_HOURS))
    if end <= start:
        raise ValueError("Requested period leaves no mature two-hour label horizon")
    features = all_features.loc[all_features["feature_timestamp"].between(start, end, inclusive="left")].copy()
    receive = pd.to_datetime(features.get("feature_max_receive_time"), utc=True, errors="coerce")
    feature_after_decision = int((receive > features["feature_timestamp"]).sum()) if len(receive) else 0
    if feature_after_decision:
        raise ValueError(f"feature-after-decision violations: {feature_after_decision}")
    requested_variants = set(args.variants or [name for name, _, _ in VARIANTS])
    known_variants = {name for name, _, _ in VARIANTS}
    unknown_variants = sorted(requested_variants - known_variants)
    if unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}")
    states: dict[str, pd.DataFrame] = {}
    for name, columns, minimum in VARIANTS:
        if name not in requested_variants:
            continue
        state = candidate_state(features, columns, minimum)
        states[name] = state.loc[state["new_event"] & state["direction"].ne(0)].reset_index(drop=True)
    decisions = {pd.Timestamp(value) for state in states.values() for value in state["decision_time"]}
    quote_end = end + pd.Timedelta(hours=HORIZON_HOURS)
    quotes, quote_report = read_sync_quotes(day_files(Path(args.sync_dir), start.floor("D"), quote_end + pd.Timedelta(days=1)), start, quote_end)
    quote_frame = quotes.loc[quotes["has_fbs_tick"].gt(0)].sort_values("quote_time").reset_index(drop=True)
    if quote_frame.empty:
        raise ValueError("No executable Tickmill quotes in requested period")
    quote_times_ns = pd.to_datetime(quote_frame["quote_time"], utc=True).dt.as_unit("ns").astype("int64").to_numpy(np.int64)
    outcomes = build_outcome_cache(decisions, quote_frame, quote_times_ns)
    variants: dict[str, Any] = {}
    for name, candidates in states.items():
        at_5 = comparison(replay(candidates, outcomes, 5.0), coin_distribution(candidates, outcomes, 5.0))
        at_10 = comparison(replay(candidates, outcomes, 10.0), coin_distribution(candidates, outcomes, 10.0))
        variants[name] = {"signals": int(len(candidates)), "at_5bps": at_5, "at_10bps": at_10}
    payload = {
        "run_id": args.run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0),
        "status": "observed_tickmill_diagnostic_only",
        "period": {"start": start, "end_exclusive": end},
        "signal_contract": {
            "columns": list(SIGNAL_COLUMNS),
            "reference_hours": REFERENCE_HOURS,
            "quantile": QUANTILE,
            "persistence_minutes": PERSISTENCE_MINUTES,
            "current_live_rule": "two_of_three",
        },
        "economics": {"stop_width_bps": STOP_WIDTH_BPS, "target_r": TARGET_R, "horizon_hours": HORIZON_HOURS},
        "data": {"feature_rows": int(len(features)), "feature_after_decision": feature_after_decision, "quote_report": quote_report},
        "variants": variants,
        "limitations": [
            "All compared Tickmill data has already been observed during C1 development.",
            "The 50/50 control tests direction on the same candidate opportunities; it does not validate C1 frequency selection.",
            "No result changes a live runner or authorizes a component threshold.",
        ],
    }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"c1_agreement_component_audit_{args.run_id}.json"
    md_path = report_dir / f"c1_agreement_component_audit_{args.run_id}.md"
    atomic_write_json(json_path, payload)
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    payload["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    return payload


def smoke() -> dict[str, Any]:
    timestamps = pd.date_range("2026-01-01", periods=185, freq="min", tz="UTC")
    features = pd.DataFrame(
        {
            "feature_timestamp": timestamps,
            "agg_trade_imbalance_60s": np.arange(185, dtype=float),
            "signed_trade_notional_60s": np.arange(185, dtype=float),
            "fbs_return_60s": np.arange(185, dtype=float),
        }
    )
    state = candidate_state(features, SIGNAL_COLUMNS, 2)
    assert int(state["new_event"].sum()) >= 1
    durations = replay(
        pd.DataFrame(
            {
                "decision_time": [timestamps[-1]],
                "direction": [1],
                "agreement_components": ["synthetic"],
            }
        ),
        {
            (int(timestamps[-1].value), 1): {
                "valid": True,
                "entry_time": timestamps[-1],
                "exit_time": timestamps[-1] + pd.Timedelta(minutes=7),
                "gross_r": 0.5,
                "reason": "tp_hit",
            }
        },
        10.0,
    )
    assert durations["holding_minutes"]["mean"] == 7.0
    return {"status": "smoke_passed", "state_rows": int(len(state)), "holding_minutes": 7.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke")
    audit = commands.add_parser("audit")
    audit.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    audit.add_argument("--sync-dir", default=str(DEFAULT_SYNC_DIR))
    audit.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    audit.add_argument("--start")
    audit.add_argument("--end")
    audit.add_argument("--variants", nargs="+", choices=[name for name, _, _ in VARIANTS], help="Restrict a research rerun to named component variants.")
    audit.add_argument("--run-id", default="tickmill_current_components_20260912")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "smoke":
        print(json.dumps(smoke(), indent=2, default=str))
        return 0
    print(json.dumps(run(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
