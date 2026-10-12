"""Read-only frozen T9 champion shadow evaluator.

Runs beside the live bot and logs what the frozen research champion would have
done at the same H1/M15 decision points. It never sends orders and does not
modify live T9 state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.mt5_executor import MT5Executor
from models.trend.entry_quality_filter import EntryQualityFilter
from models.trend.event_gate import EventGate, EventGateConfig
from models.trend.regime_filter import REGIME_TRENDING, RegimeFilter, RegimeFilterConfig
from models.trend.t9_champion_config import make_t9_champion_config
from models.trend.trend_engine import TrendConfig, TrendEngine


SYMBOL = "BTCUSD"
LOG_DIR = ROOT / "logs"
DEFAULT_SHADOW_LOG = LOG_DIR / "champion_shadow_decisions.jsonl"
DEFAULT_LIVE_LOG = LOG_DIR / "shadow_decisions.jsonl"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(json_ready(payload), sort_keys=True, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute_atr(h1_df: pd.DataFrame, idx: int, window: int = 24) -> float:
    highs = h1_df["high"].astype(float).values
    lows = h1_df["low"].astype(float).values
    if idx < window:
        return float(np.mean(highs[: idx + 1] - lows[: idx + 1]))
    return float(np.mean(highs[idx - window : idx + 1] - lows[idx - window : idx + 1]))


def h1_atr_percentile(h1_df: pd.DataFrame, h1_idx: int, lookback: int = 120) -> float:
    ranges = (h1_df["high"].astype(float) - h1_df["low"].astype(float)).values
    if h1_idx <= 1 or h1_idx >= len(ranges):
        return 0.5
    start = max(0, int(h1_idx) - int(lookback))
    hist = ranges[start:h1_idx]
    hist = hist[np.isfinite(hist)]
    current = float(ranges[h1_idx])
    if len(hist) < 10 or not np.isfinite(current):
        return 0.5
    return float(np.mean(hist <= current))


def current_spread_bps(executor: MT5Executor) -> Optional[float]:
    try:
        tick = executor.mt5.symbol_info_tick(SYMBOL) if executor.mt5 is not None else None
        if tick and tick.bid and tick.ask:
            bid = float(tick.bid)
            ask = float(tick.ask)
            mid = (bid + ask) / 2.0
            return (ask - bid) / max(mid, 1e-9) * 10000.0
    except Exception:
        return None
    return None


def load_regime_baseline() -> Optional[pd.DataFrame]:
    try:
        from data.sources.regime_state import RegimeStateConfig, RegimeStateLoader

        df = RegimeStateLoader(RegimeStateConfig(source="csv")).load_range("2026-05-01", "2026-06-17")
        if df is not None and len(df) > 100:
            return df
    except Exception:
        return None
    return None


class ChampionShadowEvaluator:
    def __init__(
        self,
        executor: MT5Executor,
        output_log: Path,
        live_log: Path,
        fetch_h1: int = 300,
    ) -> None:
        self.executor = executor
        self.output_log = output_log
        self.live_log = live_log
        self.fetch_h1 = int(fetch_h1)
        self.config = make_t9_champion_config()
        self.regime_filter = RegimeFilter(RegimeFilterConfig(allow_trading_in_stress=True))
        self.event_gate = EventGate(EventGateConfig(enabled=True))
        self.trend_engine = TrendEngine(TrendConfig())
        self.entry_filter = EntryQualityFilter(self.config.entry_quality)
        self.regime_data = load_regime_baseline()
        self.processed_keys = self._load_processed_keys()
        self._regime_cache_time = 0.0
        self._regime_cache: tuple[Optional[float], Optional[float]] = (None, None)
        self._regime_history: dict[str, list[float]] = {"f": [], "o": []}

    def _load_processed_keys(self) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for row in read_jsonl(self.output_log):
            keys.add((str(row.get("decision_time")), str(row.get("stage"))))
        return keys

    def run_once(self) -> dict[str, Any]:
        h1_df = self.executor.get_bars("H1", count=self.fetch_h1)
        m15_df = self.executor.get_bars("M15", count=self.fetch_h1 * 4)
        h1_df = self._normalise_bars(h1_df)
        m15_df = self._normalise_bars(m15_df)
        if h1_df is None or m15_df is None or len(h1_df) < 200 or len(m15_df) < 2:
            return self._log_payload("not_ready", None, None, None, None, reason="insufficient_history")

        h1_idx = len(h1_df) - 2
        if h1_idx < 200:
            return self._log_payload("not_ready", h1_df, h1_idx, None, None, reason="h1_warmup")

        h1_close_ts = pd.Timestamp(h1_df["timestamp"].iloc[h1_idx]) + pd.Timedelta(hours=1)
        m15_ts = pd.to_datetime(m15_df["timestamp"].values)
        m15_idx = int(np.searchsorted(m15_ts, h1_close_ts, side="left"))
        last_completed_m15 = len(m15_df) - 2
        regime = self._get_regime(h1_df, h1_idx)

        if m15_idx > last_completed_m15:
            return self._log_payload(
                "pending",
                h1_df,
                h1_idx,
                None,
                None,
                h1_close_ts=h1_close_ts,
                pending_m15_idx=m15_idx,
                last_completed_m15=last_completed_m15,
                regime=regime,
                reason="waiting_for_completed_m15",
            )
        if m15_idx >= len(m15_df) or pd.Timestamp(m15_ts[m15_idx]) < h1_close_ts:
            return self._log_payload(
                "blocked",
                h1_df,
                h1_idx,
                m15_df,
                m15_idx,
                h1_close_ts=h1_close_ts,
                regime=regime,
                reason="invalid_m15_alignment",
            )

        decision_ts = pd.Timestamp(m15_ts[m15_idx])
        if not self.regime_filter.trading_allowed(regime):
            return self._log_payload(
                "blocked",
                h1_df,
                h1_idx,
                m15_df,
                m15_idx,
                h1_close_ts=h1_close_ts,
                regime=regime,
                reason="regime_not_allowed",
            )
        if regime == REGIME_TRENDING:
            return self._log_payload(
                "blocked",
                h1_df,
                h1_idx,
                m15_df,
                m15_idx,
                h1_close_ts=h1_close_ts,
                regime=regime,
                reason="trending_regime_block",
            )

        event_allowed = self.event_gate.trading_allowed(decision_ts)
        if not event_allowed:
            return self._log_payload(
                "blocked",
                h1_df,
                h1_idx,
                m15_df,
                m15_idx,
                h1_close_ts=h1_close_ts,
                regime=regime,
                reason="event_gate_block",
            )

        h4_for_vote = None
        h4_vote = 0
        h1_vote = self.trend_engine._h1_vote(h1_df, h1_idx)
        m15_vote = self.trend_engine._m15_vote(m15_df, m15_idx)
        direction, ep = self.trend_engine.get_signal_vote(h1_df, m15_df, h4_for_vote, h1_idx, m15_idx, regime)
        strength = ep.get("strength", "WEAK") if ep else "WEAK"
        price = float(m15_df["close"].iloc[m15_idx])
        atr = compute_atr(h1_df, h1_idx)
        if direction == 0:
            return self._log_payload(
                "no_trade",
                h1_df,
                h1_idx,
                m15_df,
                m15_idx,
                h1_close_ts=h1_close_ts,
                regime=regime,
                reason="no_trade_vote",
                h4_vote=h4_vote,
                h1_vote=h1_vote,
                m15_vote=m15_vote,
                direction=direction,
                strength=strength,
                price=price,
                atr=atr,
            )

        stop_distance = max(float(atr) * float(ep.get("sl_atr", 2.0)), 1e-9)
        quality = self.entry_filter.evaluate(
            h1_df,
            m15_df,
            h1_idx,
            m15_idx,
            direction,
            price,
            stop_distance,
            ep,
            str(regime),
            event_allowed,
        )
        if self.config.entry_quality.enabled and not quality.allow_entry:
            return self._log_payload(
                "blocked",
                h1_df,
                h1_idx,
                m15_df,
                m15_idx,
                h1_close_ts=h1_close_ts,
                regime=regime,
                reason=f"entry_quality:{quality.block_reason}",
                h4_vote=h4_vote,
                h1_vote=h1_vote,
                m15_vote=m15_vote,
                direction=direction,
                strength=strength,
                price=price,
                atr=atr,
                stop_distance=stop_distance,
                quality_score=quality.quality_score,
                quality_features=quality.features,
            )

        return self._log_payload(
            "would_enter",
            h1_df,
            h1_idx,
            m15_df,
            m15_idx,
            h1_close_ts=h1_close_ts,
            regime=regime,
            reason=f"VOTE_{strength}",
            h4_vote=h4_vote,
            h1_vote=h1_vote,
            m15_vote=m15_vote,
            direction=direction,
            strength=strength,
            price=price,
            atr=atr,
            stop_distance=stop_distance,
            sl_price=price - direction * atr * float(ep.get("sl_atr", 2.0)),
            tp_price=price + direction * atr * float(ep.get("tp_atr", 3.5)),
            max_hold=int(ep.get("max_hold", 24)),
            quality_score=quality.quality_score,
            quality_features=quality.features,
        )

    @staticmethod
    def _normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if len(out) and hasattr(out["timestamp"].iloc[0], "tzinfo") and out["timestamp"].iloc[0].tzinfo is not None:
            out["timestamp"] = out["timestamp"].dt.tz_localize(None)
        return out

    def _get_regime(self, h1_df: pd.DataFrame, h1_idx: int) -> str:
        if self.regime_data is not None and len(self.regime_data) > 0:
            try:
                labels = self.regime_filter.classify(h1_df.iloc[: h1_idx + 1].copy(), self.regime_data)
                if len(labels) > 0:
                    return str(labels.iloc[-1])
            except Exception:
                pass
        return self._get_regime_fallback(h1_df, h1_idx)

    def _get_regime_fallback(self, h1_df: pd.DataFrame, h1_idx: int) -> str:
        atr = compute_atr(h1_df, h1_idx)
        close = float(h1_df["close"].iloc[h1_idx])
        vol_pct = (atr / max(close, 1e-9)) * 100.0
        fz, oz = self._get_live_regime_z()
        is_stress = vol_pct > 1.5
        if fz is not None and abs(fz) > 2.5:
            is_stress = True
        if oz is not None and abs(oz) > 2.5:
            is_stress = True
        if is_stress:
            return "STRESS"
        if vol_pct < 0.5:
            return "CHOPPY"
        return "TRENDING"

    def _get_live_regime_z(self) -> tuple[Optional[float], Optional[float]]:
        now = time.time()
        if now - self._regime_cache_time < 300:
            return self._regime_cache
        try:
            req = urllib.request.Request(
                "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                funding = float(json.loads(response.read().decode())[0]["fundingRate"])
            req2 = urllib.request.Request(
                "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req2, timeout=10) as response2:
                oi = float(json.loads(response2.read().decode())["openInterest"])
        except Exception:
            self._regime_cache = (None, None)
            self._regime_cache_time = now
            return self._regime_cache

        hist = self._regime_history
        hist["f"].append(funding)
        hist["o"].append(oi)
        hist["f"] = hist["f"][-168:]
        hist["o"] = hist["o"][-168:]
        fa = hist["f"]
        oa = hist["o"]
        fm = sum(fa) / len(fa)
        fs = (sum((x - fm) ** 2 for x in fa) / len(fa)) ** 0.5 if len(fa) > 1 else 1e-9
        fz = (funding - fm) / max(fs, 1e-9)
        if len(oa) > 1:
            od = [oa[i] - oa[i - 1] for i in range(1, len(oa))]
            om = sum(od) / len(od)
            os_ = (sum((x - om) ** 2 for x in od) / len(od)) ** 0.5 if len(od) > 1 else 1e-9
            oz = (oa[-1] - oa[-2]) / max(os_, 1e-9)
        else:
            oz = 0.0
        self._regime_cache = (fz, oz)
        self._regime_cache_time = now
        return self._regime_cache

    def _actual_live_match(self, decision_time: str) -> dict[str, Any]:
        rows = read_jsonl(self.live_log)
        matches = [row for row in rows if str(row.get("decision_time")) == str(decision_time)]
        if not matches:
            return {"matched": False}
        row = matches[-1]
        return {
            "matched": True,
            "action": row.get("action"),
            "reason": row.get("reason"),
            "direction": row.get("direction"),
            "strength": row.get("strength"),
            "vote_sum": row.get("vote_sum"),
            "local_time": row.get("local_time"),
        }

    def _log_payload(
        self,
        stage: str,
        h1_df: Optional[pd.DataFrame],
        h1_idx: Optional[int],
        m15_df: Optional[pd.DataFrame],
        m15_idx: Optional[int],
        **fields: Any,
    ) -> dict[str, Any]:
        h1_close_ts = fields.pop("h1_close_ts", None)
        if h1_df is not None and h1_idx is not None and h1_close_ts is None:
            h1_close_ts = pd.Timestamp(h1_df["timestamp"].iloc[h1_idx]) + pd.Timedelta(hours=1)
        m15_ts = None
        if m15_df is not None and m15_idx is not None and 0 <= int(m15_idx) < len(m15_df):
            m15_ts = pd.Timestamp(m15_df["timestamp"].iloc[int(m15_idx)])
        decision_ts = m15_ts or h1_close_ts or pd.Timestamp.utcnow()
        key = (str(pd.Timestamp(decision_ts)), str(stage))
        if key in self.processed_keys:
            return {"logged": False, "duplicate": True, "decision_time": str(decision_ts), "stage": stage}

        payload = {
            "local_time": datetime.utcnow().isoformat() + "Z",
            "probe_join_time": datetime.utcnow().isoformat() + "Z",
            "decision_time": str(pd.Timestamp(decision_ts)),
            "stage": stage,
            "action": f"champion_shadow_{stage}",
            "source": "frozen_t9_champion_shadow",
            "use_h4": bool(self.config.use_h4),
            "entry_quality_enabled": bool(self.config.entry_quality.enabled),
            "addon_policy_enabled": bool(self.config.addon_policy.enabled),
            "spread_bps": current_spread_bps(self.executor),
            **fields,
        }
        if h1_df is not None and h1_idx is not None and 0 <= int(h1_idx) < len(h1_df):
            payload["h1_timestamp"] = str(pd.Timestamp(h1_df["timestamp"].iloc[int(h1_idx)]))
            payload["h1_close_time"] = str(pd.Timestamp(h1_close_ts))
            payload["h1_idx"] = int(h1_idx)
            payload["h1_atr_percentile"] = h1_atr_percentile(h1_df, int(h1_idx))
        if m15_ts is not None:
            payload["m15_timestamp"] = str(m15_ts)
            payload["m15_idx"] = int(m15_idx)
        votes = [payload.get("h4_vote"), payload.get("h1_vote"), payload.get("m15_vote")]
        if all(v is not None for v in votes):
            payload["vote_sum"] = int(sum(int(v) for v in votes))
        if payload.get("price") is not None and payload.get("atr") is not None:
            payload["stop_width_bps_at_2atr"] = float(payload["atr"]) * 2.0 / max(float(payload["price"]), 1e-9) * 10000.0
        payload["actual_live"] = self._actual_live_match(str(payload["decision_time"]))
        append_jsonl(self.output_log, payload)
        self.processed_keys.add(key)
        return {"logged": True, "payload": payload}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", type=int, default=None)
    parser.add_argument("--terminal", type=str, default=None)
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--fetch-h1", type=int, default=300)
    parser.add_argument("--output-log", default=str(DEFAULT_SHADOW_LOG))
    parser.add_argument("--live-log", default=str(DEFAULT_LIVE_LOG))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    executor = MT5Executor(symbol=str(args.symbol), magic=20260517, deviation=20, terminal_path=args.terminal)
    if not executor.connect():
        raise RuntimeError("MT5 connection failed")
    try:
        if args.account is not None:
            info = executor.mt5.account_info() if executor.mt5 is not None else None
            actual = info.login if info else None
            if actual != int(args.account):
                raise RuntimeError(f"Wrong account: got {actual}, expected {args.account}")
        evaluator = ChampionShadowEvaluator(
            executor,
            output_log=Path(args.output_log),
            live_log=Path(args.live_log),
            fetch_h1=int(args.fetch_h1),
        )
        while True:
            try:
                result = evaluator.run_once()
                print(json.dumps(json_ready(result), sort_keys=True))
            except Exception as exc:
                print(json.dumps({"logged": False, "error": str(exc), "local_time": datetime.utcnow().isoformat() + "Z"}))
            if args.once:
                return 0
            time.sleep(max(int(args.interval), 1))
    finally:
        executor.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
