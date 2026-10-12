"""Leak-safe voting ensemble backtest utilities.

All timestamps from V3DataLoader are treated as bar-open timestamps.  A bar's
OHLC values are only available at its close_time.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_sandbox = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _sandbox not in sys.path:
    sys.path.insert(0, _sandbox)

from core.data_loader import V3DataLoader
from data.sources.regime_state import RegimeStateConfig, RegimeStateLoader
from models.trend.entry_quality_filter import EntryQualityConfig, EntryQualityFilter
from models.trend.event_gate import EventGate, EventGateConfig
from models.trend.execution_model import ExecutionCostConfig, ExecutionModel
from models.trend.external_spree_controller import ExternalSpreeConfig, ExternalSpreeController, ExternalSpreeState
from models.trend.regime_filter import RegimeFilter, RegimeFilterConfig, REGIME_TRENDING
from models.trend.fade_module import FadeModule, FadeModuleConfig, FadeClose
from models.trend.spree_addon_policy import SpreeAddonConfig, SpreeAddonDecision, SpreeAddonPolicy
from models.trend.spree_expansion_policy import SpreeExpansionConfig, SpreeExpansionPolicy
from models.trend.trade_spree import SpreePosition
from models.trend.trend_engine import TrendConfig, TrendEngine

YEARS = [
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
]

@dataclass
class ClosedTrade:
    position_id: int
    spree_id: int
    sequence: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    pnl_dollar: float
    pnl_r: float
    reason: str
    stop_distance: float
    gross_pnl_dollar: float = 0.0
    gross_pnl_r: float = 0.0
    spread_cost_r: float = 0.0
    slippage_cost_r: float = 0.0
    fee_cost_r: float = 0.0
    total_cost_r: float = 0.0
    mid_entry_price: float = 0.0
    mid_exit_price: float = 0.0
    lots: float = 0.0
    size_multiplier: float = 0.0
    risk_scale: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0

@dataclass
class EntryAudit:
    position_id: int
    spree_id: int
    sequence: int
    entry_time: pd.Timestamp
    signal_time: pd.Timestamp
    h1_idx: int
    h4_last_close: Optional[pd.Timestamp]
    direction: int
    price: float
    gated_allowed: bool
    strength: str
    h1_open: float = 0.0
    h1_close: float = 0.0
    regime: str = ""
    h4_vote: int = 0
    h1_vote: int = 0
    m15_vote: int = 0
    vote_sum: int = 0
    atr: float = 0.0
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    max_hold: int = 0
    lots: float = 0.0
    size_multiplier: float = 0.0
    risk_scale: float = 0.0
    opposite_first_r: Optional[float] = None
    quality_score: float = 0.0
    quality_reason: str = ""
    expansion_reason: str = ""
    addon_reason: str = ""
    addon_policy_mode: str = ""
    open_spree_r_at_add: float = 0.0
    peak_spree_r_at_add: float = 0.0
    giveback_from_peak_r_at_add: float = 0.0
    lead_mfe_r_at_add: float = 0.0
    bars_since_last_add_at_add: float = 0.0
    open_positions_at_add: float = 0.0
    worst_case_spree_loss_r_at_add: float = 0.0
    pre_entry_r: float = 0.0
    h1_body_r: float = 0.0
    m15_alignment: float = 0.0
    m15_range_expansion: float = 0.0
    h1_atr_percentile: float = 0.0

@dataclass
class BlockedEntryAudit:
    entry_time: pd.Timestamp
    h1_idx: int
    direction: int
    price: float
    strength: str
    regime: str
    quality_score: float
    block_reason: str
    h1_vote: int = 0
    m15_vote: int = 0
    vote_sum: int = 0
    pre_entry_r: float = 0.0
    h1_body_r: float = 0.0
    m15_alignment: float = 0.0
    m15_range_expansion: float = 0.0
    h1_atr_percentile: float = 0.0
    atr: float = 0.0
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    max_hold: int = 0
    same_direction_first_r: Optional[float] = None
    opposite_first_r: Optional[float] = None

@dataclass
class BlockedAddonAudit:
    entry_time: pd.Timestamp
    spree_id: int
    sequence: int
    direction: int
    price: float
    stop_distance: float
    stop_width_bps: float
    modeled_cost_bps: float
    modeled_cost_r: float
    block_reason: str
    counterfactual_r: Optional[float] = None
    addon_reason: str = ""
    total_spree_r: float = 0.0

@dataclass
class VotingResult:
    stats: List[tuple] = field(default_factory=list)
    strong_count: int = 0
    weak_count: int = 0
    balance: float = 10000.0
    entries: List[EntryAudit] = field(default_factory=list)
    blocked_entries: List[BlockedEntryAudit] = field(default_factory=list)
    blocked_addons: List[BlockedAddonAudit] = field(default_factory=list)
    trades: List[ClosedTrade] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    monthly_pnl: Dict[str, float] = field(default_factory=dict)
    monthly_r: Dict[str, float] = field(default_factory=dict)
    monthly_trades: Dict[str, int] = field(default_factory=dict)
    monthly_sprees: Dict[str, int] = field(default_factory=dict)
    monthly_closed_seen: set = field(default_factory=set)
    opened_ids: set = field(default_factory=set)
    closed_ids: set = field(default_factory=set)
    equity_curve: List[float] = field(default_factory=list)
    bars_evaluated: int = 0
    execution_rejections: Dict[str, int] = field(default_factory=dict)
    fade_trades: list = field(default_factory=list)
    fade_monthly_pnl: Dict[str, float] = field(default_factory=dict)
    fade_monthly_r: Dict[str, float] = field(default_factory=dict)

@dataclass
class VotingBacktestConfig:
    allow_pyramiding: bool = True
    first_trade_gate: bool = True
    kill_switch: bool = True
    use_h4: bool = True
    use_m15: bool = True
    use_event_gate: bool = True
    max_positions: int = 15
    weak_max_positions: int = 8
    delayed_pyramid_bars: int = 0
    sl_atr: Optional[float] = None
    tp_atr: Optional[float] = None
    max_hold: Optional[int] = None
    kill_window: int = 5
    kill_threshold: int = 3
    adaptive_kill: bool = False
    early_kill_window: int = 3
    early_kill_threshold: int = 2
    protected_kill_window: int = 5
    protected_kill_threshold: int = 3
    protect_after_r: float = 5.0
    protect_after_positions: int = 10
    max_spree_loss_r: Optional[float] = None
    protect_spree_peak_after_r: Optional[float] = None
    max_spree_giveback_r: Optional[float] = None
    max_open_spree_risk_r: Optional[float] = None
    relaxed_open_spree_risk_r: Optional[float] = None
    relax_open_risk_after_total_r: Optional[float] = None
    scheduled_max_open_spree_risk_r: Optional[float] = None
    scheduled_relaxed_open_spree_risk_r: Optional[float] = None
    scheduled_relax_open_risk_after_total_r: Optional[float] = None
    open_risk_active_months: Optional[set[str]] = None
    open_risk_active_month_numbers: Optional[set[int]] = None
    open_risk_active_quarters: Optional[set[str]] = None
    high_atr_risk_percentile: Optional[float] = None
    high_atr_max_open_spree_risk_r: Optional[float] = None
    high_atr_relaxed_open_spree_risk_r: Optional[float] = None
    high_atr_hold_percentile: Optional[float] = None
    high_atr_max_hold: Optional[int] = None
    high_atr_hold_min_sequence: int = 0
    high_atr_hold_after_total_r: Optional[float] = None
    high_atr_hold_active_months: Optional[set[str]] = None
    high_atr_hold_active_month_numbers: Optional[set[int]] = None
    high_atr_hold_active_quarters: Optional[set[str]] = None
    max_spree_entries: Optional[int] = None
    monthly_loss_pause_r: Optional[float] = None
    loss_spree_cooldown_bars: int = 0
    month_start_entry_skip_days: int = 0
    month_start_entry_skip_month_numbers: Optional[set[int]] = None
    risk_drawdown_start_pct: Optional[float] = None
    risk_drawdown_full_pct: Optional[float] = None
    risk_drawdown_floor_scale: float = 1.0
    addon_drawdown_start_pct: Optional[float] = None
    addon_drawdown_full_pct: Optional[float] = None
    addon_drawdown_floor_scale: float = 1.0
    invert_signal: bool = False
    reverse_after_first_loss: bool = False
    broker_volume_min: Optional[float] = None
    broker_volume_step: Optional[float] = None
    broker_volume_max: Optional[float] = None
    broker_volume_rounding_mode: str = "reject"
    broker_volume_floor_min_ratio: float = 0.0
    broker_volume_floor_min_total_spree_r: Optional[float] = None
    broker_volume_floor_min_peak_spree_r: Optional[float] = None
    broker_volume_floor_max_giveback_r: Optional[float] = None
    broker_volume_floor_min_lead_mfe_r: Optional[float] = None
    broker_volume_floor_min_m15_range: Optional[float] = None
    execution_profile: str = "zero"
    spread_bps: Optional[float] = None
    entry_slippage_bps: Optional[float] = None
    exit_slippage_bps: Optional[float] = None
    stop_slippage_bps: Optional[float] = None
    tp_slippage_bps: Optional[float] = None
    commission_bps: Optional[float] = None
    fixed_fee_usd: Optional[float] = None
    max_allowed_spread_bps: Optional[float] = None
    addon_max_cost_r: Optional[float] = None
    addon_min_stop_width_bps: Optional[float] = None
    addon_adaptive_spread_threshold_bps: Optional[float] = None
    addon_low_spread_min_stop_width_bps: Optional[float] = None
    addon_high_spread_max_cost_r: Optional[float] = None
    record_blocked_entry_counterfactuals: bool = True
    external_spree_enabled: bool = False
    disable_t9_addon_policy_when_external: bool = True
    external_danger_start_minutes: float = 120.0
    external_danger_end_minutes: float = 480.0
    external_late_tail_minutes: float = 960.0
    external_idle_expiry_minutes: float = 960.0
    external_min_stop_width_bps: float = 150.0
    external_late_tail_min_stop_width_bps: float = 200.0
    external_max_cost_r: float = 0.075
    external_min_open_r: float = 0.75
    external_danger_min_open_r: float = 1.5
    external_max_giveback_r: float = 1.0
    external_min_bars_between_adds: int = 1
    external_min_m15_range_expansion: float = 1.0
    external_min_directional_momentum_r: float = 0.0
    external_base_size_multiplier: float = 0.5
    external_strong_open_r: float = 1.5
    external_strong_size_multiplier: float = 1.0
    external_max_size_multiplier: float = 1.0
    external_mode: str = "combined"
    external_max_positions_override: Optional[int] = None
    entry_quality: EntryQualityConfig = field(default_factory=EntryQualityConfig)
    expansion_policy: SpreeExpansionConfig = field(default_factory=SpreeExpansionConfig)
    addon_policy: SpreeAddonConfig = field(default_factory=SpreeAddonConfig)
    fade_module: FadeModuleConfig = field(default_factory=FadeModuleConfig)

class VotingBacktest:
    """Shared leak-safe simulator for voting ensemble scripts."""

    def __init__(self, initial_balance: float = 10000.0, shift: int = 0,
                 config: Optional[VotingBacktestConfig] = None,
                 loader: Optional[V3DataLoader] = None,
                 regime_loader: Optional[RegimeStateLoader] = None,
                 data_cache: Optional[dict] = None,
                 state_cache: Optional[dict] = None):
        self.initial_balance = float(initial_balance)
        self.shift = int(shift)
        self.config = config or VotingBacktestConfig()
        self.loader = loader or V3DataLoader()
        self.regime_loader = regime_loader or RegimeStateLoader(RegimeStateConfig(source="csv"))
        self.data_cache = data_cache
        self.state_cache = state_cache
        self.rf = RegimeFilter(RegimeFilterConfig(allow_trading_in_stress=True))
        self.gate = EventGate(EventGateConfig(enabled=True))
        self.engine = TrendEngine(TrendConfig())
        self.entry_filter = EntryQualityFilter(self.config.entry_quality)
        self.expansion_policy = SpreeExpansionPolicy(self.config.expansion_policy)
        self.addon_policy = SpreeAddonPolicy(self.config.addon_policy)
        self.external_spree = ExternalSpreeController(self._external_spree_config())
        self.fade_module = FadeModule(self.config.fade_module)
        self.execution = ExecutionModel(
            ExecutionCostConfig.from_profile(
                self.config.execution_profile,
                spread_bps=self.config.spread_bps,
                entry_slippage_bps=self.config.entry_slippage_bps,
                exit_slippage_bps=self.config.exit_slippage_bps,
                stop_slippage_bps=self.config.stop_slippage_bps,
                tp_slippage_bps=self.config.tp_slippage_bps,
                commission_bps=self.config.commission_bps,
                fixed_fee_usd=self.config.fixed_fee_usd,
                max_allowed_spread_bps=self.config.max_allowed_spread_bps,
            )
        )

    def run(self, start: str, end: str, trade_start: Optional[str] = None) -> VotingResult:
        cache_key = (str(start), str(end))
        if self.data_cache is not None and cache_key in self.data_cache:
            data = self.data_cache[cache_key]
        else:
            data = self.loader.load_range(start, end)
            if self.data_cache is not None:
                self.data_cache[cache_key] = data
        h1 = data["h1"].copy()
        m15 = data["m15"].copy()
        m5 = data["m5"].copy()
        self._normalise_times(h1, m15, m5)
        self._apply_shift(h1, m15)

        if self.state_cache is not None and cache_key in self.state_cache:
            state = self.state_cache[cache_key]
        else:
            state = self.regime_loader.load_range(start, end)
            if self.state_cache is not None:
                self.state_cache[cache_key] = state
        regime_labels = self.rf.classify(h1, state)
        h1_atrs = pd.Series(h1["high"].values - h1["low"].values).rolling(24, min_periods=1).mean().values
        self._prepare_entry_quality_features(h1, h1_atrs)
        h1_ts = pd.to_datetime(h1["timestamp"].values)
        m15_ts = pd.to_datetime(m15["timestamp"].values)
        h1_map = [max(0, int(np.searchsorted(h1_ts, ts, side="right") - 1)) for ts in m15_ts]
        h4_full = self._build_h4(h1)

        result = VotingResult(balance=self.initial_balance, equity_curve=[self.initial_balance])
        positions: List[SpreePosition] = []
        closed_pnls: List[float] = []
        t9_entry_times: list = []  # M15 timestamps of accepted T9 entries
        spree_direction = 0
        spree_active = False
        first_closed = False
        max_positions = 15
        cstart = None
        cpos = 0
        cr = 0.0
        current_spree_id = 0
        current_spree_quality_score = 0.0
        current_spree_quality_reason = ""
        current_spree_quality_features = {}
        current_spree_last_add_i = -10**9
        current_spree_peak_r = 0.0
        loss_spree_cooldown_until_i = -1
        next_spree_id = 1
        next_position_id = 1
        trade_start_ts = pd.Timestamp(trade_start) if trade_start is not None else None

        for i in range(300, len(m15)):
            ts = pd.Timestamp(m15_ts[i])
            trade_window_open = trade_start_ts is None or ts >= trade_start_ts
            if trade_window_open:
                result.bars_evaluated += 1
            elif not positions:
                continue
            price = float(m15["open"].iloc[i])
            close_price = float(m15["close"].iloc[i])
            m15_high = float(m15["high"].iloc[i])
            m15_low = float(m15["low"].iloc[i])
            h1_idx = h1_map[i] - 1
            h4 = self._h4_for_signal(h4_full, ts) if self.config.use_h4 else None
            for pos in positions:
                pos.last_mark_price = price

            if positions or cstart is not None:
                just_closed = self._close_intrabar(positions, m5, ts, result)
                positions = [p for p in positions if not p.closed]
                for pos in just_closed:
                    result.balance += pos.pnl_dollar
                    result.equity_curve.append(result.balance)
                    cr += pos.pnl_r
                    cpos += 1
                    closed_pnls.append(pos.pnl_r)
                    self._record_monthly_close(result, pos)
                    if not first_closed:
                        first_closed = True
                        spree_active = pos.pnl_r > 0

                for pos in positions:
                    pos.bars_held += 1
                    sd = self._stop_distance(pos)
                    mark = price  # M15 close as mark
                    open_r_val = (mark - pos.entry_price) / max(sd, 1e-9) if pos.direction == 1 else (pos.entry_price - mark) / max(sd, 1e-9)
                    pos.mfe_r = max(pos.mfe_r, open_r_val)
                    pos.mae_r = min(pos.mae_r, open_r_val)

            # ── Fade module exit checking per M5 bar ──
            if self.fade_module.has_open_position():
                # Check all M5 bars since last M15 decision
                m5_start = max(0, i * 4) if i > 0 else 0
                m5_end = min((i + 1) * 4, len(m5))
                for m5_j in range(m5_start, m5_end):
                    if m5_j < len(m5):
                        m5_ts = pd.to_datetime(m5["timestamp"].iloc[m5_j]) if "timestamp" in m5.columns else ts
                        fc = self.fade_module.check_exit(
                            float(m5["high"].iloc[m5_j]),
                            float(m5["low"].iloc[m5_j]),
                            float(m5["close"].iloc[m5_j]),
                            m5_ts,
                            cost_bps=self.config.spread_bps or 5.0,
                            m5_idx=m5_j,
                        )
                        if fc is not None:
                            result.fade_trades.append(fc)
                            fade_month = str(fc.exit_time)[:7]
                            result.fade_monthly_pnl[fade_month] = result.fade_monthly_pnl.get(fade_month, 0.0) + fc.net_r * fc.risk_r
                            result.fade_monthly_r[fade_month] = result.fade_monthly_r.get(fade_month, 0.0) + fc.net_r

            if positions or cstart is not None:
                # Time-based exits for T9 positions
                for pos in positions:
                    if pos.bars_held >= pos.max_hold:
                        self._close_position(pos, close_price, "time_exit", ts, result)

                time_closed = [p for p in positions if p.closed]
                positions = [p for p in positions if not p.closed]
                for pos in time_closed:
                    result.balance += pos.pnl_dollar
                    result.equity_curve.append(result.balance)
                    cr += pos.pnl_r
                    cpos += 1
                    closed_pnls.append(pos.pnl_r)
                    self._record_monthly_close(result, pos)
                    if not first_closed:
                        first_closed = True
                        spree_active = pos.pnl_r > 0

                if self.config.first_trade_gate and first_closed and not spree_active:
                    if (
                        self.config.reverse_after_first_loss
                        and not positions
                        and h1_idx >= 0
                        and (not self.config.use_event_gate or self.gate.trading_allowed(ts))
                    ):
                        rev_dir = -int(spree_direction)
                        pos = self._build_position(
                            h1_atrs, h1_idx, result.balance, price, rev_dir, {"strength": "STRONG"}, ts,
                            next_position_id, current_spree_id, cpos,
                            risk_scale=self._risk_scale(result),
                            h1_atr_percentile=self._h1_atr_percentile(h1, h1_idx),
                            total_spree_r=cr,
                            entry_ts=ts,
                            result=result,
                        )
                        if pos is not None:
                            next_position_id += 1
                            positions = [pos]
                            spree_direction = rev_dir
                            spree_active = True
                            first_closed = True
                            max_positions = self.config.max_positions
                            self._record_entry(result, pos, ts, h1, h1_idx, h4, rev_dir, price, "REVERSAL", str(regime_labels.iloc[h1_idx] if h1_idx < len(regime_labels) else "TRENDING"), {"strength": "REVERSAL", "votes": {}}, m5)
                            current_spree_last_add_i = i
                            current_spree_peak_r = 0.0
                            continue
                    self._close_all(positions, price, ts, "first_trade_loss", result)
                    for p in positions:
                        result.balance += p.pnl_dollar
                        result.equity_curve.append(result.balance)
                        cr += p.pnl_r
                        cpos += 1
                        self._record_monthly_close(result, p)
                    dur = (ts - pd.Timestamp(cstart)).total_seconds() / 3600 if cstart else 0
                    result.stats.append((cr, cpos, True, dur, spree_direction))
                    loss_spree_cooldown_until_i = self._cooldown_until(i, cr)
                    if cstart:
                        result.monthly_sprees[str(pd.Timestamp(cstart))[:7]] = result.monthly_sprees.get(str(pd.Timestamp(cstart))[:7], 0) + 1
                    positions = []
                    closed_pnls = []
                    spree_direction = 0
                    spree_active = False
                    first_closed = False
                    cstart = None
                    cpos = 0
                    cr = 0.0
                    current_spree_quality_score = 0.0
                    current_spree_quality_reason = ""
                    current_spree_quality_features = {}
                    current_spree_last_add_i = -10**9
                    current_spree_peak_r = 0.0
                    continue

                total_spree_r = self._spree_total_r(cr, positions)
                current_spree_peak_r = max(current_spree_peak_r, total_spree_r)
                forced_reason = self._spree_force_close_reason(total_spree_r, current_spree_peak_r)
                if forced_reason:
                    self._close_all(positions, price, ts, forced_reason, result)
                    for p in positions:
                        result.balance += p.pnl_dollar
                        result.equity_curve.append(result.balance)
                        cr += p.pnl_r
                        cpos += 1
                        self._record_monthly_close(result, p)
                    dur = (ts - pd.Timestamp(cstart)).total_seconds() / 3600 if cstart else 0
                    result.stats.append((cr, cpos, True, dur, spree_direction))
                    loss_spree_cooldown_until_i = self._cooldown_until(i, cr)
                    if cstart:
                        result.monthly_sprees[str(pd.Timestamp(cstart))[:7]] = result.monthly_sprees.get(str(pd.Timestamp(cstart))[:7], 0) + 1
                    positions = []
                    closed_pnls = []
                    spree_direction = 0
                    spree_active = False
                    first_closed = False
                    cstart = None
                    cpos = 0
                    cr = 0.0
                    current_spree_quality_score = 0.0
                    current_spree_quality_reason = ""
                    current_spree_quality_features = {}
                    current_spree_last_add_i = -10**9
                    current_spree_peak_r = 0.0
                    continue

                if self.config.kill_switch and self._should_kill(closed_pnls, cr, cpos):
                    self._close_all(positions, price, ts, "spree_kill", result)
                    for p in positions:
                        result.balance += p.pnl_dollar
                        result.equity_curve.append(result.balance)
                        cr += p.pnl_r
                        cpos += 1
                        self._record_monthly_close(result, p)
                    dur = (ts - pd.Timestamp(cstart)).total_seconds() / 3600 if cstart else 0
                    result.stats.append((cr, cpos, True, dur, spree_direction))
                    loss_spree_cooldown_until_i = self._cooldown_until(i, cr)
                    if cstart:
                        result.monthly_sprees[str(pd.Timestamp(cstart))[:7]] = result.monthly_sprees.get(str(pd.Timestamp(cstart))[:7], 0) + 1
                    positions = []
                    closed_pnls = []
                    spree_direction = 0
                    spree_active = False
                    first_closed = False
                    cstart = None
                    cpos = 0
                    cr = 0.0
                    current_spree_quality_score = 0.0
                    current_spree_quality_reason = ""
                    current_spree_quality_features = {}
                    current_spree_last_add_i = -10**9
                    current_spree_peak_r = 0.0
                    continue

                if (self.config.allow_pyramiding and self.config.external_spree_enabled and spree_active and h1_idx >= 0
                        and len(positions) < max_positions):
                    if self.config.delayed_pyramid_bars and cstart is not None:
                        bars_since_start = int((ts - pd.Timestamp(cstart)).total_seconds() // (15 * 60))
                        if bars_since_start < self.config.delayed_pyramid_bars:
                            continue
                    bars_since_start = int((ts - pd.Timestamp(cstart)).total_seconds() // (15 * 60)) if cstart else 0
                    range_expansion = self._m15_range_expansion(m15, i)
                    stop_distance = self._planned_stop_distance(
                        h1_atrs, h1_idx, price, {"strength": "STRONG"}
                    )
                    stop_width_bps = stop_distance / max(float(price), 1e-9) * 10000.0
                    cost_bps = float(self.config.spread_bps or 0.0)
                    modeled_cost_r = cost_bps / max(stop_width_bps, 1e-9) if cost_bps > 0 else 0.0
                    directional_momentum = self._m15_directional_momentum(
                        m15, i, price, spree_direction, stop_distance
                    )
                    external_decision = self.external_spree.evaluate(
                        ExternalSpreeState(
                            positions=positions,
                            closed_pnls=closed_pnls,
                            realized_r=cr,
                            sequence=len(closed_pnls) + len(positions),
                            bars_since_start=bars_since_start,
                            bars_since_last_add=i - current_spree_last_add_i,
                            m15_range_expansion=range_expansion,
                            directional_momentum_r=directional_momentum,
                            peak_spree_r=current_spree_peak_r,
                            stop_width_bps=stop_width_bps,
                            modeled_cost_r=modeled_cost_r,
                            h1_atr_percentile=self._h1_atr_percentile(h1, h1_idx),
                        )
                    )
                    effective_max_positions = external_decision.max_positions_override or max_positions
                    risk_allowed = self._can_add_spree_risk(
                        cr, positions, self._spree_total_r(cr, positions), self._h1_atr_percentile(h1, h1_idx), ts
                    )
                    lifetime_allowed = self._can_add_spree_entry_count(closed_pnls, positions)
                    if len(positions) < effective_max_positions and external_decision.allow_add and risk_allowed and lifetime_allowed:
                        added = self._maybe_addon_enter(
                            h1, h4, regime_labels, h1_atrs, h1_idx, ts, result.balance,
                            price, positions, spree_direction, result, current_spree_id,
                            len(closed_pnls) + len(positions), next_position_id,
                            external_decision, current_spree_quality_score,
                            current_spree_quality_reason, current_spree_quality_features,
                            self._spree_total_r(cr, positions),
                            m5,
                            addon_policy_mode="external_spree",
                            expansion_reason="external",
                        )
                        if added:
                            next_position_id += 1
                            current_spree_last_add_i = i
                            continue

                t9_addons_enabled = not (
                    self.config.external_spree_enabled
                    and self.config.disable_t9_addon_policy_when_external
                )
                if (t9_addons_enabled and self.config.allow_pyramiding and self.config.addon_policy.enabled and spree_active and h1_idx >= 0
                        and len(positions) < max_positions and self._signal_context_ready(h4)):
                    if self.config.delayed_pyramid_bars and cstart is not None:
                        bars_since_start = int((ts - pd.Timestamp(cstart)).total_seconds() // (15 * 60))
                        if bars_since_start < self.config.delayed_pyramid_bars:
                            continue
                    bars_since_start = int((ts - pd.Timestamp(cstart)).total_seconds() // (15 * 60)) if cstart else 0
                    range_expansion = self._m15_range_expansion(m15, i)
                    stop_distance = self._planned_stop_distance(
                        h1_atrs, h1_idx, price, {"strength": "STRONG"}
                    )
                    directional_momentum = self._m15_directional_momentum(
                        m15, i, price, spree_direction, stop_distance
                    )
                    addon_decision = self.addon_policy.evaluate(
                        positions, closed_pnls, cr, len(closed_pnls) + len(positions),
                        bars_since_start, i - current_spree_last_add_i, range_expansion,
                        directional_momentum, current_spree_peak_r,
                        self._h1_atr_percentile(h1, h1_idx),
                        regime_label=str(regime),
                    )
                    effective_max_positions = addon_decision.max_positions_override or max_positions
                    risk_allowed = self._can_add_spree_risk(
                        cr, positions, self._spree_total_r(cr, positions), self._h1_atr_percentile(h1, h1_idx), ts
                    )
                    lifetime_allowed = self._can_add_spree_entry_count(closed_pnls, positions)
                    if len(positions) < effective_max_positions and addon_decision.allow_add and risk_allowed and lifetime_allowed:
                        added = self._maybe_addon_enter(
                            h1, h4, regime_labels, h1_atrs, h1_idx, ts, result.balance,
                            price, positions, spree_direction, result, current_spree_id,
                            len(closed_pnls) + len(positions), next_position_id,
                            addon_decision, current_spree_quality_score,
                            current_spree_quality_reason, current_spree_quality_features,
                            self._spree_total_r(cr, positions),
                            m5,
                        )
                        if added:
                            next_position_id += 1
                            current_spree_last_add_i = i
                            continue
                if (t9_addons_enabled and self.config.allow_pyramiding and spree_active and h1_idx >= 0
                        and len(positions) < max_positions and self._signal_context_ready(h4)):
                    if self.config.delayed_pyramid_bars and cstart is not None:
                        bars_since_start = int((ts - pd.Timestamp(cstart)).total_seconds() // (15 * 60))
                        if bars_since_start < self.config.delayed_pyramid_bars:
                            continue
                    bars_since_start = int((ts - pd.Timestamp(cstart)).total_seconds() // (15 * 60)) if cstart else 0
                    expansion_decision = self.expansion_policy.evaluate(
                        positions, closed_pnls, cr, len(closed_pnls) + len(positions),
                        bars_since_start, current_spree_quality_score,
                    )
                    effective_max_positions = expansion_decision.max_positions_override or max_positions
                    risk_allowed = self._can_add_spree_risk(
                        cr, positions, self._spree_total_r(cr, positions), self._h1_atr_percentile(h1, h1_idx), ts
                    )
                    lifetime_allowed = self._can_add_spree_entry_count(closed_pnls, positions)
                    if len(positions) >= effective_max_positions or not expansion_decision.allow_add or not risk_allowed or not lifetime_allowed:
                        if not positions:
                            dur = (ts - pd.Timestamp(cstart)).total_seconds() / 3600 if cstart else 0
                            result.stats.append((cr, cpos, False, dur, spree_direction))
                            loss_spree_cooldown_until_i = self._cooldown_until(i, cr)
                            if cstart:
                                result.monthly_sprees[str(pd.Timestamp(cstart))[:7]] = result.monthly_sprees.get(str(pd.Timestamp(cstart))[:7], 0) + 1
                            closed_pnls = []
                            spree_direction = 0
                            spree_active = False
                            first_closed = False
                            cstart = None
                            cpos = 0
                            cr = 0.0
                            current_spree_quality_score = 0.0
                            current_spree_quality_reason = ""
                            current_spree_quality_features = {}
                            current_spree_last_add_i = -10**9
                            current_spree_peak_r = 0.0
                        continue
                    added = self._maybe_enter(
                        h1, m15, h4, regime_labels, h1_atrs, h1_idx, i, ts,
                        result.balance, price, positions, spree_direction, result,
                        current_spree_id, len(closed_pnls) + len(positions), next_position_id,
                        expansion_decision.size_multiplier, expansion_decision.reason,
                        current_spree_quality_score, current_spree_quality_reason, current_spree_quality_features,
                        self._spree_total_r(cr, positions),
                        m5,
                    )
                    if added:
                        next_position_id += 1
                if (
                    not positions
                    and self.config.external_spree_enabled
                    and spree_active
                    and cstart is not None
                    and cr > 0
                    and (ts - pd.Timestamp(cstart)).total_seconds() / 60.0 < self.config.external_idle_expiry_minutes
                ):
                    continue

                if not positions:
                    dur = (ts - pd.Timestamp(cstart)).total_seconds() / 3600 if cstart else 0
                    result.stats.append((cr, cpos, False, dur, spree_direction))
                    loss_spree_cooldown_until_i = self._cooldown_until(i, cr)
                    if cstart:
                        result.monthly_sprees[str(pd.Timestamp(cstart))[:7]] = result.monthly_sprees.get(str(pd.Timestamp(cstart))[:7], 0) + 1
                    closed_pnls = []
                    spree_direction = 0
                    spree_active = False
                    first_closed = False
                    cstart = None
                    cpos = 0
                    cr = 0.0
                    current_spree_quality_score = 0.0
                    current_spree_quality_reason = ""
                    current_spree_quality_features = {}
                    current_spree_last_add_i = -10**9
                    current_spree_peak_r = 0.0
                    continue
                if positions:
                    continue

            if not trade_window_open:
                continue
            if h1_idx < 0 or not self._signal_context_ready(h4):
                continue
            if i < loss_spree_cooldown_until_i:
                continue
            regime = regime_labels.iloc[h1_idx] if h1_idx < len(regime_labels) else "TRENDING"
            t9_can_trade = self.rf.trading_allowed(regime) and regime != REGIME_TRENDING

            # ── Dormancy detection + Fade Module ──
            fade_cfg = self.config.fade_module
            recent_t9 = sum(1 for et in t9_entry_times
                           if (ts - et).total_seconds() <= fade_cfg.dormancy_entry_window_m15 * 15 * 60)
            is_t9_dormant = (not t9_can_trade) or (recent_t9 < fade_cfg.dormancy_entry_threshold)

            if is_t9_dormant and self.fade_module.can_enter() and self.fade_module.config.enabled:
                self.fade_module.on_dormancy_start(ts)
                # Evaluate fade candidate at current M5 bar
                # M5 bar corresponding to this M15 decision point
                m5_idx = i * 4 + 3 if len(m5) > i * 4 + 3 else len(m5) - 1  # last M5 of M15 bar
                if m5_idx > 0:
                    fade_dec = self.fade_module.evaluate(m5, m5_idx)
                    if fade_dec.enter:
                        # Execute fade entry via execution model
                        fade_pos = self.fade_module.open_position(fade_dec, ts)
                        if fade_pos is not None:
                            pass  # Position opened; exits checked in intrabar loop
                if not t9_can_trade:
                    continue  # T9 can't trade, skip to next bar

            if not t9_can_trade:
                continue
            if self.config.use_event_gate and not self.gate.trading_allowed(ts):
                continue
            if self._monthly_pause_active(result, ts):
                continue
            if self._month_start_entry_skip_active(ts):
                continue
            direction, ep = self._get_signal_vote(h1, m15, h4, h1_idx, i, regime)
            if direction == 0:
                continue
            strength = ep.get("strength", "WEAK")
            stop_distance = self._planned_stop_distance(h1_atrs, h1_idx, price, ep)
            quality = self.entry_filter.evaluate(
                h1, m15, h1_idx, i, direction, price, stop_distance, ep, str(regime),
                (not self.config.use_event_gate or self.gate.trading_allowed(ts)),
            )
            if self.config.entry_quality.enabled and not quality.allow_entry:
                self._record_blocked_entry(result, ts, h1_atrs, h1_idx, direction, price, strength, str(regime), ep, quality, m5)
                continue
            if strength == "STRONG":
                result.strong_count += 1
            else:
                result.weak_count += 1
            current_spree_id = next_spree_id
            next_spree_id += 1
            pos = self._build_position(
                h1_atrs, h1_idx, result.balance, price, direction, ep, ts,
                next_position_id, current_spree_id, 0,
                risk_scale=self._risk_scale(result),
                h1_atr_percentile=self._h1_atr_percentile(h1, h1_idx),
                total_spree_r=0.0,
                entry_ts=ts,
                result=result,
            )
            if pos is None:
                continue
            pos.entry_quality_score = quality.quality_score
            pos.entry_quality_reason = quality.block_reason
            pos.entry_quality_features = quality.features
            pos.expansion_reason = "initial"
            current_spree_quality_score = quality.quality_score
            current_spree_quality_reason = quality.block_reason
            current_spree_quality_features = quality.features
            current_spree_last_add_i = i
            current_spree_peak_r = 0.0
            next_position_id += 1
            positions = [pos]
            closed_pnls = []
            spree_direction = direction
            spree_active = not self.config.first_trade_gate
            first_closed = not self.config.first_trade_gate
            cstart = ts
            t9_entry_times.append(ts)
            self.fade_module.on_t9_activated()
            cpos = 0
            cr = 0.0
            max_positions = self.config.max_positions if strength == "STRONG" else self.config.weak_max_positions
            self._record_entry(result, pos, ts, h1, h1_idx, h4, direction, price, strength, str(regime), ep, m5)

        if positions:
            final_ts = pd.Timestamp(m15["close_time"].iloc[-1])
            final_price = float(m15["close"].iloc[-1])
            self._close_all(positions, final_price, final_ts, "end_of_data", result)
            for p in positions:
                result.balance += p.pnl_dollar
                result.equity_curve.append(result.balance)
                cr += p.pnl_r
                cpos += 1
                self._record_monthly_close(result, p)
            dur = (final_ts - pd.Timestamp(cstart)).total_seconds() / 3600 if cstart else 0
            result.stats.append((cr, cpos, False, dur, spree_direction))
            if cstart:
                result.monthly_sprees[str(pd.Timestamp(cstart))[:7]] = result.monthly_sprees.get(str(pd.Timestamp(cstart))[:7], 0) + 1

        # Force-close open fade position at end of data
        if self.fade_module.has_open_position():
            last_close = float(m5["close"].iloc[-1]) if len(m5) > 0 else 0.0
            last_ts = pd.to_datetime(m15["close_time"].iloc[-1]) if len(m15) > 0 else pd.Timestamp.now()
            # Force exit via timeout: override max_hold
            pos = self.fade_module._open_pos
            if pos is not None:
                risk_dist = max(pos.risk_distance, 1e-9)
                gross_r = (pos.entry_price - last_close) / risk_dist if pos.direction == -1 else (last_close - pos.entry_price) / risk_dist
                entry_cost = pos.entry_price * (self.config.spread_bps or 5.0) / 10000.0
                exit_cost = last_close * (self.config.spread_bps or 5.0) / 10000.0
                cost_r = (entry_cost + exit_cost) / risk_dist
                net_r = gross_r - cost_r
                fc = FadeClose(
                    direction=pos.direction, entry_price=pos.entry_price, exit_price=last_close,
                    entry_time=pos.entry_time, exit_time=last_ts, exit_reason="end_of_data",
                    gross_r=gross_r, net_r=net_r, risk_r=pos.risk_r, features=pos.features,
                )
                self.fade_module.closed_trades.append(fc)
                self.fade_module._open_pos = None
                result.fade_trades.append(fc)
                fade_month = str(fc.exit_time)[:7]
                result.fade_monthly_pnl[fade_month] = result.fade_monthly_pnl.get(fade_month, 0.0) + fc.net_r * fc.risk_r

        self._reconcile_spree_stats(result)
        self._audit_result(result)
        return result

    def _maybe_enter(self, h1, m15, h4, regimes, h1_atrs, h1_idx, m15_idx, ts, balance, price, positions, spree_direction, result, spree_id, sequence, position_id, size_multiplier: float = 1.0, expansion_reason: str = "", entry_quality_score: float = 0.0, entry_quality_reason: str = "", entry_quality_features: Optional[dict] = None, total_spree_r: float = 0.0, m5: Optional[pd.DataFrame] = None) -> bool:
        if self.config.use_event_gate and not self.gate.trading_allowed(ts):
            return False
        regime = regimes.iloc[h1_idx] if h1_idx < len(regimes) else "TRENDING"
        direction, ep = self._get_signal_vote(h1, m15, h4, h1_idx, m15_idx, regime)
        if direction != spree_direction:
            return False
        if int(sequence) > 0:
            atr = float(h1_atrs[h1_idx]) if h1_idx < len(h1_atrs) else price * 0.01
            stop_distance = self._planned_stop_distance(h1_atrs, h1_idx, price, ep)
            max_hold = int(self.config.max_hold if self.config.max_hold is not None else ep.get("max_hold", 24))
            max_hold = self._position_max_hold(max_hold, int(sequence), self._h1_atr_percentile(h1, h1_idx), total_spree_r, ts)
            if not self._addon_cost_gate_allows(
                result, ts, spree_id, sequence, direction, price, stop_distance,
                expansion_reason, total_spree_r, atr, ep, max_hold, m5
            ):
                return False
        pos = self._build_position(
            h1_atrs, h1_idx, balance, price, direction, ep, ts, position_id, spree_id, sequence,
            size_multiplier, self._risk_scale(result), self._h1_atr_percentile(h1, h1_idx),
            total_spree_r,
            ts,
            result=result,
        )
        if pos is None:
            return False
        pos.entry_quality_score = float(entry_quality_score)
        pos.entry_quality_reason = str(entry_quality_reason)
        pos.entry_quality_features = entry_quality_features or {}
        pos.expansion_reason = expansion_reason
        positions.append(pos)
        self._record_entry(result, pos, ts, h1, h1_idx, h4, direction, price, ep.get("strength", "WEAK"), str(regime), ep, None)
        return True

    def _maybe_addon_enter(self, h1, h4, regimes, h1_atrs, h1_idx, ts, balance, price,
                           positions, spree_direction, result, spree_id, sequence, position_id,
                           decision: SpreeAddonDecision, entry_quality_score: float = 0.0,
                           entry_quality_reason: str = "", entry_quality_features: Optional[dict] = None,
                           total_spree_r: float = 0.0, m5: Optional[pd.DataFrame] = None,
                           addon_policy_mode: Optional[str] = None,
                           expansion_reason: str = "addon") -> bool:
        if self.config.use_event_gate and not self.gate.trading_allowed(ts):
            return False
        if spree_direction == 0:
            return False
        regime = regimes.iloc[h1_idx] if h1_idx < len(regimes) else "TRENDING"
        ep = self.engine._regime_params(regime)
        ep["strength"] = "ADDON"
        ep["votes"] = {"h4": 0, "h1": 0, "m15": 0, "sum": 0}
        atr = float(h1_atrs[h1_idx]) if h1_idx < len(h1_atrs) else price * 0.01
        stop_distance = self._planned_stop_distance(h1_atrs, h1_idx, price, ep)
        max_hold = int(self.config.max_hold if self.config.max_hold is not None else ep.get("max_hold", 24))
        max_hold = self._position_max_hold(max_hold, int(sequence), self._h1_atr_percentile(h1, h1_idx), total_spree_r, ts)
        if not self._addon_cost_gate_allows(
            result, ts, spree_id, sequence, int(spree_direction), price, stop_distance,
            decision.reason, total_spree_r, atr, ep, max_hold, m5
        ):
            return False
        addon_size_multiplier = decision.size_multiplier * self._addon_size_scale(result)
        pos = self._build_position(
            h1_atrs, h1_idx, balance, price, int(spree_direction), ep, ts,
            position_id, spree_id, sequence, addon_size_multiplier, self._risk_scale(result),
            self._h1_atr_percentile(h1, h1_idx), total_spree_r, ts,
            decision.features,
            result=result,
        )
        if pos is None:
            return False
        pos.entry_quality_score = float(entry_quality_score)
        pos.entry_quality_reason = str(entry_quality_reason)
        addon_quality_features = dict(entry_quality_features or {})
        addon_quality_features.update({
            "m15_range_expansion": float(decision.features.get("m15_range_expansion", 0.0)),
            "h1_atr_percentile": float(decision.features.get("h1_atr_percentile", 0.0)),
        })
        pos.entry_quality_features = addon_quality_features
        pos.expansion_reason = expansion_reason
        pos.addon_reason = decision.reason
        pos.addon_policy_mode = addon_policy_mode or self.config.addon_policy.mode
        pos.open_spree_r_at_add = float(decision.features.get("total_open_realized_r", 0.0))
        pos.peak_spree_r_at_add = float(decision.features.get("peak_spree_r", 0.0))
        pos.giveback_from_peak_r_at_add = float(decision.features.get("giveback_from_peak_r", 0.0))
        pos.lead_mfe_r_at_add = float(decision.features.get("lead_mfe_r", 0.0))
        pos.bars_since_last_add_at_add = float(decision.features.get("bars_since_last_add", 0.0))
        pos.open_positions_at_add = float(decision.features.get("open_count", 0.0))
        pos.worst_case_spree_loss_r_at_add = float(decision.features.get("worst_case_loss_after_next_r", 0.0))
        positions.append(pos)
        self._record_entry(result, pos, ts, h1, h1_idx, h4, int(spree_direction), price, "ADDON", str(regime), ep, None)
        return True

    def _addon_cost_gate_allows(
        self,
        result: VotingResult,
        ts: pd.Timestamp,
        spree_id: int,
        sequence: int,
        direction: int,
        price: float,
        stop_distance: float,
        addon_reason: str,
        total_spree_r: float,
        atr: float,
        ep: dict,
        max_hold: int,
        m5: Optional[pd.DataFrame],
    ) -> bool:
        if int(sequence) <= 0:
            return True
        stop_width_bps = float(stop_distance) / max(float(price), 1e-9) * 10000.0
        cost_bps = float(self.config.spread_bps or 0.0)
        modeled_cost_r = cost_bps / max(stop_width_bps, 1e-9) if cost_bps > 0 else 0.0
        max_cost = self.config.addon_max_cost_r
        min_width = self.config.addon_min_stop_width_bps
        adaptive_threshold = self.config.addon_adaptive_spread_threshold_bps
        if adaptive_threshold is not None:
            if cost_bps <= float(adaptive_threshold):
                if self.config.addon_low_spread_min_stop_width_bps is not None:
                    min_width = float(self.config.addon_low_spread_min_stop_width_bps)
            elif self.config.addon_high_spread_max_cost_r is not None:
                max_cost = float(self.config.addon_high_spread_max_cost_r)
        block_reason = ""
        if max_cost is not None and modeled_cost_r > float(max_cost):
            block_reason = "addon_cost_r_too_high"
        if not block_reason and min_width is not None and stop_width_bps < float(min_width):
            block_reason = "addon_stop_width_too_narrow"
        if not block_reason:
            return True

        counterfactual_r = None
        if m5 is not None:
            counterfactual_r = self._simulate_first_r(
                int(direction),
                pd.Timestamp(ts),
                float(price),
                float(atr),
                float(self.config.sl_atr if self.config.sl_atr is not None else ep.get("sl_atr", 2.0)),
                float(self.config.tp_atr if self.config.tp_atr is not None else ep.get("tp_atr", 3.5)),
                int(max_hold),
                m5,
            )
        result.blocked_addons.append(
            BlockedAddonAudit(
                entry_time=pd.Timestamp(ts),
                spree_id=int(spree_id),
                sequence=int(sequence),
                direction=int(direction),
                price=float(price),
                stop_distance=float(stop_distance),
                stop_width_bps=float(stop_width_bps),
                modeled_cost_bps=float(cost_bps),
                modeled_cost_r=float(modeled_cost_r),
                block_reason=block_reason,
                counterfactual_r=counterfactual_r,
                addon_reason=str(addon_reason),
                total_spree_r=float(total_spree_r),
            )
        )
        return False

    def _build_position(
        self,
        h1_atrs,
        h1_idx,
        balance,
        price,
        direction,
        ep,
        ts,
        position_id,
        spree_id,
        sequence,
        size_multiplier: float = 1.0,
        risk_scale: float = 1.0,
        h1_atr_percentile: Optional[float] = None,
        total_spree_r: float = 0.0,
        entry_ts: Optional[pd.Timestamp] = None,
        floor_context: Optional[dict] = None,
        result: Optional[VotingResult] = None,
    ) -> Optional[SpreePosition]:
        atr = float(h1_atrs[h1_idx]) if h1_idx < len(h1_atrs) else price * 0.01
        lots = balance * 0.25 / max((atr * 16.0) / price, 0.05) / price
        lots = float(np.clip(lots, balance * 0.002 / price, balance * 0.10 / price))
        if ep.get("strength", "WEAK") == "WEAK":
            lots *= 0.5
        lots *= max(float(size_multiplier), 0.0)
        lots *= max(float(risk_scale), 0.0)
        lots = self._normalise_lots(lots, int(sequence), total_spree_r, floor_context)
        if lots <= 0:
            self._record_execution_rejection(result, "volume_rejected")
            return None
        if not self.execution.spread_allowed():
            self._record_execution_rejection(result, "spread_gate")
            return None
        sl_m = float(self.config.sl_atr if self.config.sl_atr is not None else ep.get("sl_atr", 2.0))
        tp_m = float(self.config.tp_atr if self.config.tp_atr is not None else ep.get("tp_atr", 3.5))
        max_hold = int(self.config.max_hold if self.config.max_hold is not None else ep.get("max_hold", 24))
        max_hold = self._position_max_hold(max_hold, int(sequence), h1_atr_percentile, total_spree_r, entry_ts)
        sl = price - direction * atr * sl_m
        tp = price + direction * atr * tp_m
        entry_fill = self.execution.entry_fill(price, direction, lots)
        pos = SpreePosition(str(ts), entry_fill.fill_price, direction, lots, atr, sl, tp, max_hold)
        pos.position_id = int(position_id)
        pos.spree_id = int(spree_id)
        pos.sequence = int(sequence)
        pos.last_checked_time = pd.Timestamp(ts)
        pos.mid_entry_price = float(price)
        pos.entry_spread_cost = float(entry_fill.spread_cost)
        pos.entry_slippage_cost = float(entry_fill.slippage_cost)
        pos.entry_fee = float(entry_fill.fee)
        pos.stop_distance = abs(float(pos.entry_price) - sl)
        pos.sl_atr_mult = sl_m
        pos.tp_atr_mult = tp_m
        pos.size_multiplier = float(size_multiplier)
        pos.risk_scale = float(risk_scale)
        return pos

    @staticmethod
    def _record_execution_rejection(result: Optional[VotingResult], reason: str) -> None:
        if result is None:
            return
        result.execution_rejections[reason] = result.execution_rejections.get(reason, 0) + 1

    def _normalise_lots(
        self,
        lots: float,
        sequence: int = 0,
        total_spree_r: float = 0.0,
        floor_context: Optional[dict] = None,
    ) -> float:
        lots = float(lots)
        if lots <= 0:
            return 0.0
        original_lots = lots
        max_volume = self.config.broker_volume_max
        if max_volume is not None and max_volume > 0:
            lots = min(lots, float(max_volume))
        step = self.config.broker_volume_step
        if step is not None and step > 0:
            lots = float(np.floor(lots / float(step) + 1e-12) * float(step))
            lots = round(lots, 10)
        min_volume = self.config.broker_volume_min
        if min_volume is not None and min_volume > 0 and lots < float(min_volume):
            mode = str(getattr(self.config, "broker_volume_rounding_mode", "reject"))
            should_floor = mode == "floor_to_min" or (
                mode == "floor_addons_to_min" and int(sequence) > 0
            )
            min_ratio = max(float(getattr(self.config, "broker_volume_floor_min_ratio", 0.0)), 0.0)
            large_enough_to_floor = original_lots >= float(min_volume) * min_ratio
            if should_floor and original_lots > 0 and large_enough_to_floor:
                min_total_spree_r = self.config.broker_volume_floor_min_total_spree_r
                if min_total_spree_r is not None and float(total_spree_r) < float(min_total_spree_r):
                    return 0.0
                if not self._broker_floor_context_allowed(floor_context):
                    return 0.0
                return float(min_volume)
            return 0.0
        return lots

    def _broker_floor_context_allowed(self, floor_context: Optional[dict]) -> bool:
        checks = (
            self.config.broker_volume_floor_min_peak_spree_r,
            self.config.broker_volume_floor_max_giveback_r,
            self.config.broker_volume_floor_min_lead_mfe_r,
            self.config.broker_volume_floor_min_m15_range,
        )
        if all(value is None for value in checks):
            return True
        ctx = floor_context or {}
        if (
            self.config.broker_volume_floor_min_peak_spree_r is not None
            and float(ctx.get("peak_spree_r", 0.0)) < float(self.config.broker_volume_floor_min_peak_spree_r)
        ):
            return False
        if (
            self.config.broker_volume_floor_max_giveback_r is not None
            and float(ctx.get("giveback_from_peak_r", 0.0)) > float(self.config.broker_volume_floor_max_giveback_r)
        ):
            return False
        if (
            self.config.broker_volume_floor_min_lead_mfe_r is not None
            and float(ctx.get("lead_mfe_r", 0.0)) < float(self.config.broker_volume_floor_min_lead_mfe_r)
        ):
            return False
        if (
            self.config.broker_volume_floor_min_m15_range is not None
            and float(ctx.get("m15_range_expansion", 0.0)) < float(self.config.broker_volume_floor_min_m15_range)
        ):
            return False
        return True

    def _risk_scale(self, result: VotingResult) -> float:
        return self._drawdown_scale(
            result,
            self.config.risk_drawdown_start_pct,
            self.config.risk_drawdown_full_pct,
            self.config.risk_drawdown_floor_scale,
        )

    def _addon_size_scale(self, result: VotingResult) -> float:
        return self._drawdown_scale(
            result,
            self.config.addon_drawdown_start_pct,
            self.config.addon_drawdown_full_pct,
            self.config.addon_drawdown_floor_scale,
        )

    def _drawdown_scale(
        self,
        result: VotingResult,
        start_pct: Optional[float],
        full_pct: Optional[float],
        floor_scale: float,
    ) -> float:
        if start_pct is None:
            return 1.0
        peak = max(result.equity_curve) if result.equity_curve else self.initial_balance
        dd_pct = (float(peak) - float(result.balance)) / max(float(peak), 1e-9) * 100.0
        start = max(float(start_pct), 0.0)
        if dd_pct <= start:
            return 1.0
        floor = float(np.clip(floor_scale, 0.0, 1.0))
        full = full_pct
        if full is None or float(full) <= start:
            return floor
        progress = min(max((dd_pct - start) / (float(full) - start), 0.0), 1.0)
        return float(1.0 - progress * (1.0 - floor))

    def _prepare_entry_quality_features(self, h1: pd.DataFrame, h1_atrs) -> None:
        lookback = max(int(self.config.entry_quality.h1_atr_percentile_lookback), 1)
        col = f"_entry_h1_atr_percentile_{lookback}"
        if col in h1.columns:
            h1["_entry_h1_atr_percentile"] = h1[col]
            return
        atrs = np.asarray(h1_atrs, dtype=float)
        pct = np.full(len(atrs), 0.5, dtype=float)
        for idx in range(len(atrs)):
            start = max(0, idx - lookback)
            hist = atrs[start:idx]
            hist = hist[np.isfinite(hist)]
            if len(hist) >= 10 and np.isfinite(atrs[idx]):
                pct[idx] = float(np.mean(hist <= atrs[idx]))
        h1[col] = pct
        h1["_entry_h1_atr_percentile"] = pct

    def _planned_stop_distance(self, h1_atrs, h1_idx, price: float, ep: dict) -> float:
        atr = float(h1_atrs[h1_idx]) if h1_idx < len(h1_atrs) else price * 0.01
        sl_m = float(self.config.sl_atr if self.config.sl_atr is not None else ep.get("sl_atr", 2.0))
        return max(abs(atr * sl_m), 1e-9)

    @staticmethod
    def _m15_range_expansion(m15: pd.DataFrame, m15_idx: int, recent: int = 8, baseline: int = 32) -> float:
        if m15_idx <= 1:
            return 0.0
        end = int(m15_idx)
        recent_start = max(0, end - recent)
        base_start = max(0, end - baseline)
        recent_ranges = (m15["high"].iloc[recent_start:end] - m15["low"].iloc[recent_start:end]).astype(float)
        base_ranges = (m15["high"].iloc[base_start:end] - m15["low"].iloc[base_start:end]).astype(float)
        recent_mean = float(recent_ranges.mean()) if len(recent_ranges) else 0.0
        base_mean = float(base_ranges.mean()) if len(base_ranges) else 0.0
        return recent_mean / max(base_mean, 1e-9)

    @staticmethod
    def _m15_directional_momentum(m15: pd.DataFrame, m15_idx: int, price: float,
                                  direction: int, stop_distance: float,
                                  lookback: int = 4) -> float:
        if m15_idx <= 0 or direction == 0:
            return 0.0
        ref_idx = max(0, int(m15_idx) - int(lookback))
        ref_price = float(m15["open"].iloc[ref_idx])
        return float(direction) * (float(price) - ref_price) / max(float(stop_distance), 1e-9)

    def _close_intrabar(self, positions: List[SpreePosition], m5: pd.DataFrame, decision_time: pd.Timestamp, result: VotingResult) -> List[SpreePosition]:
        closed = []
        m5_close = self._m5_close_values(m5)
        for pos in positions:
            if pos.closed:
                continue
            start = pd.Timestamp(getattr(pos, "last_checked_time", pos.entry_time))
            start_idx = int(np.searchsorted(m5_close, np.datetime64(start), side="right"))
            end_idx = int(np.searchsorted(m5_close, np.datetime64(decision_time), side="right"))
            for j in range(start_idx, end_idx):
                bar_close = pd.Timestamp(m5["close_time"].iloc[j])
                if bar_close > decision_time:
                    result.issues.append(f"M5_FUTURE_CHECK:{bar_close}>{decision_time}")
                hi = float(m5["high"].iloc[j])
                lo = float(m5["low"].iloc[j])
                sd = self._stop_distance(pos)
                if pos.direction == 1:
                    pos.mfe_r = max(pos.mfe_r, (hi - pos.entry_price) / max(sd, 1e-9))
                    pos.mae_r = min(pos.mae_r, (lo - pos.entry_price) / max(sd, 1e-9))
                else:
                    pos.mfe_r = max(pos.mfe_r, (pos.entry_price - lo) / max(sd, 1e-9))
                    pos.mae_r = min(pos.mae_r, (pos.entry_price - hi) / max(sd, 1e-9))
                if pos.direction == 1:
                    if lo <= pos.sl_price:
                        stop_mid = self._stop_fill_mid(pos, m5, j)
                        self._close_position(pos, stop_mid, "sl_hit", bar_close, result)
                        closed.append(pos)
                        break
                    if hi >= pos.tp_price:
                        self._close_position(pos, pos.tp_price, "tp_hit", bar_close, result)
                        closed.append(pos)
                        break
                else:
                    if hi >= pos.sl_price:
                        stop_mid = self._stop_fill_mid(pos, m5, j)
                        self._close_position(pos, stop_mid, "sl_hit", bar_close, result)
                        closed.append(pos)
                        break
                    if lo <= pos.tp_price:
                        self._close_position(pos, pos.tp_price, "tp_hit", bar_close, result)
                        closed.append(pos)
                        break
            if not pos.closed:
                pos.last_checked_time = pd.Timestamp(decision_time)
        return closed

    def _close_position(self, pos: SpreePosition, exit_price: float, reason: str, exit_time: pd.Timestamp, result: VotingResult) -> None:
        pos.closed = True
        mid_exit_price = float(exit_price)
        exit_fill = self.execution.exit_fill(mid_exit_price, pos.direction, pos.lots, reason)
        pos.close_price = float(exit_fill.fill_price)
        pos.close_reason = reason
        pos.close_time = pd.Timestamp(exit_time)
        sd = self._stop_distance(pos)
        mid_entry_price = float(getattr(pos, "mid_entry_price", pos.entry_price))
        if pos.direction == 1:
            gross_pnl_dollar = (mid_exit_price - mid_entry_price) * pos.lots
        else:
            gross_pnl_dollar = (mid_entry_price - mid_exit_price) * pos.lots
        if pos.direction == 1:
            pos.pnl_dollar = (pos.close_price - pos.entry_price) * pos.lots
        else:
            pos.pnl_dollar = (pos.entry_price - pos.close_price) * pos.lots
        entry_fee = float(getattr(pos, "entry_fee", 0.0))
        entry_spread = float(getattr(pos, "entry_spread_cost", 0.0))
        entry_slippage = float(getattr(pos, "entry_slippage_cost", 0.0))
        fee_cost = entry_fee + float(exit_fill.fee)
        spread_cost = entry_spread + float(exit_fill.spread_cost)
        slippage_cost = entry_slippage + float(exit_fill.slippage_cost)
        pos.pnl_dollar -= fee_cost
        pos.pnl_r = pos.pnl_dollar / max(sd * pos.lots, 1e-9)
        gross_pnl_r = gross_pnl_dollar / max(sd * pos.lots, 1e-9)
        spread_cost_r = spread_cost / max(sd * pos.lots, 1e-9)
        slippage_cost_r = slippage_cost / max(sd * pos.lots, 1e-9)
        fee_cost_r = fee_cost / max(sd * pos.lots, 1e-9)
        total_cost_r = gross_pnl_r - pos.pnl_r
        pos.gross_pnl_dollar = float(gross_pnl_dollar)
        pos.gross_pnl_r = float(gross_pnl_r)
        pos.spread_cost_r = float(spread_cost_r)
        pos.slippage_cost_r = float(slippage_cost_r)
        pos.fee_cost_r = float(fee_cost_r)
        pos.total_cost_r = float(total_cost_r)
        pos.mid_exit_price = float(mid_exit_price)
        position_id = int(getattr(pos, "position_id", -1))
        if position_id in result.closed_ids:
            result.issues.append(f"DUPLICATE_CLOSE:{position_id}")
        result.closed_ids.add(position_id)
        result.trades.append(
            ClosedTrade(
                position_id=position_id,
                spree_id=int(getattr(pos, "spree_id", -1)),
                sequence=int(getattr(pos, "sequence", -1)),
                entry_time=pd.Timestamp(pos.entry_time),
                exit_time=pos.close_time,
                direction=pos.direction,
                entry_price=pos.entry_price,
                exit_price=pos.close_price,
                pnl_dollar=pos.pnl_dollar,
                pnl_r=pos.pnl_r,
                reason=reason,
                stop_distance=sd,
                gross_pnl_dollar=float(gross_pnl_dollar),
                gross_pnl_r=float(gross_pnl_r),
                spread_cost_r=float(spread_cost_r),
                slippage_cost_r=float(slippage_cost_r),
                fee_cost_r=float(fee_cost_r),
                total_cost_r=float(total_cost_r),
                mid_entry_price=float(mid_entry_price),
                mid_exit_price=float(mid_exit_price),
                lots=float(getattr(pos, "lots", 0.0)),
                size_multiplier=float(getattr(pos, "size_multiplier", 0.0)),
                risk_scale=float(getattr(pos, "risk_scale", 0.0)),
                mfe_r=float(getattr(pos, "mfe_r", 0.0)),
                mae_r=float(getattr(pos, "mae_r", 0.0)),
            )
        )

    def _close_all(self, positions: List[SpreePosition], price: float, ts: pd.Timestamp, reason: str, result: VotingResult) -> None:
        for pos in positions:
            if not pos.closed:
                self._close_position(pos, price, reason, ts, result)

    @staticmethod
    def _stop_fill_mid(pos: SpreePosition, m5: pd.DataFrame, idx: int) -> float:
        stop = float(pos.sl_price)
        if "open" not in m5.columns:
            return stop
        bar_open = float(m5["open"].iloc[idx])
        if int(pos.direction) == 1:
            return min(stop, bar_open)
        return max(stop, bar_open)

    def _should_kill(self, closed_pnls: List[float], spree_r: float = 0.0, closed_count: int = 0) -> bool:
        if self.config.adaptive_kill:
            protected = (
                spree_r >= self.config.protect_after_r
                or closed_count >= self.config.protect_after_positions
            )
            window = self.config.protected_kill_window if protected else self.config.early_kill_window
            threshold = self.config.protected_kill_threshold if protected else self.config.early_kill_threshold
        else:
            window = self.config.kill_window
            threshold = self.config.kill_threshold
        if len(closed_pnls) < window:
            return False
        return sum(1 for r in closed_pnls[-window:] if r <= 0) >= threshold

    def _spree_total_r(self, realized_r: float, positions: List[SpreePosition]) -> float:
        return float(realized_r) + self._spree_open_r(positions)

    def _external_spree_config(self) -> ExternalSpreeConfig:
        return ExternalSpreeConfig(
            enabled=bool(self.config.external_spree_enabled),
            mode=str(self.config.external_mode),
            disable_t9_addon_policy_when_external=bool(self.config.disable_t9_addon_policy_when_external),
            danger_start_minutes=float(self.config.external_danger_start_minutes),
            danger_end_minutes=float(self.config.external_danger_end_minutes),
            late_tail_minutes=float(self.config.external_late_tail_minutes),
            idle_expiry_minutes=float(self.config.external_idle_expiry_minutes),
            min_stop_width_bps=float(self.config.external_min_stop_width_bps),
            late_tail_min_stop_width_bps=float(self.config.external_late_tail_min_stop_width_bps),
            max_cost_r=float(self.config.external_max_cost_r),
            min_open_r=float(self.config.external_min_open_r),
            danger_min_open_r=float(self.config.external_danger_min_open_r),
            max_giveback_r=float(self.config.external_max_giveback_r),
            min_bars_between_adds=int(self.config.external_min_bars_between_adds),
            min_m15_range_expansion=float(self.config.external_min_m15_range_expansion),
            min_directional_momentum_r=float(self.config.external_min_directional_momentum_r),
            base_size_multiplier=float(self.config.external_base_size_multiplier),
            strong_open_r=float(self.config.external_strong_open_r),
            strong_size_multiplier=float(self.config.external_strong_size_multiplier),
            max_size_multiplier=float(self.config.external_max_size_multiplier),
            max_positions_override=self.config.external_max_positions_override,
        )

    def _spree_open_r(self, positions: List[SpreePosition]) -> float:
        open_r = 0.0
        for pos in positions:
            mark = float(getattr(pos, "last_mark_price", pos.entry_price))
            sd = self._stop_distance(pos)
            if pos.direction == 1:
                open_r += (mark - float(pos.entry_price)) / sd
            else:
                open_r += (float(pos.entry_price) - mark) / sd
        return float(open_r)

    def _spree_force_close_reason(self, total_spree_r: float, peak_spree_r: float) -> str:
        if self.config.max_spree_loss_r is not None:
            if total_spree_r <= -abs(float(self.config.max_spree_loss_r)):
                return "spree_loss_cap"
        if (
            self.config.protect_spree_peak_after_r is not None
            and self.config.max_spree_giveback_r is not None
            and peak_spree_r >= float(self.config.protect_spree_peak_after_r)
            and (peak_spree_r - total_spree_r) >= float(self.config.max_spree_giveback_r)
        ):
            return "spree_profit_giveback"
        return ""

    def _can_add_spree_risk(
        self,
        realized_r: float,
        positions: List[SpreePosition],
        total_spree_r: Optional[float] = None,
        h1_atr_percentile: Optional[float] = None,
        ts: Optional[pd.Timestamp] = None,
    ) -> bool:
        if self.config.max_open_spree_risk_r is None:
            return True
        budget = self._open_spree_risk_budget(total_spree_r, h1_atr_percentile, ts)
        worst_after_next = float(realized_r) - float(len(positions)) - 1.0
        return worst_after_next >= -budget

    def _open_spree_risk_budget(
        self,
        total_spree_r: Optional[float] = None,
        h1_atr_percentile: Optional[float] = None,
        ts: Optional[pd.Timestamp] = None,
    ) -> float:
        budget = abs(float(self.config.max_open_spree_risk_r))
        relaxed_budget = self.config.relaxed_open_spree_risk_r
        relax_after = self.config.relax_open_risk_after_total_r
        if self._open_risk_schedule_active_at(ts):
            if self.config.scheduled_max_open_spree_risk_r is not None:
                budget = abs(float(self.config.scheduled_max_open_spree_risk_r))
            if self.config.scheduled_relaxed_open_spree_risk_r is not None:
                relaxed_budget = self.config.scheduled_relaxed_open_spree_risk_r
            if self.config.scheduled_relax_open_risk_after_total_r is not None:
                relax_after = self.config.scheduled_relax_open_risk_after_total_r
        if self._high_atr_risk_active(h1_atr_percentile):
            if self.config.high_atr_max_open_spree_risk_r is not None:
                budget = abs(float(self.config.high_atr_max_open_spree_risk_r))
            if self.config.high_atr_relaxed_open_spree_risk_r is not None:
                relaxed_budget = self.config.high_atr_relaxed_open_spree_risk_r
        if (
            relaxed_budget is not None
            and relax_after is not None
            and total_spree_r is not None
            and float(total_spree_r) >= float(relax_after)
        ):
            budget = max(budget, abs(float(relaxed_budget)))
        return float(budget)

    def _open_risk_schedule_active_at(self, ts: Optional[pd.Timestamp]) -> bool:
        if (
            self.config.open_risk_active_months is None
            and self.config.open_risk_active_month_numbers is None
            and self.config.open_risk_active_quarters is None
        ):
            return False
        if ts is None:
            return False
        stamp = pd.Timestamp(ts)
        if self.config.open_risk_active_months is not None:
            month = f"{stamp.year:04d}-{stamp.month:02d}"
            if month in self.config.open_risk_active_months:
                return True
        if self.config.open_risk_active_month_numbers is not None:
            if int(stamp.month) in {int(month) for month in self.config.open_risk_active_month_numbers}:
                return True
        if self.config.open_risk_active_quarters is not None:
            quarter = f"{stamp.year:04d}Q{((stamp.month - 1) // 3) + 1}"
            if quarter in self.config.open_risk_active_quarters:
                return True
        return False

    def _high_atr_risk_active(self, h1_atr_percentile: Optional[float]) -> bool:
        if self.config.high_atr_risk_percentile is None or h1_atr_percentile is None:
            return False
        return float(h1_atr_percentile) >= float(self.config.high_atr_risk_percentile)

    def _position_max_hold(
        self,
        base_max_hold: int,
        sequence: int,
        h1_atr_percentile: Optional[float],
        total_spree_r: float = 0.0,
        entry_ts: Optional[pd.Timestamp] = None,
    ) -> int:
        if (
            self.config.high_atr_hold_percentile is None
            or self.config.high_atr_max_hold is None
            or h1_atr_percentile is None
        ):
            return int(base_max_hold)
        if not self._high_atr_hold_active_at(entry_ts):
            return int(base_max_hold)
        if int(sequence) < int(self.config.high_atr_hold_min_sequence):
            return int(base_max_hold)
        if (
            self.config.high_atr_hold_after_total_r is not None
            and float(total_spree_r) < float(self.config.high_atr_hold_after_total_r)
        ):
            return int(base_max_hold)
        if float(h1_atr_percentile) < float(self.config.high_atr_hold_percentile):
            return int(base_max_hold)
        return max(1, min(int(base_max_hold), int(self.config.high_atr_max_hold)))

    def _high_atr_hold_active_at(self, ts: Optional[pd.Timestamp]) -> bool:
        if (
            self.config.high_atr_hold_active_months is None
            and self.config.high_atr_hold_active_month_numbers is None
            and self.config.high_atr_hold_active_quarters is None
        ):
            return True
        if ts is None:
            return False
        stamp = pd.Timestamp(ts)
        if self.config.high_atr_hold_active_months is not None:
            month = f"{stamp.year:04d}-{stamp.month:02d}"
            if month in self.config.high_atr_hold_active_months:
                return True
        if self.config.high_atr_hold_active_month_numbers is not None:
            if int(stamp.month) in {int(month) for month in self.config.high_atr_hold_active_month_numbers}:
                return True
        if self.config.high_atr_hold_active_quarters is not None:
            quarter = f"{stamp.year:04d}Q{((stamp.month - 1) // 3) + 1}"
            if quarter in self.config.high_atr_hold_active_quarters:
                return True
        return False

    @staticmethod
    def _h1_atr_percentile(h1: pd.DataFrame, h1_idx: int) -> float:
        if "_entry_h1_atr_percentile" not in h1.columns or h1_idx < 0 or h1_idx >= len(h1):
            return 0.5
        return float(h1["_entry_h1_atr_percentile"].iloc[h1_idx])

    def _can_add_spree_entry_count(
        self, closed_pnls: List[float], positions: List[SpreePosition]
    ) -> bool:
        if self.config.max_spree_entries is None:
            return True
        next_count = len(closed_pnls) + len(positions) + 1
        return next_count <= int(self.config.max_spree_entries)

    def _monthly_pause_active(self, result: VotingResult, ts: pd.Timestamp) -> bool:
        if self.config.monthly_loss_pause_r is None:
            return False
        month = str(pd.Timestamp(ts))[:7]
        closed_r = float(result.monthly_r.get(month, 0.0))
        return closed_r <= -abs(float(self.config.monthly_loss_pause_r))

    def _month_start_entry_skip_active(self, ts: pd.Timestamp) -> bool:
        days = int(self.config.month_start_entry_skip_days or 0)
        if days <= 0:
            return False
        stamp = pd.Timestamp(ts)
        if self.config.month_start_entry_skip_month_numbers is not None:
            active_months = {int(month) for month in self.config.month_start_entry_skip_month_numbers}
            if int(stamp.month) not in active_months:
                return False
        return int(stamp.day) <= days

    def _cooldown_until(self, current_i: int, spree_r: float) -> int:
        if self.config.loss_spree_cooldown_bars <= 0 or spree_r > 0:
            return -1
        return int(current_i) + int(self.config.loss_spree_cooldown_bars)

    @staticmethod
    def _stop_distance(pos: SpreePosition) -> float:
        return max(abs(float(pos.entry_price) - float(pos.sl_price)), 1e-9)

    def _signal_context_ready(self, h4: Optional[pd.DataFrame]) -> bool:
        return (not self.config.use_h4) or (h4 is not None and len(h4) >= 50)

    def _get_signal_vote(self, h1, m15, h4, h1_idx, m15_idx, regime):
        if h1_idx < 0 or m15_idx < self.engine.config.m15_momentum_bars:
            return 0, {}
        h4_vote = self.engine._h4_vote(h4) if self.config.use_h4 and h4 is not None else 0
        h1_vote = self.engine._h1_vote(h1, h1_idx)
        m15_vote = self.engine._m15_vote(m15, m15_idx) if self.config.use_m15 else 0
        vote_sum = h4_vote + h1_vote + m15_vote
        if abs(vote_sum) < 2:
            return 0, {}
        direction = 1 if vote_sum > 0 else -1
        if self.config.invert_signal:
            direction *= -1
        strength = "STRONG" if abs(vote_sum) >= 3 else "WEAK"
        exit_params = self.engine._regime_params(regime)
        exit_params["strength"] = strength
        exit_params["votes"] = {"h4": h4_vote, "h1": h1_vote, "m15": m15_vote, "sum": vote_sum}
        return direction, exit_params

    @staticmethod
    def _build_h4(h1: pd.DataFrame) -> pd.DataFrame:
        h4 = h1.set_index("timestamp").resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna().reset_index()
        h4["close_time"] = pd.to_datetime(h4["timestamp"]) + pd.Timedelta(hours=4)
        return h4

    @staticmethod
    def _h4_for_signal(h4: pd.DataFrame, signal_time: pd.Timestamp) -> Optional[pd.DataFrame]:
        mask = pd.to_datetime(h4["close_time"].values) <= pd.Timestamp(signal_time)
        if not bool(mask.any()):
            return None
        return h4[mask].reset_index(drop=True)

    def _record_entry(self, result: VotingResult, pos: SpreePosition, ts: pd.Timestamp, h1: pd.DataFrame, h1_idx: int, h4: pd.DataFrame, direction: int, price: float, strength: str, regime: str, ep: dict, m5: Optional[pd.DataFrame]) -> None:
        h4_last_close = pd.Timestamp(h4["close_time"].iloc[-1]) if h4 is not None and len(h4) else None
        position_id = int(getattr(pos, "position_id", -1))
        votes = ep.get("votes", {})
        opposite_first_r = None
        if int(getattr(pos, "sequence", -1)) == 0 and m5 is not None:
            opposite_first_r = self._simulate_opposite_first_r(pos, m5)
        h1_open = float(h1["open"].iloc[h1_idx]) if 0 <= h1_idx < len(h1) else 0.0
        h1_close = float(h1["close"].iloc[h1_idx]) if 0 <= h1_idx < len(h1) else 0.0
        quality_features = getattr(pos, "entry_quality_features", {}) or {}
        result.opened_ids.add(position_id)
        result.entries.append(
            EntryAudit(
                position_id=position_id,
                spree_id=int(getattr(pos, "spree_id", -1)),
                sequence=int(getattr(pos, "sequence", -1)),
                entry_time=pd.Timestamp(ts),
                signal_time=pd.Timestamp(ts),
                h1_idx=h1_idx,
                h4_last_close=h4_last_close,
                direction=direction,
                price=price,
                gated_allowed=self.gate.trading_allowed(ts),
                strength=strength,
                h1_open=h1_open,
                h1_close=h1_close,
                regime=regime,
                h4_vote=int(votes.get("h4", 0)),
                h1_vote=int(votes.get("h1", 0)),
                m15_vote=int(votes.get("m15", 0)),
                vote_sum=int(votes.get("sum", 0)),
                atr=float(pos.atr),
                sl_atr=float(getattr(pos, "sl_atr_mult", 0.0)),
                tp_atr=float(getattr(pos, "tp_atr_mult", 0.0)),
                max_hold=int(pos.max_hold),
                lots=float(getattr(pos, "lots", 0.0)),
                size_multiplier=float(getattr(pos, "size_multiplier", 0.0)),
                risk_scale=float(getattr(pos, "risk_scale", 0.0)),
                opposite_first_r=opposite_first_r,
                quality_score=float(getattr(pos, "entry_quality_score", 0.0)),
                quality_reason=str(getattr(pos, "entry_quality_reason", "")),
                expansion_reason=str(getattr(pos, "expansion_reason", "")),
                addon_reason=str(getattr(pos, "addon_reason", "")),
                addon_policy_mode=str(getattr(pos, "addon_policy_mode", "")),
                open_spree_r_at_add=float(getattr(pos, "open_spree_r_at_add", 0.0)),
                peak_spree_r_at_add=float(getattr(pos, "peak_spree_r_at_add", 0.0)),
                giveback_from_peak_r_at_add=float(getattr(pos, "giveback_from_peak_r_at_add", 0.0)),
                lead_mfe_r_at_add=float(getattr(pos, "lead_mfe_r_at_add", 0.0)),
                bars_since_last_add_at_add=float(getattr(pos, "bars_since_last_add_at_add", 0.0)),
                open_positions_at_add=float(getattr(pos, "open_positions_at_add", 0.0)),
                worst_case_spree_loss_r_at_add=float(getattr(pos, "worst_case_spree_loss_r_at_add", 0.0)),
                pre_entry_r=float(quality_features.get("pre_entry_r", 0.0)),
                h1_body_r=float(quality_features.get("h1_body_r", 0.0)),
                m15_alignment=float(quality_features.get("m15_alignment", 0.0)),
                m15_range_expansion=float(quality_features.get("m15_range_expansion", 0.0)),
                h1_atr_percentile=float(quality_features.get("h1_atr_percentile", 0.0)),
            )
        )

    def _record_blocked_entry(self, result: VotingResult, ts: pd.Timestamp, h1_atrs, h1_idx: int,
                              direction: int, price: float, strength: str, regime: str,
                              ep: dict, quality, m5: pd.DataFrame) -> None:
        votes = ep.get("votes", {})
        features = quality.features
        atr = float(h1_atrs[h1_idx]) if h1_idx < len(h1_atrs) else price * 0.01
        sl_m = float(self.config.sl_atr if self.config.sl_atr is not None else ep.get("sl_atr", 2.0))
        tp_m = float(self.config.tp_atr if self.config.tp_atr is not None else ep.get("tp_atr", 3.5))
        max_hold = int(self.config.max_hold if self.config.max_hold is not None else ep.get("max_hold", 24))
        result.blocked_entries.append(
            BlockedEntryAudit(
                entry_time=pd.Timestamp(ts),
                h1_idx=int(h1_idx),
                direction=int(direction),
                price=float(price),
                strength=str(strength),
                regime=str(regime),
                quality_score=float(quality.quality_score),
                block_reason=str(quality.block_reason),
                h1_vote=int(votes.get("h1", 0)),
                m15_vote=int(votes.get("m15", 0)),
                vote_sum=int(votes.get("sum", 0)),
                pre_entry_r=float(features.get("pre_entry_r", 0.0)),
                h1_body_r=float(features.get("h1_body_r", 0.0)),
                m15_alignment=float(features.get("m15_alignment", 0.0)),
                m15_range_expansion=float(features.get("m15_range_expansion", 0.0)),
                h1_atr_percentile=float(features.get("h1_atr_percentile", 0.0)),
                atr=float(atr),
                sl_atr=float(sl_m),
                tp_atr=float(tp_m),
                max_hold=int(max_hold),
                same_direction_first_r=(
                    self._simulate_first_r(direction, ts, price, atr, sl_m, tp_m, max_hold, m5)
                    if self.config.record_blocked_entry_counterfactuals
                    else None
                ),
                opposite_first_r=(
                    self._simulate_first_r(-direction, ts, price, atr, sl_m, tp_m, max_hold, m5)
                    if self.config.record_blocked_entry_counterfactuals
                    else None
                ),
            )
        )

    def _simulate_opposite_first_r(self, pos: SpreePosition, m5: pd.DataFrame) -> float:
        return self._simulate_first_r(
            -int(pos.direction),
            pd.Timestamp(pos.entry_time),
            float(getattr(pos, "mid_entry_price", pos.entry_price)),
            float(pos.atr),
            float(getattr(pos, "sl_atr_mult", 2.0)),
            float(getattr(pos, "tp_atr_mult", 3.5)),
            int(pos.max_hold),
            m5,
        )

    def _simulate_first_r(self, direction: int, entry_time: pd.Timestamp, entry_price: float,
                          atr: float, sl_atr: float, tp_atr: float, max_hold: int,
                          m5: pd.DataFrame) -> float:
        entry_time = pd.Timestamp(entry_time)
        entry_fill = self.execution.entry_fill(entry_price, direction, 1.0)
        fill_entry = float(entry_fill.fill_price)
        sl = entry_price - direction * atr * sl_atr
        tp = entry_price + direction * atr * tp_atr
        stop_distance = max(abs(fill_entry - sl), 1e-9)
        end_time = entry_time + pd.Timedelta(minutes=15 * max_hold)
        m5_close = self._m5_close_values(m5)
        start_idx = int(np.searchsorted(m5_close, np.datetime64(entry_time), side="right"))
        end_idx = int(np.searchsorted(m5_close, np.datetime64(end_time), side="right"))
        last_close = entry_price
        entry_fee = float(entry_fill.fee)
        denom = max(stop_distance, 1e-9)

        def net_r(mid_exit: float, reason: str) -> float:
            exit_fill = self.execution.exit_fill(float(mid_exit), direction, 1.0, reason)
            if direction == 1:
                pnl = float(exit_fill.fill_price) - fill_entry
            else:
                pnl = fill_entry - float(exit_fill.fill_price)
            pnl -= entry_fee + float(exit_fill.fee)
            return float(pnl / denom)

        for j in range(start_idx, end_idx):
            hi = float(m5["high"].iloc[j])
            lo = float(m5["low"].iloc[j])
            last_close = float(m5["close"].iloc[j]) if "close" in m5.columns else last_close
            if direction == 1:
                if lo <= sl:
                    return net_r(self._sim_stop_fill_mid(direction, sl, m5, j), "sl_hit")
                if hi >= tp:
                    return net_r(tp, "tp_hit")
            else:
                if hi >= sl:
                    return net_r(self._sim_stop_fill_mid(direction, sl, m5, j), "sl_hit")
                if lo <= tp:
                    return net_r(tp, "tp_hit")
        return net_r(last_close, "time_exit")

    @staticmethod
    def _sim_stop_fill_mid(direction: int, stop: float, m5: pd.DataFrame, idx: int) -> float:
        if "open" not in m5.columns:
            return float(stop)
        bar_open = float(m5["open"].iloc[idx])
        if int(direction) == 1:
            return min(float(stop), bar_open)
        return max(float(stop), bar_open)

    @staticmethod
    def _m5_close_values(m5: pd.DataFrame) -> np.ndarray:
        cached = m5.attrs.get("_close_time_values")
        if cached is None:
            cached = np.asarray(pd.to_datetime(m5["close_time"].values), dtype="datetime64[ns]")
            m5.attrs["_close_time_values"] = cached
        return cached

    @staticmethod
    def _record_monthly_close(result: VotingResult, pos: SpreePosition) -> None:
        key = (str(pos.entry_time), str(getattr(pos, "close_time", "")), pos.direction, round(pos.entry_price, 8))
        if key in result.monthly_closed_seen:
            return
        result.monthly_closed_seen.add(key)
        month = str(pd.Timestamp(pos.close_time))[:7]
        result.monthly_pnl[month] = result.monthly_pnl.get(month, 0.0) + pos.pnl_dollar
        result.monthly_r[month] = result.monthly_r.get(month, 0.0) + pos.pnl_r
        result.monthly_trades[month] = result.monthly_trades.get(month, 0) + 1

    def _reconcile_spree_stats(self, result: VotingResult) -> None:
        """Rebuild spree-level stats from closed trades after all lifecycle exits."""
        by_spree: Dict[int, List[ClosedTrade]] = {}
        for trade in result.trades:
            by_spree.setdefault(int(trade.spree_id), []).append(trade)

        result.stats = []
        result.monthly_sprees = {}
        forced_reasons = {
            "first_trade_loss",
            "spree_kill",
            "max_spree_loss",
            "spree_giveback",
        }

        for spree_id in sorted(by_spree):
            trades = sorted(by_spree[spree_id], key=lambda item: (item.entry_time, item.exit_time, item.position_id))
            if not trades:
                continue
            pnl_r = float(sum(trade.pnl_r for trade in trades))
            first_trade = trades[0]
            last_trade = max(trades, key=lambda item: item.exit_time)
            duration_hours = (pd.Timestamp(last_trade.exit_time) - pd.Timestamp(first_trade.entry_time)).total_seconds() / 3600.0
            killed = any(trade.reason in forced_reasons for trade in trades)
            if self.config.first_trade_gate and len(trades) == 1 and float(first_trade.pnl_r) <= 0:
                killed = True
            result.stats.append((pnl_r, len(trades), killed, duration_hours, int(first_trade.direction)))
            month = str(pd.Timestamp(first_trade.entry_time))[:7]
            result.monthly_sprees[month] = result.monthly_sprees.get(month, 0) + 1

    @staticmethod
    def _normalise_times(h1: pd.DataFrame, m15: pd.DataFrame, m5: pd.DataFrame) -> None:
        for frame, minutes in ((h1, 60), (m15, 15), (m5, 5)):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
            if "close_time" in frame.columns:
                frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True).dt.tz_localize(None)
            else:
                frame["close_time"] = frame["timestamp"] + pd.Timedelta(minutes=minutes)

    def _apply_shift(self, h1: pd.DataFrame, m15: pd.DataFrame) -> None:
        if self.shift == 0:
            return
        for col in ["open", "high", "low", "close", "volume"]:
            if col in h1.columns:
                h1[col] = h1[col].shift(self.shift)
            if col in m15.columns:
                m15[col] = m15[col].shift(self.shift)
        h1.dropna(subset=["open", "high", "low", "close"], inplace=True)
        m15.dropna(subset=["open", "high", "low", "close"], inplace=True)
        h1.reset_index(drop=True, inplace=True)
        m15.reset_index(drop=True, inplace=True)

    def _audit_result(self, result: VotingResult) -> None:
        entry_keys = set()
        for entry in result.entries:
            if entry.h4_last_close is not None and entry.h4_last_close > entry.signal_time:
                result.issues.append(f"H4_LOOKAHEAD:{entry.h4_last_close}>{entry.signal_time}")
            if not entry.gated_allowed:
                result.issues.append(f"EVENT_GATE_ENTRY:{entry.entry_time}")
            key = (str(entry.entry_time)[:16], entry.direction, round(entry.price, 2))
            if key in entry_keys:
                result.issues.append(f"DUPLICATE_ENTRY:{key}")
            entry_keys.add(key)
        for trade in result.trades:
            if trade.exit_time <= trade.entry_time:
                result.issues.append(f"EXIT_NOT_AFTER_ENTRY:{trade.entry_time}->{trade.exit_time}")
            if abs(trade.stop_distance - abs(trade.entry_price - (trade.entry_price - trade.direction * trade.stop_distance))) > 1e-6:
                result.issues.append(f"BAD_STOP_DISTANCE:{trade.entry_time}")
        orphan_opens = sorted(result.opened_ids - result.closed_ids)
        orphan_closes = sorted(result.closed_ids - result.opened_ids)
        if orphan_opens:
            result.issues.append(f"ORPHAN_OPEN_IDS:{orphan_opens[:10]}")
        if orphan_closes:
            result.issues.append(f"ORPHAN_CLOSE_IDS:{orphan_closes[:10]}")
        if len(result.opened_ids) != len(result.entries):
            result.issues.append(f"OPEN_ID_COUNT_MISMATCH:{len(result.opened_ids)}!={len(result.entries)}")
        if len(result.closed_ids) != len(result.trades):
            result.issues.append(f"CLOSE_ID_COUNT_MISMATCH:{len(result.closed_ids)}!={len(result.trades)}")

def run_year(start: str, end: str, shift: int = 0,
             config: Optional[VotingBacktestConfig] = None,
             loader: Optional[V3DataLoader] = None,
             regime_loader: Optional[RegimeStateLoader] = None,
             data_cache: Optional[dict] = None,
             state_cache: Optional[dict] = None,
             trade_start: Optional[str] = None,
             initial_balance: float = 10000.0) -> VotingResult:
    return VotingBacktest(
        initial_balance=initial_balance,
        shift=shift,
        config=config,
        loader=loader,
        regime_loader=regime_loader,
        data_cache=data_cache,
        state_cache=state_cache,
    ).run(start, end, trade_start=trade_start)
