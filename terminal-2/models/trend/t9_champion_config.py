"""Frozen T9/no-H4 champion configuration for reproducible research runs."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from models.trend.entry_quality_filter import EntryQualityConfig
from models.trend.spree_addon_policy import SpreeAddonConfig
from models.trend.voting_backtest import VotingBacktestConfig


def make_t9_champion_config(
    cost_bps: Optional[float] = None,
    execution_profile: str = "zero",
) -> VotingBacktestConfig:
    """Return the post-fix no-H4 champion config used as the research baseline."""
    return VotingBacktestConfig(
        use_h4=False,
        max_spree_loss_r=4.0,
        max_open_spree_risk_r=6.0,
        relaxed_open_spree_risk_r=8.0,
        relax_open_risk_after_total_r=3.0,
        risk_drawdown_start_pct=1.5,
        risk_drawdown_full_pct=5.0,
        risk_drawdown_floor_scale=0.40,
        addon_drawdown_start_pct=0.0,
        addon_drawdown_full_pct=4.0,
        addon_drawdown_floor_scale=0.50,
        high_atr_hold_percentile=0.95,
        high_atr_max_hold=16,
        high_atr_hold_min_sequence=1,
        high_atr_hold_active_month_numbers={3, 4, 10, 12},
        broker_volume_min=0.001,
        broker_volume_step=0.001,
        broker_volume_rounding_mode="floor_addons_to_min",
        execution_profile=execution_profile,
        spread_bps=cost_bps,
        entry_quality=EntryQualityConfig(
            enabled=True,
            max_pre_entry_r=1.0,
            max_h1_body_r=None,
            require_m15_alignment=False,
            min_m15_range_expansion=None,
            block_score_below=-999.0,
            min_entry_directional_momentum_r=0.25,
            min_entry_m15_range_expansion=1.0,
        ),
        addon_policy=SpreeAddonConfig(
            enabled=True,
            mode="m15_confirmed",
            min_total_r=0.35,
            min_m15_range_expansion=1.0,
            min_directional_momentum_r=0.0,
            base_size_multiplier=0.5,
            strong_size_multiplier=1.0,
            scale_state_multiplier=0.05,
            scale_open_r_min=3.0,
            scale_open_r_max=5.0,
            scale_peak_r_min=3.0,
            scale_peak_r_max=5.0,
            prove_first_min_total_r=1.75,
            prove_first_relax_after_total_r=3.0,
            prove_first_relaxed_min_total_r=0.875,
            relax_addon_gates_after_total_r=1.5,
        ),
    )


def config_to_jsonable(value: Any) -> Any:
    """Convert nested dataclasses/sets into stable JSON-compatible values."""
    if is_dataclass(value):
        return {key: config_to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, set):
        return sorted(config_to_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [config_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): config_to_jsonable(val) for key, val in value.items()}
    return value
