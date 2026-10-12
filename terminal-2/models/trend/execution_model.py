"""Live-like execution costs for leak-safe voting backtests."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionCostConfig:
    profile: str = "zero"
    spread_bps: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    stop_slippage_bps: float = 0.0
    tp_slippage_bps: float = 0.0
    commission_bps: float = 0.0
    fixed_fee_usd: float = 0.0
    max_allowed_spread_bps: float | None = None

    @classmethod
    def from_profile(cls, profile: str = "zero", **overrides) -> "ExecutionCostConfig":
        name = str(profile or "zero").lower()
        if name == "zero":
            config = cls(profile="zero")
        elif name == "conservative_btc":
            config = cls(
                profile="conservative_btc",
                spread_bps=3.0,
                entry_slippage_bps=1.0,
                exit_slippage_bps=1.0,
                stop_slippage_bps=2.0,
                tp_slippage_bps=1.0,
                commission_bps=2.0,
                fixed_fee_usd=0.0,
                max_allowed_spread_bps=12.0,
            )
        elif name == "live_observed":
            config = cls(profile="live_observed")
        else:
            raise ValueError(f"Unknown execution profile: {profile}")
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config


@dataclass
class ExecutionFill:
    mid_price: float
    fill_price: float
    spread_cost: float
    slippage_cost: float
    fee: float


class ExecutionModel:
    """Adverse bid/ask, slippage, and fee model using only decision-time inputs."""

    def __init__(self, config: ExecutionCostConfig | None = None):
        self.config = config or ExecutionCostConfig()

    def spread_allowed(self) -> bool:
        cap = self.config.max_allowed_spread_bps
        return cap is None or float(self.config.spread_bps) <= float(cap)

    def entry_fill(self, mid_price: float, direction: int, lots: float) -> ExecutionFill:
        return self._fill(mid_price, direction, lots, self.config.entry_slippage_bps)

    def exit_fill(self, mid_price: float, direction: int, lots: float, reason: str) -> ExecutionFill:
        if reason == "sl_hit":
            slippage_bps = self.config.stop_slippage_bps
        elif reason == "tp_hit":
            slippage_bps = self.config.tp_slippage_bps
        else:
            slippage_bps = self.config.exit_slippage_bps
        return self._fill(mid_price, -int(direction), lots, slippage_bps)

    def _fill(self, mid_price: float, trade_side: int, lots: float, slippage_bps: float) -> ExecutionFill:
        mid = float(mid_price)
        side = 1 if int(trade_side) >= 0 else -1
        volume = max(float(lots), 0.0)
        half_spread = abs(float(self.config.spread_bps)) / 20000.0 * mid
        slippage = abs(float(slippage_bps)) / 10000.0 * mid
        spread_cost = half_spread * volume
        slippage_cost = slippage * volume
        fill_price = mid + side * (half_spread + slippage)
        fee = abs(fill_price * volume) * abs(float(self.config.commission_bps)) / 10000.0
        if volume > 0:
            fee += abs(float(self.config.fixed_fee_usd))
        return ExecutionFill(
            mid_price=mid,
            fill_price=float(fill_price),
            spread_cost=float(spread_cost),
            slippage_cost=float(slippage_cost),
            fee=float(fee),
        )
