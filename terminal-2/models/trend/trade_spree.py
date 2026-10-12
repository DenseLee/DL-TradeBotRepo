"""
Trade Spree — pyramid entries during confirmed trend moves.

When the trend engine signals a direction and early trades succeed,
aggressively add positions on every qualifying bar. Track a sliding
window of recent trade outcomes. If M out of N recent trades go negative,
close the entire spree.

Per the institutional report [§37]: kill switches at strategy level.

Usage:
  from models.trend.trade_spree import TradeSpree
  spree = TradeSpree(max_positions=10, window=5, kill_threshold=3)

  # On each bar:
  spree.update(positions, current_prices)
  if spree.should_kill():
      close_all(spree.positions)
  elif spree.can_add():
      enter_new_position(direction, lots)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SpreePosition:
    """A single position within a trade spree."""
    entry_time: str
    entry_price: float
    direction: int  # 1=LONG, -1=SHORT
    lots: float
    atr: float
    sl_price: float
    tp_price: float
    max_hold: int
    bars_held: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    closed: bool = False
    close_price: float = 0.0
    close_reason: str = ""
    pnl_r: float = 0.0
    pnl_dollar: float = 0.0
    initial_stop_distance: float = 0.0
    initial_sl_price: float = 0.0
    sequence: int = 0
    seed_protected: bool = False


@dataclass
class TradeSpree:
    """Manages a pyramid of positions during a confirmed trend move.

    Once the first 2 trades are profitable, enters "active spree" mode
    where every qualifying bar adds another position. Tracks a sliding
    window of recent outcomes. Kills the spree when the trend fades.
    """

    # ── Configuration ──
    max_positions: int = 10          # Max concurrent positions in a spree
    window: int = 5                   # Sliding window for kill decision
    kill_threshold: int = 3           # M negative trades in window → kill
    warmup_trades: int = 2            # Need N profitable trades before pyramiding

    # ── State ──
    positions: List[SpreePosition] = field(default_factory=list)
    closed_pnls: List[float] = field(default_factory=list)  # recent outcomes
    direction: int = 0                # 0 = no active spree
    active: bool = False              # True once warmup threshold met
    total_pnl_r: float = 0.0
    kill_reason: str = ""

    def start(self, position: SpreePosition) -> None:
        """Begin a new spree with the first position."""
        self.positions = [position]
        self.closed_pnls = []
        self.direction = position.direction
        self.active = False
        self.total_pnl_r = 0.0
        self.kill_reason = ""

    def add(self, position: SpreePosition) -> None:
        """Add another position to the spree."""
        if len(self.positions) >= self.max_positions:
            return
        if position.direction != self.direction:
            return
        self.positions.append(position)

    def update_positions(
        self, current_high: float, current_low: float, current_price: float
    ) -> List[SpreePosition]:
        """Check SL/TP/time for all open positions. Returns newly closed positions."""
        newly_closed = []
        for pos in self.positions:
            if pos.closed:
                continue
            pos.bars_held += 1

            # Check SL/TP (simplified — uses bar high/low)
            if pos.direction == 1:  # LONG
                if current_low <= pos.sl_price:
                    pos.closed = True
                    pos.close_price = pos.sl_price
                    pos.close_reason = "sl_hit"
                elif current_high >= pos.tp_price:
                    pos.closed = True
                    pos.close_price = pos.tp_price
                    pos.close_reason = "tp_hit"
            else:  # SHORT
                if current_high >= pos.sl_price:
                    pos.closed = True
                    pos.close_price = pos.sl_price
                    pos.close_reason = "sl_hit"
                elif current_low <= pos.tp_price:
                    pos.closed = True
                    pos.close_price = pos.tp_price
                    pos.close_reason = "tp_hit"

            # Time stop
            if not pos.closed and pos.bars_held >= pos.max_hold:
                pos.closed = True
                pos.close_price = current_price
                pos.close_reason = "time_exit"

            # Update MFE/MAE
            sd = max(abs(pos.entry_price - pos.sl_price), 1e-9)
            if pos.direction == 1:
                pos.mfe_r = max(pos.mfe_r, (current_high - pos.entry_price) / max(sd, 1e-9))
                pos.mae_r = min(pos.mae_r, (current_low - pos.entry_price) / max(sd, 1e-9))
            else:
                pos.mfe_r = max(pos.mfe_r, (pos.entry_price - current_low) / max(sd, 1e-9))
                pos.mae_r = min(pos.mae_r, (pos.entry_price - current_high) / max(sd, 1e-9))

            if pos.closed:
                # Compute PnL
                if pos.direction == 1:
                    pos.pnl_dollar = (pos.close_price - pos.entry_price) * pos.lots
                else:
                    pos.pnl_dollar = (pos.entry_price - pos.close_price) * pos.lots
                pos.pnl_r = pos.pnl_dollar / max(sd * pos.lots, 1e-9)

                self.closed_pnls.append(pos.pnl_r)
                self.total_pnl_r += pos.pnl_r
                newly_closed.append(pos)

                # Trim closed_pnls to window
                if len(self.closed_pnls) > self.window * 3:
                    self.closed_pnls = self.closed_pnls[-self.window * 2:]

                # Check if warmup is achieved
                if not self.active and len(self.closed_pnls) >= self.warmup_trades:
                    recent = self.closed_pnls[-self.warmup_trades:]
                    if all(r > 0 for r in recent):
                        self.active = True

        # Remove closed positions
        self.positions = [p for p in self.positions if not p.closed]
        return newly_closed

    def should_kill(self) -> bool:
        """Check if the spree should be killed based on recent outcomes."""
        if len(self.closed_pnls) < self.window:
            return False

        recent = self.closed_pnls[-self.window:]
        negative_count = sum(1 for r in recent if r <= 0)

        if negative_count >= self.kill_threshold:
            self.kill_reason = (
                f"kill: {negative_count}/{self.window} negative in recent window"
            )
            return True

        return False

    def can_add(self) -> bool:
        """Check if we can add another position (spree active + under max)."""
        open_count = sum(1 for p in self.positions if not p.closed)
        return self.active and open_count < self.max_positions

    def open_count(self) -> int:
        """Number of currently open positions."""
        return sum(1 for p in self.positions if not p.closed)

    def close_all(self, current_price: float, reason: str = "spree_kill"):
        """Force-close all remaining open positions."""
        for pos in self.positions:
            if pos.closed:
                continue
            pos.closed = True
            pos.close_price = current_price
            pos.close_reason = reason

            sd = max(abs(pos.entry_price - pos.sl_price), 1e-9)
            if pos.direction == 1:
                pos.pnl_dollar = (current_price - pos.entry_price) * pos.lots
            else:
                pos.pnl_dollar = (pos.entry_price - current_price) * pos.lots
            pos.pnl_r = pos.pnl_dollar / max(sd * pos.lots, 1e-9)

            self.closed_pnls.append(pos.pnl_r)
            self.total_pnl_r += pos.pnl_r

        self.positions = [p for p in self.positions if not p.closed]
        self.active = False
        self.direction = 0

    def summary(self) -> dict:
        """Return summary stats for this spree."""
        all_positions = self.positions + [
            p for p in self.positions if p.closed
        ]
        # Need to also include already-removed positions
        total_positions = len(self.closed_pnls)
        wins = sum(1 for r in self.closed_pnls if r > 0)
        return {
            "total_positions": total_positions,
            "wins": wins,
            "win_rate": wins / max(total_positions, 1) * 100,
            "total_pnl_r": self.total_pnl_r,
            "avg_pnl_r": self.total_pnl_r / max(total_positions, 1),
            "kill_reason": self.kill_reason,
        }


# ── Quick smoke test ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a short spree
    spree = TradeSpree(warmup_trades=2, window=5, kill_threshold=3)

    # Add first winning trade
    p1 = SpreePosition("01", 100.0, -1, 0.01, 2.0, 105.0, 95.0, 24)
    spree.start(p1)

    # Simulate bars
    prices = [101, 102, 99, 98, 97, 96, 95, 96, 97, 100, 102, 103, 104, 103]

    for i, price in enumerate(prices):
        hi = price + 0.5
        lo = price - 0.5

        # Add new position if active
        if spree.can_add():
            p = SpreePosition(f"bar{i}", price, -1, 0.01, 2.0, price + 5, price - 5, 24)
            spree.add(p)
            print(f"  Bar {i}: ADDED SHORT @ {price} (total open: {spree.open_count()})")

        # Update positions
        closed = spree.update_positions(hi, lo, price)
        for c in closed:
            print(f"  Bar {i}: CLOSED {c.direction} @ {c.close_price} → {c.pnl_r:+.2f}R ({c.close_reason})")

        # Check kill
        if spree.should_kill():
            spree.close_all(price)
            print(f"  Bar {i}: SPREE KILLED — {spree.kill_reason}")
            break

    print(f"\nSpree summary: {spree.summary()}")
