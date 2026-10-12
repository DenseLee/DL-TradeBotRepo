"""Minimal MT5 execution adapter used by ``bot_manager.py``.

The strategy code owns signal generation and risk decisions.  This module only
wraps the MetaTrader5 package behind a small, testable executor interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd


@dataclass
class ExecutionResult:
    success: bool
    ticket: Optional[int] = None
    price: float = 0.0
    error: str = ""


class DryRunExecutor:
    def __init__(self, symbol: str = "BTCUSD", initial_balance: float = 10000.0):
        self.symbol = symbol
        self.balance = float(initial_balance)
        self.price = 0.0
        self.positions: dict[int, dict[str, Any]] = {}
        self.next_ticket = 1
        self.volume_min = 0.001
        self.volume_step = 0.001
        self.volume_max = 100.0
        self.mt5 = None

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    def reconnect(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True

    def set_price(self, price: float) -> None:
        self.price = float(price)

    def get_current_price(self) -> Optional[float]:
        return self.price or None

    def get_account_balance(self) -> float:
        return self.balance

    def get_positions(self, symbol: Optional[str] = None):
        from collections import namedtuple

        Position = namedtuple("Position", "ticket symbol type volume price_open sl tp")
        rows = []
        for ticket, pos in self.positions.items():
            if symbol is not None and pos["symbol"] != symbol:
                continue
            rows.append(
                Position(
                    ticket=ticket,
                    symbol=pos["symbol"],
                    type=0 if pos["direction"] == 1 else 1,
                    volume=pos["lots"],
                    price_open=pos["price"],
                    sl=pos["sl"],
                    tp=pos["tp"],
                )
            )
        return rows

    def open_position(self, direction: int, lots: float, sl: float, tp: float, comment: str = "") -> ExecutionResult:
        price = float(self.price or 0.0)
        if price <= 0:
            return ExecutionResult(False, error="dry-run price is not set")
        ticket = self.next_ticket
        self.next_ticket += 1
        self.positions[ticket] = {
            "symbol": self.symbol,
            "direction": int(direction),
            "lots": float(lots),
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "comment": comment,
        }
        return ExecutionResult(True, ticket=ticket, price=price)

    def close_position(self, ticket: int) -> ExecutionResult:
        pos = self.positions.pop(int(ticket), None)
        if pos is None:
            return ExecutionResult(False, error=f"ticket {ticket} not found")
        price = float(self.price or pos["price"])
        pnl = (price - pos["price"]) * pos["lots"] if pos["direction"] == 1 else (pos["price"] - price) * pos["lots"]
        self.balance += float(pnl)
        return ExecutionResult(True, ticket=int(ticket), price=price)

    def modify_position_stops(self, ticket: int, sl: float, tp: float) -> ExecutionResult:
        pos = self.positions.get(int(ticket))
        if pos is None:
            return ExecutionResult(False, ticket=int(ticket), error=f"ticket {ticket} not found")
        pos["sl"] = float(sl)
        pos["tp"] = float(tp)
        return ExecutionResult(True, ticket=int(ticket), price=float(self.price or pos["price"]))

    def get_bars(self, timeframe: str, count: int = 500):
        raise RuntimeError("DryRunExecutor does not provide live bars")


class MT5Executor:
    def __init__(self, symbol: str = "BTCUSD", magic: int = 20260517, deviation: int = 20, terminal_path: str | None = None):
        self.symbol = symbol
        self.magic = int(magic)
        self.deviation = int(deviation)
        self.terminal_path = terminal_path
        self.mt5 = None
        self.volume_min = 0.0
        self.volume_step = 0.0
        self.volume_max = 0.0
        self.filling_mode = None

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
        except Exception as exc:  # pragma: no cover - depends on local MT5 install
            raise RuntimeError(f"MetaTrader5 package is unavailable: {exc}") from exc

        self.mt5 = mt5
        ok = mt5.initialize(path=self.terminal_path) if self.terminal_path else mt5.initialize()
        if not ok:
            return False
        # The public archive is deliberately unable to connect to a real-money
        # account. Keep this guard in the shared adapter so every archived
        # runner inherits it before any quote or order operation.
        account = mt5.account_info()
        if account is None or int(getattr(account, "trade_mode", -1)) != 0:
            mt5.shutdown()
            raise RuntimeError("This archive is demo-account-only; real-money accounts are blocked.")
        mt5.symbol_select(self.symbol, True)
        self._load_symbol_metadata()
        return True

    def disconnect(self) -> None:
        if self.mt5 is not None:
            self.mt5.shutdown()

    def reconnect(self) -> bool:
        self.disconnect()
        return self.connect()

    def health_check(self) -> bool:
        if self.mt5 is None:
            return False
        try:
            return self.mt5.terminal_info() is not None and self.mt5.account_info() is not None
        except Exception:
            return False

    def _load_symbol_metadata(self) -> None:
        info = self.mt5.symbol_info(self.symbol) if self.mt5 is not None else None
        self.volume_min = float(getattr(info, "volume_min", 0.0) or 0.0)
        self.volume_step = float(getattr(info, "volume_step", 0.0) or 0.0)
        self.volume_max = float(getattr(info, "volume_max", 0.0) or 0.0)
        self.filling_mode = getattr(info, "filling_mode", None)

    def get_account_balance(self) -> Optional[float]:
        info = self.mt5.account_info() if self.mt5 is not None else None
        return float(info.balance) if info is not None else None

    def get_current_price(self) -> Optional[float]:
        tick = self.mt5.symbol_info_tick(self.symbol) if self.mt5 is not None else None
        if tick is None or not tick.bid or not tick.ask:
            return None
        return (float(tick.bid) + float(tick.ask)) / 2.0

    def get_positions(self, symbol: Optional[str] = None):
        if self.mt5 is None:
            return []
        positions = self.mt5.positions_get(symbol=symbol or self.symbol)
        return list(positions or [])

    def get_bars(self, timeframe: str, count: int = 500) -> pd.DataFrame:
        if self.mt5 is None:
            raise RuntimeError("MT5 is not connected")
        tf_map = {
            "M1": self.mt5.TIMEFRAME_M1,
            "M5": self.mt5.TIMEFRAME_M5,
            "M15": self.mt5.TIMEFRAME_M15,
            "H1": self.mt5.TIMEFRAME_H1,
            "H4": self.mt5.TIMEFRAME_H4,
        }
        tf = tf_map.get(str(timeframe).upper())
        if tf is None:
            raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
        rates = self.mt5.copy_rates_from_pos(self.symbol, tf, 0, int(count))
        if rates is None:
            raise RuntimeError(f"MT5 copy_rates_from_pos failed: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=False)
        volume_col = "tick_volume" if "tick_volume" in df.columns else "real_volume"
        return df.rename(columns={volume_col: "volume"})[["timestamp", "open", "high", "low", "close", "volume"]]

    def open_position(self, direction: int, lots: float, sl: float, tp: float, comment: str = "") -> ExecutionResult:
        if self.mt5 is None:
            return ExecutionResult(False, error="MT5 is not connected")
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return ExecutionResult(False, error="No symbol tick available")

        is_long = int(direction) == 1
        order_type = self.mt5.ORDER_TYPE_BUY if is_long else self.mt5.ORDER_TYPE_SELL
        price = float(tick.ask if is_long else tick.bid)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": str(comment)[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
        }
        result = self._send_order_with_filling_retry(request)
        if result is None:
            return ExecutionResult(False, error=f"order_send returned None: {self.mt5.last_error()}")
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            return ExecutionResult(False, error=f"retcode={result.retcode} comment={getattr(result, 'comment', '')}")
        ticket = int(getattr(result, "order", 0) or getattr(result, "deal", 0) or 0)
        return ExecutionResult(True, ticket=ticket, price=float(getattr(result, "price", price) or price))

    def close_position(self, ticket: int) -> ExecutionResult:
        if self.mt5 is None:
            return ExecutionResult(False, error="MT5 is not connected")
        positions = self.mt5.positions_get(ticket=int(ticket))
        if not positions:
            return ExecutionResult(False, ticket=int(ticket), error=f"position {ticket} not found")
        pos = positions[0]
        tick = self.mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return ExecutionResult(False, ticket=int(ticket), error="No symbol tick available")

        is_buy_pos = int(pos.type) == self.mt5.POSITION_TYPE_BUY
        close_type = self.mt5.ORDER_TYPE_SELL if is_buy_pos else self.mt5.ORDER_TYPE_BUY
        price = float(tick.bid if is_buy_pos else tick.ask)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "position": int(ticket),
            "volume": float(pos.volume),
            "type": close_type,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "bot_close",
            "type_time": self.mt5.ORDER_TIME_GTC,
        }
        result = self._send_order_with_filling_retry(request, symbol=pos.symbol)
        if result is None:
            return ExecutionResult(False, ticket=int(ticket), error=f"order_send returned None: {self.mt5.last_error()}")
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            return ExecutionResult(False, ticket=int(ticket), error=f"retcode={result.retcode} comment={getattr(result, 'comment', '')}")
        return ExecutionResult(True, ticket=int(ticket), price=float(getattr(result, "price", price) or price))

    def modify_position_stops(self, ticket: int, sl: float, tp: float) -> ExecutionResult:
        """Modify only the protective levels of an existing MT5 position."""
        if self.mt5 is None:
            return ExecutionResult(False, ticket=int(ticket), error="MT5 is not connected")
        positions = self.mt5.positions_get(ticket=int(ticket))
        if not positions:
            return ExecutionResult(False, ticket=int(ticket), error=f"position {ticket} not found")
        pos = positions[0]
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": int(ticket),
            "sl": float(sl),
            "tp": float(tp),
            "magic": self.magic,
        }
        result = self.mt5.order_send(request)
        if result is None:
            return ExecutionResult(False, ticket=int(ticket), error=f"order_send returned None: {self.mt5.last_error()}")
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            return ExecutionResult(False, ticket=int(ticket), error=f"retcode={result.retcode} comment={getattr(result, 'comment', '')}")
        tick = self.mt5.symbol_info_tick(pos.symbol)
        price = 0.0 if tick is None else float((tick.bid + tick.ask) / 2.0)
        return ExecutionResult(True, ticket=int(ticket), price=price)

    def _filling_candidates(self, symbol: Optional[str] = None) -> list[int]:
        info = self.mt5.symbol_info(symbol or self.symbol) if self.mt5 is not None else None
        preferred = getattr(info, "filling_mode", None) if info is not None else self.filling_mode
        candidates: list[int] = []
        for value in [
            preferred,
            getattr(self.mt5, "ORDER_FILLING_FOK", None),
            getattr(self.mt5, "ORDER_FILLING_IOC", None),
            getattr(self.mt5, "ORDER_FILLING_RETURN", None),
        ]:
            if value is None:
                continue
            ivalue = int(value)
            if ivalue not in candidates:
                candidates.append(ivalue)
        return candidates

    def _send_order_with_filling_retry(self, request: dict[str, Any], symbol: Optional[str] = None):
        last_result = None
        for filling in self._filling_candidates(symbol):
            trial = dict(request)
            trial["type_filling"] = int(filling)
            result = self.mt5.order_send(trial)
            if result is None:
                return None
            last_result = result
            if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                return result
            # 10030 is TRADE_RETCODE_INVALID_FILL on MT5 builds that expose it.
            invalid_fill = getattr(self.mt5, "TRADE_RETCODE_INVALID_FILL", 10030)
            if int(result.retcode) != int(invalid_fill):
                return result
        return last_result
