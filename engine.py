"""
Limit Order Book (LOB) Matching Engine
Price-time priority, partial fills, market and limit orders.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import itertools
import json
import time


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()


class OrderStatus(Enum):
    OPEN = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()


_id_counter = itertools.count(1)


@dataclass
class Order:
    side: Side
    order_type: OrderType
    qty: float
    price: float | None          # None for market orders
    order_id: int = field(default_factory=lambda: next(_id_counter))
    timestamp: float = field(default_factory=time.monotonic)
    filled_qty: float = 0.0
    status: OrderStatus = OrderStatus.OPEN

    @property
    def remaining_qty(self) -> float:
        return self.qty - self.filled_qty

    def __repr__(self) -> str:
        price_str = f"{self.price:.2f}" if self.price is not None else "MKT"
        return (
            f"Order(id={self.order_id}, {self.side.name} {self.order_type.name} "
            f"qty={self.qty} price={price_str} filled={self.filled_qty} "
            f"status={self.status.name})"
        )


@dataclass
class Trade:
    trade_id: int = field(default_factory=lambda: next(_id_counter))
    buy_order_id: int = 0
    sell_order_id: int = 0
    price: float = 0.0
    qty: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.trade_id}, buy_order={self.buy_order_id}, "
            f"sell_order={self.sell_order_id}, price={self.price:.2f}, qty={self.qty})"
        )


@dataclass
class TickSummary:
    tick: int
    clearing_price: float | None   # last matched price during this tick
    mid_price: float | None        # (best_bid + best_ask) / 2 after matching
    spread: float | None           # best_ask - best_bid after matching
    volume: float                  # total qty traded this tick
    num_trades: int                # number of executions this tick
    open_bids: int                 # resting bid orders carried into next tick
    open_asks: int                 # resting ask orders carried into next tick
    kyles_lambda: float | None = None            # price impact per unit signed order flow
    amihud_illiquidity: float | None = None      # |return| / dollar_volume
    spread_adverse_selection: float | None = None  # adverse-selection component of spread
    spread_inventory_cost: float | None = None     # inventory/order-processing remainder

    def __repr__(self) -> str:
        cp = f"{self.clearing_price:.2f}" if self.clearing_price is not None else "  N/A"
        mp = f"{self.mid_price:.2f}" if self.mid_price is not None else "  N/A"
        sp = f"{self.spread:.2f}" if self.spread is not None else "  N/A"
        return (
            f"Tick {self.tick:>3} | clr={cp:>7} mid={mp:>7} spread={sp:>6} "
            f"vol={self.volume:>7.2f} trades={self.num_trades:>3} "
            f"bids={self.open_bids:>3} asks={self.open_asks:>3}"
        )


class PriceLevel:
    """All orders at a single price, maintained in time order (FIFO)."""

    def __init__(self, price: float) -> None:
        self.price = price
        self._orders: list[Order] = []

    def add(self, order: Order) -> None:
        self._orders.append(order)

    def peek(self) -> Order | None:
        while self._orders and self._orders[0].remaining_qty <= 0:
            self._orders.pop(0)
        return self._orders[0] if self._orders else None

    def pop_front(self) -> Order | None:
        o = self.peek()
        if o:
            self._orders.pop(0)
        return o

    def __len__(self) -> int:
        return sum(1 for o in self._orders if o.remaining_qty > 0)

    @property
    def total_qty(self) -> float:
        return sum(o.remaining_qty for o in self._orders)


class HalfBook:
    """
    One side (bid or ask) of the order book.
    Bids: highest price first.  Asks: lowest price first.
    """

    def __init__(self, side: Side) -> None:
        self.side = side
        self._levels: dict[float, PriceLevel] = {}

    def add(self, order: Order) -> None:
        if order.price not in self._levels:
            self._levels[order.price] = PriceLevel(order.price)
        self._levels[order.price].add(order)

    def best_price(self) -> float | None:
        if not self._levels:
            return None
        if self.side == Side.BUY:
            return max(self._levels)
        return min(self._levels)

    def best_level(self) -> PriceLevel | None:
        price = self.best_price()
        if price is None:
            return None
        lvl = self._levels[price]
        if len(lvl) == 0:
            del self._levels[price]
            return self.best_level()
        return lvl

    def remove_empty_levels(self) -> None:
        empty = [p for p, lvl in self._levels.items() if len(lvl) == 0]
        for p in empty:
            del self._levels[p]

    def depth(self) -> list[tuple[float, float]]:
        """Return [(price, total_qty), ...] sorted best-first."""
        result = [
            (p, lvl.total_qty)
            for p, lvl in self._levels.items()
            if lvl.total_qty > 0
        ]
        reverse = self.side == Side.BUY
        return sorted(result, key=lambda x: x[0], reverse=reverse)

    def drain(self) -> list[Order]:
        """Remove and return all live orders sorted by timestamp (oldest first)."""
        live: list[Order] = []
        for lvl in self._levels.values():
            for o in lvl._orders:
                if o.remaining_qty > 0 and o.status not in (
                    OrderStatus.FILLED, OrderStatus.CANCELLED
                ):
                    live.append(o)
        self._levels.clear()
        live.sort(key=lambda o: o.timestamp)
        return live


class OrderBook:
    def __init__(self, symbol: str = "UNKNOWN") -> None:
        self.symbol = symbol
        self.bids = HalfBook(Side.BUY)
        self.asks = HalfBook(Side.SELL)
        self.order_history: dict[int, Order] = {}
        self.trade_log: list[Trade] = []
        self.last_clearing_price: float | None = None
        self.current_tick: int = 0
        self.price_history: list[TickSummary] = []
        self._tick_trade_start: int = 0   # index into trade_log at tick open

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> list[Trade]:
        self.order_history[order.order_id] = order
        if order.order_type == OrderType.MARKET:
            return self._match_market(order)
        return self._match_limit(order)

    def cancel(self, order_id: int) -> bool:
        order = self.order_history.get(order_id)
        if order and order.status in (OrderStatus.OPEN, OrderStatus.PARTIAL):
            order.status = OrderStatus.CANCELLED
            return True
        return False

    # ------------------------------------------------------------------
    # Tick API
    # ------------------------------------------------------------------

    def mid_price(self) -> float | None:
        """(best_bid + best_ask) / 2, or None if either side is empty."""
        bb = self.bids.best_price()
        ba = self.asks.best_price()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> float | None:
        """best_ask - best_bid, or None if either side is empty."""
        bb = self.bids.best_price()
        ba = self.asks.best_price()
        if bb is None or ba is None:
            return None
        return ba - bb

    def volume_for_tick(self, tick: int) -> float:
        """Total volume traded during a completed tick (from price_history)."""
        if tick < 0 or tick >= len(self.price_history):
            raise IndexError(f"Tick {tick} not in price_history (len={len(self.price_history)})")
        return self.price_history[tick].volume

    def advance_tick(self, new_orders: list[Order] | None = None) -> TickSummary:
        """
        Close the current tick:
          1. Submit all new_orders (matching happens inline).
          2. Snapshot tick stats (clearing price, mid, spread, volume).
          3. Drain surviving limit orders from both sides.
          4. Reset bids/asks to fresh HalfBooks.
          5. Re-insert surviving orders (preserving time priority).
          6. Increment tick counter, append TickSummary to price_history.
        """
        tick_start = self._tick_trade_start

        # Capture mid before this tick's orders arrive (needed for price-change metrics).
        prev_mid = self.mid_price()

        # Signed order flow: buy pressure minus sell pressure from submitted orders.
        buy_flow = sum(o.qty for o in (new_orders or []) if o.side == Side.BUY)
        sell_flow = sum(o.qty for o in (new_orders or []) if o.side == Side.SELL)
        net_flow = buy_flow - sell_flow

        for order in (new_orders or []):
            self.submit(order)

        tick_trades = self.trade_log[tick_start:]
        volume = sum(t.qty for t in tick_trades)
        clearing = tick_trades[-1].price if tick_trades else None
        num_trades = len(tick_trades)

        curr_mid = self.mid_price()
        spread_val = self.spread()

        # Kyle's lambda: regression coefficient of mid-price change on net order flow.
        kyles_lambda: float | None = None
        if net_flow != 0 and prev_mid is not None and curr_mid is not None:
            kyles_lambda = (curr_mid - prev_mid) / net_flow

        # Amihud illiquidity: absolute return scaled by dollar volume.
        amihud_illiquidity: float | None = None
        if volume > 0 and curr_mid is not None and prev_mid is not None and prev_mid > 0:
            abs_return = abs((curr_mid - prev_mid) / prev_mid)
            dollar_vol = volume * (clearing if clearing is not None else curr_mid)
            if dollar_vol > 0:
                amihud_illiquidity = abs_return / dollar_vol

        # Spread decomposition: adverse selection (price-impact driven) + inventory cost.
        spread_adverse_selection: float | None = None
        spread_inventory_cost: float | None = None
        if spread_val is not None and kyles_lambda is not None and num_trades > 0:
            avg_trade_size = volume / num_trades
            # Adverse selection: how much the price permanently moves per average trade.
            adverse = 2.0 * abs(kyles_lambda) * avg_trade_size
            adverse = min(adverse, spread_val)  # cannot exceed total spread
            spread_adverse_selection = adverse
            spread_inventory_cost = spread_val - adverse

        summary = TickSummary(
            tick=self.current_tick,
            clearing_price=clearing,
            mid_price=curr_mid,
            spread=spread_val,
            volume=volume,
            num_trades=num_trades,
            open_bids=sum(len(lvl) for lvl in self.bids._levels.values()),
            open_asks=sum(len(lvl) for lvl in self.asks._levels.values()),
            kyles_lambda=kyles_lambda,
            amihud_illiquidity=amihud_illiquidity,
            spread_adverse_selection=spread_adverse_selection,
            spread_inventory_cost=spread_inventory_cost,
        )

        # Carry over live limit orders; reset book sides.
        surviving_bids = self.bids.drain()
        surviving_asks = self.asks.drain()
        self.bids = HalfBook(Side.BUY)
        self.asks = HalfBook(Side.SELL)
        for o in surviving_bids + surviving_asks:
            half = self.bids if o.side == Side.BUY else self.asks
            half.add(o)

        self.price_history.append(summary)
        self.current_tick += 1
        self._tick_trade_start = len(self.trade_log)
        if clearing is not None:
            self.last_clearing_price = clearing

        return summary

    # ------------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------------

    def _match_limit(self, incoming: Order) -> list[Trade]:
        trades: list[Trade] = []
        opposite = self.asks if incoming.side == Side.BUY else self.bids

        while incoming.remaining_qty > 0:
            best_lvl = opposite.best_level()
            if best_lvl is None:
                break

            # Crossing check
            if incoming.side == Side.BUY and incoming.price < best_lvl.price:
                break
            if incoming.side == Side.SELL and incoming.price > best_lvl.price:
                break

            resting = best_lvl.peek()
            if resting is None or resting.status == OrderStatus.CANCELLED:
                best_lvl.pop_front()
                continue

            trade = self._fill(incoming, resting, resting.price)
            trades.append(trade)
            self.trade_log.append(trade)
            self.last_clearing_price = trade.price

            if resting.remaining_qty <= 0:
                best_lvl.pop_front()

        opposite.remove_empty_levels()

        # Rest unfilled portion on the book
        if incoming.remaining_qty > 0 and incoming.status != OrderStatus.CANCELLED:
            half = self.bids if incoming.side == Side.BUY else self.asks
            half.add(incoming)

        return trades

    def _match_market(self, incoming: Order) -> list[Trade]:
        trades: list[Trade] = []
        opposite = self.asks if incoming.side == Side.BUY else self.bids

        while incoming.remaining_qty > 0:
            best_lvl = opposite.best_level()
            if best_lvl is None:
                break

            resting = best_lvl.peek()
            if resting is None or resting.status == OrderStatus.CANCELLED:
                best_lvl.pop_front()
                continue

            trade = self._fill(incoming, resting, resting.price)
            trades.append(trade)
            self.trade_log.append(trade)
            self.last_clearing_price = trade.price

            if resting.remaining_qty <= 0:
                best_lvl.pop_front()

        opposite.remove_empty_levels()

        if incoming.remaining_qty > 0:
            incoming.status = OrderStatus.CANCELLED  # market orders don't rest

        return trades

    def _fill(self, aggressive: Order, passive: Order, price: float) -> Trade:
        fill_qty = min(aggressive.remaining_qty, passive.remaining_qty)

        aggressive.filled_qty += fill_qty
        passive.filled_qty += fill_qty

        for o in (aggressive, passive):
            if o.remaining_qty <= 0:
                o.status = OrderStatus.FILLED
            elif o.filled_qty > 0:
                o.status = OrderStatus.PARTIAL

        buy_id = aggressive.order_id if aggressive.side == Side.BUY else passive.order_id
        sell_id = passive.order_id if passive.side == Side.SELL else aggressive.order_id

        return Trade(buy_order_id=buy_id, sell_order_id=sell_id, price=price, qty=fill_qty)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_book(self, levels: int = 5) -> None:
        print(f"\n{'='*52}")
        print(f"  Order Book: {self.symbol}")
        if self.last_clearing_price is not None:
            print(f"  Last clearing price: {self.last_clearing_price:.2f}")
        print(f"{'='*52}")
        print(f"  {'ASKS':^46}")
        print(f"  {'Price':>10}  {'Qty':>10}")
        print(f"  {'-'*24}")

        ask_depth = self.asks.depth()[:levels]
        for price, qty in reversed(ask_depth):
            print(f"  {price:>10.2f}  {qty:>10.2f}  <<< ASK")

        spread_str = ""
        best_ask = self.asks.best_price()
        best_bid = self.bids.best_price()
        if best_ask is not None and best_bid is not None:
            spread_str = f"  spread = {best_ask - best_bid:.2f}"
        print(f"\n  --- MID ---{spread_str}\n")

        bid_depth = self.bids.depth()[:levels]
        for price, qty in bid_depth:
            print(f"  {price:>10.2f}  {qty:>10.2f}  BID >>>")

        print(f"  {'-'*24}")
        print(f"  {'BIDS':^46}")
        print(f"{'='*52}\n")

    def print_trade_log(self) -> None:
        print(f"\n{'='*52}")
        print(f"  Trade Log ({len(self.trade_log)} trades)")
        print(f"{'='*52}")
        if not self.trade_log:
            print("  (no trades)")
        for t in self.trade_log:
            print(f"  {t}")
        print(f"{'='*52}\n")

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_order(o: Order) -> dict:
        return {
            "order_id": o.order_id,
            "side": o.side.name,
            "order_type": o.order_type.name,
            "price": o.price,
            "qty": o.qty,
            "filled_qty": o.filled_qty,
            "remaining_qty": o.remaining_qty,
            "status": o.status.name,
            "timestamp": o.timestamp,
        }

    @staticmethod
    def _serialize_trade(t: Trade) -> dict:
        return {
            "trade_id": t.trade_id,
            "buy_order_id": t.buy_order_id,
            "sell_order_id": t.sell_order_id,
            "price": t.price,
            "qty": t.qty,
            "timestamp": t.timestamp,
        }

    @staticmethod
    def _serialize_tick(s: TickSummary) -> dict:
        return {
            "tick": s.tick,
            "clearing_price": s.clearing_price,
            "mid_price": s.mid_price,
            "spread": s.spread,
            "volume": s.volume,
            "num_trades": s.num_trades,
            "open_bids": s.open_bids,
            "open_asks": s.open_asks,
            "kyles_lambda": s.kyles_lambda,
            "amihud_illiquidity": s.amihud_illiquidity,
            "spread_adverse_selection": s.spread_adverse_selection,
            "spread_inventory_cost": s.spread_inventory_cost,
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """
        Serialize the full engine state to JSON.

        Schema
        ------
        {
          "symbol": str,
          "exported_at": float,          # time.time() wall-clock
          "current_tick": int,
          "last_clearing_price": float | null,
          "mid_price": float | null,
          "spread": float | null,
          "tick_history": [ TickSummary, ... ],
          "trade_log": [ Trade, ... ],
          "order_book": {
            "bids": [ {"price": float, "qty": float}, ... ],  # best-first
            "asks": [ {"price": float, "qty": float}, ... ],  # best-first
            "best_bid": float | null,
            "best_ask": float | null,
          },
          "order_history": [ Order, ... ]   # all orders ever submitted
        }

        If `path` is given, the JSON is also written to that file.
        Returns the JSON string in all cases.
        """
        payload = {
            "symbol": self.symbol,
            "exported_at": time.time(),
            "current_tick": self.current_tick,
            "last_clearing_price": self.last_clearing_price,
            "mid_price": self.mid_price(),
            "spread": self.spread(),
            "tick_history": [self._serialize_tick(s) for s in self.price_history],
            "trade_log": [self._serialize_trade(t) for t in self.trade_log],
            "order_book": {
                "bids": [
                    {"price": p, "qty": q} for p, q in self.bids.depth()
                ],
                "asks": [
                    {"price": p, "qty": q} for p, q in self.asks.depth()
                ],
                "best_bid": self.bids.best_price(),
                "best_ask": self.asks.best_price(),
            },
            "order_history": [
                self._serialize_order(o)
                for o in sorted(self.order_history.values(), key=lambda o: o.order_id)
            ],
        }

        out = json.dumps(payload, indent=indent)
        if path:
            with open(path, "w") as fh:
                fh.write(out)
        return out

    def print_price_history(self) -> None:
        print(f"\n{'='*80}")
        print(f"  Price History — {self.symbol}  ({len(self.price_history)} ticks)")
        print(f"{'='*80}")
        header = (
            f"  {'Tick':>4}  {'ClearPx':>8}  {'MidPx':>8}  {'Spread':>7}  "
            f"{'Volume':>8}  {'Trades':>6}  {'Bids':>5}  {'Asks':>5}"
        )
        print(header)
        print(f"  {'-'*74}")
        for s in self.price_history:
            cp = f"{s.clearing_price:.2f}" if s.clearing_price is not None else "    N/A"
            mp = f"{s.mid_price:.2f}" if s.mid_price is not None else "    N/A"
            sp = f"{s.spread:.2f}" if s.spread is not None else "    N/A"
            print(
                f"  {s.tick:>4}  {cp:>8}  {mp:>8}  {sp:>7}  "
                f"{s.volume:>8.2f}  {s.num_trades:>6}  {s.open_bids:>5}  {s.open_asks:>5}"
            )
        print(f"{'='*80}\n")
