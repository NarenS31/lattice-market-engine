"""
extensions/base_market.py

Single-file adaptation of the core market mechanism from Lopez-Lira (2025),
"Can Large Language Models Trade? Testing Financial Theories with LLM Agents
in Market Simulations" (https://github.com/alejandroll10/llm_trading_sim).

What is preserved from the upstream design:
  * Persistent limit-order book with price-time priority (limits rest across
    rounds; partial fills are supported).
  * Stochastic per-round dividends (high / low realisation), an interest rate
    paid on cash, and a terminal redemption value.
  * Fundamental value = PV of expected dividends + PV of redemption,
    calibrated so the FV stays constant across rounds.
  * LLM agent archetypes from the paper (Value, Momentum, Market Maker,
    Contrarian) plus a generic LLM agent base class.
  * A shared `BaseSimulation` that drives the round loop:
        agents observe state -> submit orders -> match -> pay dividends -> mark
        wealth -> repeat.

What is intentionally dropped (not relevant to the research extensions):
  multi-stock support, leverage / borrowing, social-feed and news scenarios,
  the full service / repository / verifier architecture, plot generation,
  and provider abstractions. The aim of this file is a compact, readable
  reproduction of the upstream market mechanism so the extensions in
  ``supply_chain_extensions.py`` can layer cleanly on top.

Reference: Lopez-Lira, A. (2025). https://ssrn.com/abstract=5217340
"""
from __future__ import annotations

import json
import os
import random
import time
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Order / Trade primitives
# ---------------------------------------------------------------------------

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
    price: Optional[float]                   # None for market orders
    agent_id: str = ""
    order_id: int = field(default_factory=lambda: next(_id_counter))
    timestamp: float = field(default_factory=time.monotonic)
    filled_qty: float = 0.0
    status: OrderStatus = OrderStatus.OPEN

    @property
    def remaining_qty(self) -> float:
        return self.qty - self.filled_qty


@dataclass
class Trade:
    buy_order_id: int
    sell_order_id: int
    buyer_id: str
    seller_id: str
    price: float
    qty: float


@dataclass
class MarketResult:
    """Per-round outcome of the matching engine."""
    round_number: int
    clearing_price: Optional[float]
    mid_price: Optional[float]
    spread: Optional[float]
    volume: float
    num_trades: int
    trades: list


# ---------------------------------------------------------------------------
# Order book — price-time priority, persistent across rounds
# ---------------------------------------------------------------------------

class OrderBook:
    """
    A single-stock LOB. Limits that don't fully execute persist across rounds
    in best-price / earliest-time order; market orders are filled as much as
    possible against resting liquidity and any unfilled remainder is dropped.
    """

    def __init__(self) -> None:
        self.bids: list[Order] = []   # sorted: high price, then early ts
        self.asks: list[Order] = []   # sorted: low price, then early ts
        self.trade_log: list[Trade] = []
        self.last_clearing_price: Optional[float] = None

    # -- queries ------------------------------------------------------------

    def best_bid(self) -> Optional[float]:
        for o in self.bids:
            if o.remaining_qty > 0:
                return o.price
        return None

    def best_ask(self) -> Optional[float]:
        for o in self.asks:
            if o.remaining_qty > 0:
                return o.price
        return None

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def book_snapshot(self, depth: int = 5) -> dict:
        bids = [(o.price, o.remaining_qty) for o in self.bids if o.remaining_qty > 0]
        asks = [(o.price, o.remaining_qty) for o in self.asks if o.remaining_qty > 0]
        return {
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "mid": self.mid_price(),
            "spread": self.spread(),
            "bids": bids[:depth],
            "asks": asks[:depth],
            "last_clearing_price": self.last_clearing_price,
        }

    # -- internal book maintenance -----------------------------------------

    def _resort(self) -> None:
        # Drop fully filled / cancelled orders, then sort.
        self.bids = [o for o in self.bids
                     if o.remaining_qty > 0 and o.status != OrderStatus.CANCELLED]
        self.asks = [o for o in self.asks
                     if o.remaining_qty > 0 and o.status != OrderStatus.CANCELLED]
        self.bids.sort(key=lambda o: (-o.price, o.timestamp))
        self.asks.sort(key=lambda o: (o.price, o.timestamp))

    def _add_resting(self, order: Order) -> None:
        if order.side == Side.BUY:
            self.bids.append(order)
        else:
            self.asks.append(order)

    # -- matching ----------------------------------------------------------

    def _fill(self, aggressive: Order, passive: Order, price: float) -> Trade:
        qty = min(aggressive.remaining_qty, passive.remaining_qty)
        aggressive.filled_qty += qty
        passive.filled_qty += qty
        for o in (aggressive, passive):
            if o.remaining_qty <= 0:
                o.status = OrderStatus.FILLED
            elif o.filled_qty > 0:
                o.status = OrderStatus.PARTIAL
        if aggressive.side == Side.BUY:
            buyer, seller = aggressive, passive
        else:
            buyer, seller = passive, aggressive
        trade = Trade(
            buy_order_id=buyer.order_id,
            sell_order_id=seller.order_id,
            buyer_id=buyer.agent_id,
            seller_id=seller.agent_id,
            price=price,
            qty=qty,
        )
        self.trade_log.append(trade)
        self.last_clearing_price = price
        return trade

    def submit(self, order: Order) -> list[Trade]:
        """Match `order` against the resting opposite side; rest the remainder."""
        trades: list[Trade] = []
        opposite = self.asks if order.side == Side.BUY else self.bids

        while order.remaining_qty > 0:
            # Skip dead orders at the head of the queue.
            while opposite and (
                opposite[0].remaining_qty <= 0
                or opposite[0].status == OrderStatus.CANCELLED
            ):
                opposite.pop(0)
            if not opposite:
                break
            top = opposite[0]
            if order.order_type == OrderType.LIMIT:
                if order.side == Side.BUY and order.price < top.price:
                    break
                if order.side == Side.SELL and order.price > top.price:
                    break
            trades.append(self._fill(order, top, top.price))
            if top.remaining_qty <= 0:
                opposite.pop(0)

        if order.order_type == OrderType.LIMIT and order.remaining_qty > 0:
            self._add_resting(order)
        elif order.order_type == OrderType.MARKET and order.remaining_qty > 0:
            order.status = OrderStatus.CANCELLED  # market orders don't rest

        self._resort()
        return trades

    def match_round(self, new_orders: list[Order], round_number: int) -> MarketResult:
        """Submit a batch of orders and return a MarketResult for the round."""
        round_trades: list[Trade] = []
        # Process in submission order — gives randomised agents a fair shot.
        for o in new_orders:
            round_trades.extend(self.submit(o))

        clearing = round_trades[-1].price if round_trades else None
        volume = sum(t.qty for t in round_trades)
        return MarketResult(
            round_number=round_number,
            clearing_price=clearing,
            mid_price=self.mid_price(),
            spread=self.spread(),
            volume=volume,
            num_trades=len(round_trades),
            trades=round_trades,
        )


# ---------------------------------------------------------------------------
# Dividend model & fundamental value (Lopez-Lira §2)
# ---------------------------------------------------------------------------

@dataclass
class DividendModel:
    """High/low dividend coin-flip used in the upstream paper.

    Default parameters reproduce the canonical experiment:
        E[d] = 0.5 * 1.6 + 0.5 * 1.2 = 1.4
    """
    high: float = 1.6
    low: float = 1.2
    prob_high: float = 0.5

    @property
    def expected(self) -> float:
        return self.prob_high * self.high + (1 - self.prob_high) * self.low

    def realise(self) -> float:
        return self.high if random.random() < self.prob_high else self.low


def fundamental_value(
    num_rounds: int,
    expected_dividend: float,
    interest_rate: float,
    redemption_value: float,
    current_round: int = 1,
) -> float:
    """PV of remaining expected dividends plus PV of terminal redemption.

    FV_t = E[d] * sum_{tau=1..T-t+1} (1+r)^-tau  +  K * (1+r)^-(T-t+1)

    With the calibration K = E[d]/r * (1+r)^T this is constant across rounds.
    """
    remaining = num_rounds - current_round + 1
    fv = sum(expected_dividend / (1 + interest_rate) ** t
             for t in range(1, remaining + 1))
    fv += redemption_value / (1 + interest_rate) ** remaining
    return fv


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Holds the cash/shares state, a trade history, and exposes decide()."""

    def __init__(
        self,
        agent_id: str,
        initial_cash: float,
        initial_shares: int = 0,
        position_limit: Optional[int] = None,
    ) -> None:
        self.agent_id = agent_id
        self.cash = initial_cash
        self.shares = initial_shares
        self.position_limit = position_limit
        self.dividend_cash: float = 0.0
        self.trade_history: list[Trade] = []
        self.decision_history: list[dict] = []
        self.archetype: str = self.__class__.__name__

    # -- accounting --------------------------------------------------------

    def wealth(self, mark_price: float) -> float:
        return self.cash + self.shares * mark_price + self.dividend_cash

    def apply_trade(self, trade: Trade) -> None:
        if trade.buyer_id == self.agent_id:
            self.cash -= trade.price * trade.qty
            self.shares += int(trade.qty)
        elif trade.seller_id == self.agent_id:
            self.cash += trade.price * trade.qty
            self.shares -= int(trade.qty)
        self.trade_history.append(trade)

    def receive_dividend(self, per_share: float) -> None:
        self.dividend_cash += per_share * max(self.shares, 0)

    def receive_interest(self, rate: float) -> None:
        if self.cash > 0:
            self.cash += self.cash * rate

    def receive_redemption(self, redemption_value: float) -> None:
        self.cash += self.shares * redemption_value
        self.shares = 0

    # -- decision interface -----------------------------------------------

    @abstractmethod
    def decide(self, market_state: dict) -> tuple[list[Order], str]:
        """Return (orders_to_submit, reasoning_string)."""
        ...


# ---------------------------------------------------------------------------
# LLM agent — Anthropic-backed concrete agent
# ---------------------------------------------------------------------------

try:
    import anthropic
    _client: Optional["anthropic.Anthropic"] = anthropic.Anthropic()
except Exception:
    _client = None

_MODEL = os.environ.get("LATTICE_LLM_MODEL", "claude-sonnet-4-20250514")


class LLMAgent(BaseAgent):
    """Concrete agent that calls the Anthropic API.

    Each archetype subclass overrides `_SYSTEM_PROMPT`; the user prompt is
    constructed in `_build_user_prompt()` from the market state passed to
    `decide()`. Subclasses may override `_extra_user_lines()` to inject
    extra context (this is the hook the supply-chain extensions use).
    """

    _SYSTEM_PROMPT: str = (
        "You are a trading agent in a financial market simulation. "
        "Respond with a single JSON object only."
    )

    def _extra_user_lines(self, market_state: dict) -> list[str]:
        """Hook for subclasses / mixins to add extra context lines."""
        return []

    def _build_user_prompt(self, market_state: dict) -> str:
        round_no = market_state["round_number"]
        ref_price = (
            market_state.get("mid")
            or market_state.get("last_clearing_price")
            or market_state.get("initial_price", 28.0)
        )
        bb = market_state.get("best_bid")
        ba = market_state.get("best_ask")
        bid_str = f"${bb:.2f}" if isinstance(bb, (int, float)) else "N/A"
        ask_str = f"${ba:.2f}" if isinstance(ba, (int, float)) else "N/A"
        recent = market_state.get("recent_clearing_prices", [])
        recent_str = ", ".join(f"${p:.2f}" for p in recent[-5:]) or "no trades yet"

        fv = market_state.get("fundamental_value")
        fv_str = f"${fv:.2f}" if isinstance(fv, (int, float)) else "unknown"
        ed = market_state.get("expected_dividend", 0.0)
        rate = market_state.get("interest_rate", 0.0)
        rounds_left = market_state.get("rounds_remaining", 0)

        wealth_now = self.wealth(ref_price)

        lines = [
            f"Round {round_no} of {market_state.get('num_rounds', '?')}.",
            f"Market: best_bid={bid_str}, best_ask={ask_str}, "
            f"mid=${ref_price:.2f}, recent clearing prices: {recent_str}.",
            f"Fundamentals: expected dividend ${ed:.2f}/share, interest rate "
            f"{rate*100:.1f}%, rounds remaining {rounds_left}, "
            f"fundamental value {fv_str}.",
            f"Your position: cash=${self.cash:.2f}, shares={self.shares}, "
            f"dividend account=${self.dividend_cash:.2f}, "
            f"current wealth=${wealth_now:.2f}.",
        ]
        lines.extend(self._extra_user_lines(market_state))
        lines.append(
            "Respond with ONLY a JSON object, no preamble or markdown:\n"
            '{"action": "buy"|"sell"|"hold", "quantity": <int shares>, '
            '"limit_price": <number or null>, "reasoning": "<one sentence>"}'
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        s = text.strip()
        if s.startswith("```"):
            parts = s.split("\n")
            end = len(parts) - 1 if parts[-1].strip() == "```" else len(parts)
            s = "\n".join(parts[1:end])
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None

    def _call_llm(self, user_prompt: str) -> Optional[dict]:
        if _client is None:
            return None
        try:
            resp = _client.messages.create(
                model=_MODEL,
                max_tokens=256,
                system=[{
                    "type": "text",
                    "text": self._SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_prompt}],
            )
            return self._parse_json(resp.content[0].text)
        except Exception:
            return None

    def decide(self, market_state: dict) -> tuple[list[Order], str]:
        prompt = self._build_user_prompt(market_state)
        parsed = self._call_llm(prompt)
        if parsed is None:
            return [], f"{self.agent_id}: API unavailable, holding."

        self.decision_history.append(parsed)
        action = str(parsed.get("action", "hold")).lower().strip()
        reasoning = str(parsed.get("reasoning", "")).strip()
        if action not in ("buy", "sell"):
            return [], f"{self.agent_id}: hold — {reasoning}"

        try:
            qty = int(float(parsed.get("quantity") or 0))
        except (TypeError, ValueError):
            qty = 0
        if qty <= 0:
            return [], f"{self.agent_id}: hold (zero qty) — {reasoning}"

        ref_price = (
            market_state.get("mid")
            or market_state.get("last_clearing_price")
            or market_state.get("initial_price", 28.0)
        )
        lp = parsed.get("limit_price")
        try:
            limit_price = float(lp) if lp is not None else float(ref_price)
        except (TypeError, ValueError):
            limit_price = float(ref_price)
        # Clamp pathological prints back to a sane range around the mid.
        if limit_price <= 0 or limit_price > ref_price * 5 or limit_price < ref_price * 0.2:
            limit_price = float(ref_price)

        side = Side.BUY if action == "buy" else Side.SELL
        order = Order(
            side=side,
            order_type=OrderType.LIMIT,
            qty=float(qty),
            price=round(limit_price, 2),
            agent_id=self.agent_id,
        )
        return [order], f"{self.agent_id}: {action.upper()} {qty}@{limit_price:.2f} — {reasoning}"


# ---------------------------------------------------------------------------
# Archetype prompts (preserved from Lopez-Lira 2025, lightly trimmed)
# ---------------------------------------------------------------------------

_VALUE_PROMPT = (
    "You are a value investor who focuses on fundamental analysis. "
    "You believe in mean reversion: buy when the market price is below "
    "fundamental value, sell when it is above. Trade modest sizes (5-15 "
    "shares) and prefer limit orders close to fundamental value."
)

_MOMENTUM_PROMPT = (
    "You are a momentum trader who follows price trends and volume. "
    "When recent clearing prices trend up you buy; when they trend down "
    "you sell. Size trades 5-15 shares and place limit orders that cross "
    "or sit just inside the spread when conviction is high."
)

_MARKET_MAKER_PROMPT = (
    "You are a professional market maker providing liquidity. Your edge "
    "is capturing the bid-ask spread, not directional bets. Each round, "
    "place a limit BUY 1-3% below the mid and a limit SELL 1-3% above "
    "the mid, in sizes 10-30 shares. Never invert your quotes."
)

_CONTRARIAN_PROMPT = (
    "You are a contrarian trader. Markets often overreact, so when the "
    "recent move has been sharply down you buy, and when it has been "
    "sharply up you sell. Trade 5-15 shares, prefer limits inside the "
    "spread on reversal entries."
)

_JSON_TAIL = (
    "\n\nAlways respond with ONLY a single JSON object — no markdown, no "
    "preamble. Format: "
    '{"action": "buy"|"sell"|"hold", "quantity": <int>, '
    '"limit_price": <number or null>, "reasoning": "<one sentence>"}'
)


class ValueInvestor(LLMAgent):
    _SYSTEM_PROMPT = _VALUE_PROMPT + _JSON_TAIL


class MomentumTrader(LLMAgent):
    _SYSTEM_PROMPT = _MOMENTUM_PROMPT + _JSON_TAIL


class MarketMaker(LLMAgent):
    _SYSTEM_PROMPT = _MARKET_MAKER_PROMPT + _JSON_TAIL

    def decide(self, market_state: dict) -> tuple[list[Order], str]:
        """Market maker submits a paired bid + ask each round.

        We run the LLM call to get the size, but always submit two orders
        bracketing the mid even if the model emits a single side.
        """
        orders, reasoning = super().decide(market_state)
        ref = (
            market_state.get("mid")
            or market_state.get("last_clearing_price")
            or market_state.get("initial_price", 28.0)
        )
        # If the model returned one side, mirror it on the other side at
        # symmetric distance; if it returned nothing, post a default 2% quote.
        size = 10.0
        if orders:
            size = orders[0].qty
        bid = Order(Side.BUY, OrderType.LIMIT, size,
                    round(ref * 0.98, 2), agent_id=self.agent_id)
        ask = Order(Side.SELL, OrderType.LIMIT, size,
                    round(ref * 1.02, 2), agent_id=self.agent_id)
        return [bid, ask], reasoning


class Contrarian(LLMAgent):
    _SYSTEM_PROMPT = _CONTRARIAN_PROMPT + _JSON_TAIL


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

class BaseSimulation:
    """Drive the multi-round market with dividends, interest, and redemption.

    Each round:
      1. Build a market_state dict and ask each agent for orders.
      2. Match all orders, settle trades against agents, log a MarketResult.
      3. If `dividend_model.should_pay(round)`, draw a realisation and
         credit each agent's dividend account.
      4. Pay interest on cash balances.
      5. Mark wealth.
    At the final round, every share is redeemed at `redemption_value`.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        num_rounds: int = 25,
        initial_price: float = 28.0,
        redemption_value: float = 28.0,
        interest_rate: float = 0.05,
        dividend_model: Optional[DividendModel] = None,
        dividend_frequency: int = 1,
    ) -> None:
        self.agents = agents
        self.num_rounds = num_rounds
        self.initial_price = initial_price
        self.redemption_value = redemption_value
        self.interest_rate = interest_rate
        self.dividend_model = dividend_model or DividendModel()
        self.dividend_frequency = dividend_frequency

        self.book = OrderBook()
        self.results: list[MarketResult] = []
        self.wealth_log: list[dict[str, float]] = []
        self.recent_clearing_prices: list[float] = []
        self.dividend_realisations: list[Optional[float]] = []
        self.reasoning_log: list[list[str]] = []

    # -- state --------------------------------------------------------------

    def _market_state(self, round_no: int, extra: Optional[dict] = None) -> dict:
        snap = self.book.book_snapshot()
        snap.update({
            "round_number": round_no,
            "num_rounds": self.num_rounds,
            "rounds_remaining": self.num_rounds - round_no + 1,
            "initial_price": self.initial_price,
            "expected_dividend": self.dividend_model.expected,
            "interest_rate": self.interest_rate,
            "redemption_value": self.redemption_value,
            "fundamental_value": fundamental_value(
                self.num_rounds,
                self.dividend_model.expected,
                self.interest_rate,
                self.redemption_value,
                round_no,
            ),
            "recent_clearing_prices": list(self.recent_clearing_prices[-5:]),
        })
        if extra:
            snap.update(extra)
        return snap

    def _settle_trades(self, trades: list[Trade]) -> None:
        by_id = {a.agent_id: a for a in self.agents}
        for tr in trades:
            buyer = by_id.get(tr.buyer_id)
            seller = by_id.get(tr.seller_id)
            if buyer is not None:
                buyer.apply_trade(tr)
            if seller is not None:
                seller.apply_trade(tr)

    # -- run loop -----------------------------------------------------------

    def run(self, verbose: bool = True, state_extra_fn=None) -> list[MarketResult]:
        """Execute `num_rounds` rounds. `state_extra_fn(round_no)` may return
        a dict that is merged into the per-round market_state (extension hook)."""
        for round_no in range(1, self.num_rounds + 1):
            extra = state_extra_fn(round_no) if state_extra_fn else None
            market_state = self._market_state(round_no, extra)

            new_orders: list[Order] = []
            tick_reasons: list[str] = []
            for agent in self.agents:
                orders, reasoning = agent.decide(market_state)
                new_orders.extend(orders)
                tick_reasons.append(reasoning)
            self.reasoning_log.append(tick_reasons)

            result = self.book.match_round(new_orders, round_no)
            self._settle_trades(result.trades)
            self.results.append(result)
            if result.clearing_price is not None:
                self.recent_clearing_prices.append(result.clearing_price)

            # Dividends
            if round_no % self.dividend_frequency == 0:
                d = self.dividend_model.realise()
                for a in self.agents:
                    a.receive_dividend(d)
                self.dividend_realisations.append(d)
            else:
                self.dividend_realisations.append(None)

            # Interest on cash
            for a in self.agents:
                a.receive_interest(self.interest_rate)

            # Mark wealth
            mark = (
                result.clearing_price
                or self.book.mid_price()
                or self.initial_price
            )
            self.wealth_log.append({a.agent_id: a.wealth(mark) for a in self.agents})

            if verbose:
                cp = f"{result.clearing_price:.2f}" if result.clearing_price else "N/A"
                d_str = (
                    f"d=${self.dividend_realisations[-1]:.2f}"
                    if self.dividend_realisations[-1] is not None else "no div"
                )
                print(
                    f"Round {round_no:>2} | clr={cp:>6} mid="
                    f"{(result.mid_price or 0):>6.2f} vol={result.volume:>5.1f} "
                    f"trades={result.num_trades:>2} {d_str}"
                )

        # Terminal redemption
        for a in self.agents:
            a.receive_redemption(self.redemption_value)

        return self.results
