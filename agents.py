"""
Agent layer for the LOB matching engine.

Three archetypes:
  RetailAgent        — loss-averse retail investor, emotional and delayed signals
  InstitutionalAgent — VWAP mean-reversion fund manager, methodical and deep-book
  QuantAgent         — momentum quant, aggressive with low risk aversion

Each agent's decide() builds a fixed cognitive-archetype system prompt and a
per-tick user prompt, then hands both to an `LLMBackend` (see llm_backends.py)
which returns the parsed decision: {action, quantity, limit_price, reasoning}.

The backend is selected per archetype via LATTICE_BACKEND /
LATTICE_BACKEND_<ARCHETYPE>, or CLI flags — the agent logic below is identical
whether the decision came from Claude or a local Ollama model.
"""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import llm_backends
from llm_backends import LLMBackend, backend_for
from engine import Order, OrderBook, OrderType, Side, TickSummary

# Authoritative Anthropic model name. Kept as a module global because
# experiments/model_scale.py monkeypatches it to compare model tiers; the
# resolver below keeps AnthropicBackend reading it at call time.
_MODEL = llm_backends.DEFAULT_ANTHROPIC_MODEL
llm_backends.set_anthropic_model_resolver(lambda: _MODEL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Closing line of every agent user prompt. A module constant so it is auditable
# in one place and so pilot_shock_responsiveness.py can swap in the pre-fix
# wording to measure what changing it actually did.
#
# The earlier version ("You must return buy or sell in almost all cases ... The
# market needs your participation") pressured agents to trade every tick
# regardless of context, which suppressed exactly the shock-driven inaction the
# 2x2 is trying to detect.
_CLOSING_INSTRUCTION = (
    "Choose the action you can actually justify from what you have been told. Holding is "
    "a legitimate choice when the situation genuinely warrants standing aside — for "
    "instance when the context makes the risk unattractive at this price. It should not "
    "be a habit: holding tick after tick while the situation keeps changing is not "
    "caution, it is non-participation, and it is not what you are here to do."
)


def _macro_to_nl(macro: dict) -> str:
    """Return a natural-language macro description for the agent prompt.

    Supply-chain event contexts carry a pre-built multi_sentence_narrative that
    is returned directly.  Legacy numeric contexts fall back to constructed text.
    """
    if not macro:
        return "Markets are operating under normal baseline conditions with no significant stress."
    # Supply-chain event format: use the narrative that chains shock to second-order effects.
    narrative = macro.get("multi_sentence_narrative")
    if narrative:
        return narrative
    # Legacy numeric macro format.
    fed = macro.get("fed_rate", 5.25)
    vix = macro.get("vix", 18.0)
    sent = macro.get("sentiment", 0.1)
    if fed > 5.5:
        return (
            f"The Federal Reserve has raised rates to {fed:.2f}%, spiking VIX to {vix:.0f} "
            f"— risk-off conditions are building and borrowing costs are rising sharply."
        )
    if sent < -0.6:
        return (
            f"Market sentiment has collapsed to {sent:.2f} with VIX surging to {vix:.0f} "
            f"— fear is dominating and broad-based selling is emerging across the market."
        )
    if vix > 40:
        return (
            f"Liquidity has dried up severely with VIX at {vix:.0f} and sentiment at {sent:.2f} "
            f"— markets are in acute stress and bid-ask spreads are widening dangerously."
        )
    return (
        f"Markets are calm: VIX at {vix:.0f}, sentiment at {sent:+.2f}, "
        f"Fed rate at {fed:.2f}% — normal baseline trading conditions prevail."
    )


def _parse_response(text: str) -> Optional[dict]:
    """Extract JSON from model output. Retained for backwards compatibility;
    the schema-checking implementation now lives in llm_backends."""
    decision, _reason = llm_backends.parse_decision(text)
    return decision


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class Agent(ABC):
    _SYSTEM_PROMPT: str = ""
    # Archetype key used to resolve which backend this agent talks to.
    # None means the agent never calls an LLM (e.g. the market maker).
    _ARCHETYPE: Optional[str] = None

    def __init__(
        self,
        cash: float,
        position: float,
        lam: float,
        name: str = "",
        backend: Optional[LLMBackend] = None,
    ) -> None:
        self.cash = cash
        self.position = position
        self.lam = lam
        self.name = name
        self._wealth_history: deque[float] = deque(maxlen=5)
        self._consecutive_holds = 0
        self._backend = backend
        # Per-decision telemetry, reset at the top of every decide() call.
        # A forced coin-flip order and a backend failure both used to be
        # indistinguishable from a real decision downstream; they are degeneracy
        # and have to be countable as such.
        self._last_forced_random = False
        self._last_api_failed = False

    @property
    def backend(self) -> LLMBackend:
        """The decision backend, resolved lazily on first use.

        Lazy so agents that never call an LLM (MarketMakerAgent) never
        construct a client, and so config set after __init__ still applies.
        """
        if self._backend is None:
            self._backend = backend_for(self._ARCHETYPE)
        return self._backend

    def utility(self, expected_wealth: float, variance: float) -> float:
        """U = E[W] - (lambda / 2) * Var[W]."""
        return expected_wealth - (self.lam / 2.0) * variance

    def wealth(self, price: float) -> float:
        return self.cash + self.position * price

    def append_wealth(self, w: float) -> None:
        """Record current tick wealth for prompt context (deque keeps last 5)."""
        self._wealth_history.append(w)

    def update_position(self, order: Order, fill_price: float) -> None:
        """Adjust cash and position for a (partially) filled order."""
        fill = order.filled_qty
        if fill <= 0:
            return
        if order.side == Side.BUY:
            self.cash -= fill * fill_price
            self.position += fill
        else:
            self.cash += fill * fill_price
            self.position -= fill

    def _build_user_prompt(
        self, book_snapshot: dict, macro_context: Optional[dict], tick_idx: int
    ) -> str:
        mid = book_snapshot.get("mid_price")
        last_clear = book_snapshot.get("last_clearing_price")
        ref = mid or last_clear or 100.0

        ob = book_snapshot["order_book"]
        best_bid = ob.get("best_bid")
        best_ask = ob.get("best_ask")
        bid_str = f"${best_bid:.2f}" if isinstance(best_bid, float) else "N/A"
        ask_str = f"${best_ask:.2f}" if isinstance(best_ask, float) else "N/A"

        history = book_snapshot.get("tick_history", [])
        recent_prices = [
            t["clearing_price"]
            for t in history[-5:]
            if t.get("clearing_price") is not None
        ]
        last_vol = history[-1].get("volume", 0.0) if history else 0.0

        wealth_now = self.cash + self.position * ref
        wealth_hist_str = (
            ", ".join(f"${w:.0f}" for w in self._wealth_history)
            if self._wealth_history
            else "no prior data"
        )
        prices_str = (
            ", ".join(f"${p:.2f}" for p in recent_prices) if recent_prices else "none yet"
        )

        macro_sentence = _macro_to_nl(macro_context or {})

        return (
            f"Market Tick {tick_idx}\n"
            f"Macro environment: {macro_sentence}\n\n"
            f"Your current state:\n"
            f"  Cash: ${self.cash:.2f}\n"
            f"  Position: {self.position:.1f} shares\n"
            f"  Current wealth (mark-to-market): ${wealth_now:.2f}\n"
            f"  Recent P&L context (last 5 tick wealth): {wealth_hist_str}\n\n"
            f"Market data:\n"
            f"  Reference price: ${ref:.2f}\n"
            f"  Best bid: {bid_str}  |  Best ask: {ask_str}\n"
            f"  Recent clearing prices (last 5 ticks): {prices_str}\n"
            f"  Volume last tick: {last_vol:.1f} shares\n\n"
            f"Respond with ONLY a JSON object, no other text:\n"
            f'{{"action": "buy" or "sell" or "hold", "quantity": <number of shares>, '
            f'"limit_price": <price as number, or null if hold>, '
            f'"reasoning": "<one concise sentence>"}}\n\n'
            f"{_CLOSING_INSTRUCTION}"
        )

    def _response_to_order(self, parsed: dict, ref_price: float) -> Optional[Order]:
        action = str(parsed.get("action", "hold")).lower().strip()
        if action == "hold":
            return None
        qty = float(parsed.get("quantity") or 0)
        if qty <= 0:
            return None
        lp = parsed.get("limit_price")
        limit_price = float(lp) if lp is not None else ref_price
        # Reject prices that are wildly unreasonable
        if limit_price <= 0 or limit_price > ref_price * 5 or limit_price < ref_price * 0.2:
            limit_price = ref_price
        side = Side.BUY if action == "buy" else Side.SELL
        return Order(
            side=side,
            order_type=OrderType.LIMIT,
            qty=qty,
            price=max(round(limit_price, 2), 0.01),
        )

    def _call_api(self, user_prompt: str) -> Optional[dict]:
        """Delegate to the configured backend. None means 'no usable decision'."""
        return self.backend.get_decision(self._SYSTEM_PROMPT, user_prompt)

    def decide(
        self,
        book_snapshot: dict,
        macro_context: Optional[dict] = None,
        tick_idx: int = 0,
    ) -> tuple[Optional[Order], str]:
        """Query the configured backend; return (Order | None, reasoning string)."""
        mid = book_snapshot.get("mid_price")
        last_clear = book_snapshot.get("last_clearing_price")
        ref = mid or last_clear or 100.0

        self._last_forced_random = False
        self._last_api_failed = False

        user_prompt = self._build_user_prompt(book_snapshot, macro_context, tick_idx)
        parsed = self._call_api(user_prompt)

        if parsed is None:
            # Not a hold: the agent never got to decide. Flagged so a failing
            # backend cannot masquerade as a sweep of cautious agents.
            self._last_api_failed = True
            return None, f"{self.name}: API unavailable — holding this tick."

        order = self._response_to_order(parsed, ref)
        reasoning = str(parsed.get("reasoning", "")).strip()

        if order is None:
            self._consecutive_holds += 1
            if self._consecutive_holds >= 3:
                self._consecutive_holds = 0
                # A coin flip, not a decision. Downstream this looks like a
                # real order (len(orders) != 0), so it must be flagged or it
                # inflates the apparent participation rate.
                self._last_forced_random = True
                side = Side.BUY if random.random() < 0.5 else Side.SELL
                order = Order(
                    side=side,
                    order_type=OrderType.LIMIT,
                    qty=1.0,
                    price=round(ref, 2),
                )
                reasoning = (
                    f"{self.name}: 3 consecutive HOLDs — fallback {side.name} 1 @ {ref:.2f}"
                )
        else:
            self._consecutive_holds = 0

        return order, reasoning

    def decide_orders(
        self,
        book_snapshot: dict,
        macro_context: Optional[dict] = None,
        tick_idx: int = 0,
    ) -> tuple[list[Order], str]:
        """Return a list of orders for this tick (default: wraps single-order decide)."""
        order, reasoning = self.decide(book_snapshot, macro_context, tick_idx)
        return ([order] if order is not None else []), reasoning


# ---------------------------------------------------------------------------
# MarketMakerAgent
# ---------------------------------------------------------------------------

class MarketMakerAgent(Agent):
    """Always-on liquidity provider.

    Posts a tight bid and ask at mid ± `spread_half` every tick, with no API call.
    Guarantees there is always crossable liquidity for the LLM agents to trade
    against, even if they place aggressive orders or the seed book gets consumed.
    """

    def __init__(
        self,
        cash: float = 1_000_000.0,
        position: float = 1_000.0,
        spread_half: float = 0.05,
        quote_qty: float = 50.0,
        name: str = "MM-1",
    ) -> None:
        super().__init__(cash, position, lam=0.0, name=name)
        self.spread_half = spread_half
        self.quote_qty = quote_qty
        self._fair_value: Optional[float] = None

    def decide_orders(
        self,
        book_snapshot: dict,
        macro_context: Optional[dict] = None,
        tick_idx: int = 0,
    ) -> tuple[list[Order], str]:
        mid = book_snapshot.get("mid_price")
        last_clear = book_snapshot.get("last_clearing_price")
        ref = mid or last_clear or 100.0

        if self._fair_value is None:
            self._fair_value = ref
        self._fair_value += random.uniform(-0.2, 0.2)

        bid_price = round(self._fair_value - self.spread_half, 2)
        ask_price = round(self._fair_value + self.spread_half, 2)
        bid = Order(Side.BUY, OrderType.LIMIT, qty=self.quote_qty, price=bid_price)
        ask = Order(Side.SELL, OrderType.LIMIT, qty=self.quote_qty, price=ask_price)
        reasoning = (
            f"{self.name}: quoting {bid_price:.2f} / {ask_price:.2f} "
            f"(qty {self.quote_qty:.0f} each)"
        )
        return [bid, ask], reasoning


# ---------------------------------------------------------------------------
# RetailAgent
# ---------------------------------------------------------------------------

class RetailAgent(Agent):
    _SYSTEM_PROMPT = (
        "You are a retail investor making real-time trading decisions in a simulated stock market. "
        "Your cognitive archetype: you are highly loss-averse (you dread losses far more than you "
        "enjoy gains), you rely on recent price memory rather than deep analysis, you get anxious "
        "when volatility is high, and you tend to follow the most recent price direction with a "
        "noisy, emotionally-coloured signal. You trade small quantities — 1 to 5 shares per tick. "
        "Your decisions feel driven by gut instinct and recent price memory.\n\n"
        "Your default behaviour: you feel uncomfortable holding cash when prices are moving. If "
        "the price has changed from the previous tick, your instinct is to buy the dip or sell "
        "before it gets worse. You rarely hold for more than 2 ticks in a row.\n\n"
        "That instinct is your default, not a rule you must obey. You also read the macro "
        "environment you are given each tick, and as a retail investor you react to news "
        "emotionally — often more strongly than the numbers alone would justify. Alarming context "
        "can frighten you out of a dip you would otherwise have bought, and reassuring context can "
        "embolden you. If the context materially changes how risky this feels to you, act on that "
        "instead of the pure price instinct and say why in your reasoning. If the context is "
        "routine, or does not change your read of the situation, follow your default.\n\n"
        "Always respond with ONLY a JSON object — no explanation, no preamble, no markdown. "
        'Exact format: {"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one concise sentence>"}'
    )

    _ARCHETYPE = "retail"

    def __init__(
        self,
        cash: float = 10_000.0,
        position: float = 0.0,
        name: str = "",
        backend: Optional[LLMBackend] = None,
    ) -> None:
        super().__init__(cash, position, lam=0.8, name=name, backend=backend)


# ---------------------------------------------------------------------------
# InstitutionalAgent
# ---------------------------------------------------------------------------

class InstitutionalAgent(Agent):
    _SYSTEM_PROMPT = (
        "You are an institutional fund manager making real-time trading decisions in a simulated "
        "stock market. Your cognitive archetype: disciplined and systematic, you apply VWAP "
        "mean-reversion logic — buy when price is below the recent volume-weighted average, sell "
        "when it is above. You have moderate risk aversion and trade in larger blocks — 10 to 30 "
        "shares per tick. You read the order book depth for secondary signals but your working "
        "anchor is the VWAP mean-reversion thesis.\n\n"
        "Your default rule: if the current price is below the average of the last 3 clearing "
        "prices, buy; if it is above, sell. Under normal conditions you place an order every "
        "tick unless your position is already at maximum.\n\n"
        "Mean reversion rests on an assumption: that the recent average still represents fair "
        "value. The macro environment you are given each tick can invalidate that assumption. A "
        "genuine supply or demand shock re-prices the asset, and reverting toward a stale average "
        "is then the wrong trade — it is how a disciplined desk gets run over. Weigh the context "
        "against your default: if it materially changes your estimate of fair value, deviate — "
        "trade with the re-pricing, cut your size, or stand aside — and state the fundamental "
        "reason in your reasoning. If the context is routine or does not move your fair-value "
        "estimate, follow the mean-reversion default.\n\n"
        "Always respond with ONLY a JSON object — no explanation, no preamble, no markdown. "
        'Exact format: {"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one concise sentence>"}'
    )

    _ARCHETYPE = "institutional"

    def __init__(
        self,
        cash: float = 100_000.0,
        position: float = 0.0,
        name: str = "",
        backend: Optional[LLMBackend] = None,
    ) -> None:
        super().__init__(cash, position, lam=0.3, name=name, backend=backend)


# ---------------------------------------------------------------------------
# QuantAgent
# ---------------------------------------------------------------------------

class QuantAgent(Agent):
    _SYSTEM_PROMPT = (
        "You are a quantitative momentum trader making real-time trading decisions in a simulated "
        "stock market. Your cognitive archetype: you follow price momentum relentlessly. "
        "You have very low risk aversion and act with conviction on clear signals. You trade "
        "medium-sized blocks — 5 to 15 shares per tick. Speed and signal conviction define your "
        "archetype.\n\n"
        "Your default signal is the tape: upward movement means buy, downward movement means "
        "sell, flat price means a small buy. You trade on essentially every tick where there is "
        "price history.\n\n"
        "You treat the macro environment you are given as a signal in its own right, because "
        "information reaches the tape before it is fully in the price — being early is the whole "
        "edge. If the context corroborates the momentum you see, trade it with more conviction "
        "and size up within your range. If it cuts against the tape — price still drifting up "
        "while the context is materially bearish, or the reverse — weigh the context, since you "
        "would rather be early than be run over by the reversal. Name the signal that drove you "
        "in your reasoning. With no meaningful context, follow the momentum default.\n\n"
        "Always respond with ONLY a JSON object — no explanation, no preamble, no markdown. "
        'Exact format: {"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one concise sentence>"}'
    )

    _ARCHETYPE = "quant"

    def __init__(
        self,
        cash: float = 50_000.0,
        position: float = 0.0,
        name: str = "",
        backend: Optional[LLMBackend] = None,
    ) -> None:
        super().__init__(cash, position, lam=0.1, name=name, backend=backend)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class Simulation:
    """
    Holds one OrderBook and a list of agents.
    Each tick: serialize book → agents call API → advance_tick → update positions → log wealth.
    """

    def __init__(self, book: OrderBook, agents: list[Agent]) -> None:
        self.book = book
        self.agents = agents
        self.wealth_log: list[list[float]] = []
        self.macro_log: list[dict] = []
        self.reasoning_log: list[list[str]] = []  # [tick_idx][agent_idx]
        self._order_map: dict[int, Agent] = {}

    def seed_book(self, fair_value: float = 100.0, levels: int = 3, qty: float = 10.0) -> None:
        """Prime the book with symmetric limit orders around fair_value, then run a
        few synthetic warm-up ticks that produce real trades so the LLM agents
        have clearing-price history (momentum / VWAP signals) on tick 0.

        Without this priming the QuantAgent refuses to trade with fewer than 3
        clearing prices and the InstitutionalAgent has no VWAP anchor — every
        agent returns "hold" forever and no real trades ever occur.
        """
        seed: list[Order] = []
        for i in range(1, levels + 1):
            seed.append(Order(Side.BUY, OrderType.LIMIT, qty=qty, price=round(fair_value - i * 0.5, 2)))
            seed.append(Order(Side.SELL, OrderType.LIMIT, qty=qty, price=round(fair_value + i * 0.5, 2)))
        self.book.advance_tick(seed)

        # Warm-up ticks. Each tick:
        #  1. Replenishes the inner ±0.5 level.
        #  2. Inserts a fresh ask level at `target_px` strictly inside the spread
        #     so it becomes the new best ask, then matches it with a buy at the
        #     same price — recording a clearing trade exactly at `target_px`.
        # The target_px sequence drifts upward to give the QuantAgent a clear
        # momentum signal and the InstitutionalAgent a usable VWAP anchor.
        target_prices = [99.75, 99.90, 100.05, 100.20, 100.35, 100.40, 100.45]
        for target_px in target_prices:
            warm = [
                Order(Side.BUY,  OrderType.LIMIT, qty=qty, price=round(fair_value - 0.5, 2)),
                Order(Side.SELL, OrderType.LIMIT, qty=qty, price=round(fair_value + 0.5, 2)),
                Order(Side.SELL, OrderType.LIMIT, qty=2.0, price=round(target_px, 2)),
                Order(Side.BUY,  OrderType.LIMIT, qty=2.0, price=round(target_px, 2)),
            ]
            self.book.advance_tick(warm)

    def run(
        self,
        n_ticks: int,
        factor_pulse=None,
        verbose: bool = False,
    ) -> list[TickSummary]:
        """Run n_ticks ticks; return list of TickSummary.

        If verbose=True, prints per-tick book state and every order each agent
        submits (price, qty, distance from mid, and whether it crosses).
        """
        summaries: list[TickSummary] = []

        for tick_idx in range(n_ticks):
            snap = json.loads(self.book.to_json())

            if verbose:
                ob = snap.get("order_book", {})
                mid = snap.get("mid_price")
                bb = ob.get("best_bid")
                ba = ob.get("best_ask")
                mid_s = f"{mid:.2f}" if isinstance(mid, (int, float)) else "N/A"
                bb_s = f"{bb:.2f}" if isinstance(bb, (int, float)) else "N/A"
                ba_s = f"{ba:.2f}" if isinstance(ba, (int, float)) else "N/A"
                print(
                    f"\n--- Tick {tick_idx} ---  mid={mid_s}  "
                    f"best_bid={bb_s}  best_ask={ba_s}  "
                    f"bid_levels={len(ob.get('bids', []))}  "
                    f"ask_levels={len(ob.get('asks', []))}"
                )

            macro_context: Optional[dict] = None
            if factor_pulse is not None:
                state = factor_pulse.get_state(tick_idx)
                macro_context = factor_pulse.to_context(state)
            self.macro_log.append(macro_context or {})

            ref_for_dist = (
                snap.get("mid_price")
                or snap.get("last_clearing_price")
                or 100.0
            )
            best_bid_snap = snap.get("order_book", {}).get("best_bid")
            best_ask_snap = snap.get("order_book", {}).get("best_ask")

            new_orders: list[Order] = []
            tick_reasoning: list[str] = []
            for agent in self.agents:
                orders, reasoning = agent.decide_orders(snap, macro_context, tick_idx)
                tick_reasoning.append(reasoning)
                for o in orders:
                    new_orders.append(o)
                    self._order_map[o.order_id] = agent
                if verbose:
                    if not orders:
                        print(f"  {agent.name:<14} HOLD")
                    else:
                        for o in orders:
                            if o.side == Side.BUY and isinstance(best_ask_snap, (int, float)):
                                cross = "CROSSES" if o.price >= best_ask_snap else "rests"
                            elif o.side == Side.SELL and isinstance(best_bid_snap, (int, float)):
                                cross = "CROSSES" if o.price <= best_bid_snap else "rests"
                            else:
                                cross = "rests"
                            dist = o.price - ref_for_dist
                            print(
                                f"  {agent.name:<14} {o.side.name:<4} "
                                f"qty={o.qty:>6.1f}  price={o.price:>7.2f}  "
                                f"(mid{dist:+.2f}) [{cross}]"
                            )
            self.reasoning_log.append(tick_reasoning)

            summary = self.book.advance_tick(new_orders)
            summaries.append(summary)

            if verbose:
                clr = (
                    f"{summary.clearing_price:.2f}"
                    if summary.clearing_price is not None else "N/A"
                )
                print(
                    f"  → trades={summary.num_trades}  "
                    f"vol={summary.volume:.1f}  clear={clr}"
                )

            # Credit fills to agents; use clearing price as fill price proxy
            fill_price = summary.clearing_price or self.book.last_clearing_price or 100.0
            for oid, agent in self._order_map.items():
                order = self.book.order_history.get(oid)
                if order is not None and order.filled_qty > 0:
                    agent.update_position(order, fill_price)
            self._order_map = {}

            ref_price = fill_price
            tick_wealth = [a.wealth(ref_price) for a in self.agents]
            self.wealth_log.append(tick_wealth)
            for a, w in zip(self.agents, tick_wealth):
                a.append_wealth(w)

        return summaries
