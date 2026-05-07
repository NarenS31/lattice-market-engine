"""
extensions/supply_chain_extensions.py

Three novel research contributions layered on top of the Lopez-Lira (2025)
LLM market in `base_market.py`:

  1. Supply-chain causal narrative shocks.
     At a configurable round (default 10), one of four real-world archetype
     events fires (lithium, semiconductor, energy, credit) with a
     multi-sentence narrative chaining the immediate shock to second-order
     effects. The narrative is injected into every agent's user prompt so
     the LLM observes news in natural language, not numeric severity.

  2. Microstructure metrics in the per-round summary.
     Each round we attach Kyle's lambda (price impact per unit signed
     order flow), Amihud illiquidity (|return| / dollar volume), and a
     spread decomposition into adverse-selection and inventory components.

  3. Per-agent rolling P&L memory.
     Each agent gets a deque[5] of recent wealth marks; the prompt builder
     reads these out so the LLM observes its own short-run P&L trajectory
     each round.

The supply-chain shock catalogue is reused from the project root's
`supply_chain.py` (sibling import handled below).
"""
from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# Allow `from extensions/...` to reach the project-root `supply_chain.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from supply_chain import SupplyChainPulse, SupplyChainState  # noqa: E402

from .base_market import (  # noqa: E402
    BaseAgent,
    BaseSimulation,
    DividendModel,
    LLMAgent,
    MarketResult,
    Order,
    OrderBook,
    Side,
    Trade,
    ValueInvestor,
    MomentumTrader,
    MarketMaker,
    Contrarian,
)


# ---------------------------------------------------------------------------
# Contribution 1: supply-chain shock narrative injection
# ---------------------------------------------------------------------------

class _NarrativeMixin:
    """Mixin that adds the narrative line to the agent's user prompt.

    The simulation pushes the active supply-chain narrative into
    `self._active_narrative` before each round. `LLMAgent._extra_user_lines`
    pulls it back out into the prompt.
    """

    _active_narrative: Optional[str] = None
    _active_event: Optional[str] = None
    _active_severity: Optional[float] = None
    _active_sectors: list[str] = []

    def set_supply_chain_state(self, state: Optional[SupplyChainState]) -> None:
        if state is None:
            self._active_narrative = None
            self._active_event = None
            self._active_severity = None
            self._active_sectors = []
            return
        self._active_narrative = state.multi_sentence_narrative
        self._active_event = state.event_name
        self._active_severity = state.severity
        self._active_sectors = list(state.affected_sectors)

    def _extra_user_lines(self, market_state: dict) -> list[str]:  # type: ignore[override]
        lines = super()._extra_user_lines(market_state)  # type: ignore[misc]
        # Wealth memory line (Contribution 3, set by ExtendedAgentMixin).
        wealth_hist = getattr(self, "_wealth_memory", None)
        if wealth_hist:
            hist_str = ", ".join(f"${w:.2f}" for w in wealth_hist)
            lines.append(f"Recent P&L (last {len(wealth_hist)} rounds wealth): {hist_str}.")
        # Supply-chain narrative line.
        if self._active_narrative:
            sectors = ", ".join(self._active_sectors) or "broad market"
            lines.append(
                f"Breaking news (severity {self._active_severity:.1f}, sectors: "
                f"{sectors}): {self._active_narrative}"
            )
        return lines


# ---------------------------------------------------------------------------
# Contribution 3: rolling P&L memory deque on each agent
# ---------------------------------------------------------------------------

class _WealthMemoryMixin:
    """Adds a deque[5] of recent wealth marks to a BaseAgent subclass."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wealth_memory: deque[float] = deque(maxlen=5)

    def record_wealth(self, w: float) -> None:
        self._wealth_memory.append(w)


# ---------------------------------------------------------------------------
# Concrete extended agent classes — same archetypes, with both extensions on.
# ---------------------------------------------------------------------------

class ExtendedValueInvestor(_NarrativeMixin, _WealthMemoryMixin, ValueInvestor):
    pass


class ExtendedMomentumTrader(_NarrativeMixin, _WealthMemoryMixin, MomentumTrader):
    pass


class ExtendedMarketMaker(_NarrativeMixin, _WealthMemoryMixin, MarketMaker):
    pass


class ExtendedContrarian(_NarrativeMixin, _WealthMemoryMixin, Contrarian):
    pass


# ---------------------------------------------------------------------------
# Contribution 2: microstructure metrics on the per-round summary
# ---------------------------------------------------------------------------

@dataclass
class ExtendedMarketResult:
    """A MarketResult plus three microstructure metrics."""
    base: MarketResult
    kyles_lambda: Optional[float] = None
    amihud_illiquidity: Optional[float] = None
    spread_adverse_selection: Optional[float] = None
    spread_inventory_cost: Optional[float] = None
    active_event: Optional[str] = None
    active_severity: Optional[float] = None

    @property
    def round_number(self) -> int:
        return self.base.round_number

    @property
    def clearing_price(self) -> Optional[float]:
        return self.base.clearing_price

    @property
    def mid_price(self) -> Optional[float]:
        return self.base.mid_price

    @property
    def spread(self) -> Optional[float]:
        return self.base.spread

    @property
    def volume(self) -> float:
        return self.base.volume

    @property
    def num_trades(self) -> int:
        return self.base.num_trades

    @property
    def trades(self) -> list[Trade]:
        return self.base.trades


def compute_microstructure_metrics(
    prev_mid: Optional[float],
    curr_mid: Optional[float],
    new_orders: list[Order],
    result: MarketResult,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (kyles_lambda, amihud_illiquidity, adverse_selection, inventory).

    * Kyle's lambda = (curr_mid - prev_mid) / signed_order_flow.
    * Amihud illiquidity = |return| / dollar_volume.
    * Spread = adverse selection (= 2 |lambda| * avg_trade_size, capped) +
      remainder attributed to inventory / order-processing costs.
    """
    buy_flow = sum(o.qty for o in new_orders if o.side == Side.BUY)
    sell_flow = sum(o.qty for o in new_orders if o.side == Side.SELL)
    net_flow = buy_flow - sell_flow

    kyles_lambda: Optional[float] = None
    if net_flow != 0 and prev_mid is not None and curr_mid is not None:
        kyles_lambda = (curr_mid - prev_mid) / net_flow

    amihud: Optional[float] = None
    if (
        result.volume > 0
        and curr_mid is not None
        and prev_mid is not None
        and prev_mid > 0
    ):
        abs_ret = abs((curr_mid - prev_mid) / prev_mid)
        ref_px = result.clearing_price if result.clearing_price is not None else curr_mid
        dollar_vol = result.volume * ref_px
        if dollar_vol > 0:
            amihud = abs_ret / dollar_vol

    adverse: Optional[float] = None
    inventory: Optional[float] = None
    if (
        result.spread is not None
        and kyles_lambda is not None
        and result.num_trades > 0
    ):
        avg_trade = result.volume / result.num_trades
        adverse = min(2.0 * abs(kyles_lambda) * avg_trade, result.spread)
        inventory = result.spread - adverse

    return kyles_lambda, amihud, adverse, inventory


# ---------------------------------------------------------------------------
# ExtendedSimulation — orchestrates all three contributions
# ---------------------------------------------------------------------------

class ExtendedSimulation(BaseSimulation):
    """Sub-classes BaseSimulation to plumb in:

      * SupplyChainPulse → narrative on each agent each round
      * Microstructure metrics on each round's result
      * Wealth deque update each round
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
        shock_round: int = 10,
    ) -> None:
        super().__init__(
            agents=agents,
            num_rounds=num_rounds,
            initial_price=initial_price,
            redemption_value=redemption_value,
            interest_rate=interest_rate,
            dividend_model=dividend_model,
            dividend_frequency=dividend_frequency,
        )
        self.pulse = SupplyChainPulse()
        # The pulse module hard-codes the trigger tick at 10; honour the override
        # by mapping our `shock_round` argument onto that constant for clarity.
        # We don't mutate the pulse internals — instead we just gate by round here.
        self.shock_round = shock_round
        self.extended_results: list[ExtendedMarketResult] = []
        self.shock_states: list[Optional[SupplyChainState]] = []

    # ---- per-round narrative state ---------------------------------------

    def _supply_chain_state_for(self, round_no: int) -> Optional[SupplyChainState]:
        # The pulse is keyed off its own internal threshold tick (10);
        # we simply forward the current round and let it gate.
        # If a caller passes a non-default shock_round, we shift the input.
        adjusted_tick = round_no - (self.shock_round - 10)
        return self.pulse.get_state(adjusted_tick)

    def run(self, verbose: bool = True, state_extra_fn=None) -> list[MarketResult]:
        for round_no in range(1, self.num_rounds + 1):
            sc_state = self._supply_chain_state_for(round_no)
            self.shock_states.append(sc_state)
            for agent in self.agents:
                if isinstance(agent, _NarrativeMixin):
                    agent.set_supply_chain_state(sc_state)

            # Snapshot prev_mid BEFORE this round's orders go in (for Kyle's λ).
            prev_mid = self.book.mid_price()

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

            curr_mid = self.book.mid_price()
            kl, amihud, adv, inv = compute_microstructure_metrics(
                prev_mid, curr_mid, new_orders, result
            )
            ext = ExtendedMarketResult(
                base=result,
                kyles_lambda=kl,
                amihud_illiquidity=amihud,
                spread_adverse_selection=adv,
                spread_inventory_cost=inv,
                active_event=sc_state.event_name if sc_state else None,
                active_severity=sc_state.severity if sc_state else None,
            )
            self.extended_results.append(ext)

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

            # Mark wealth + push into per-agent wealth memory deque.
            mark = (
                result.clearing_price
                or self.book.mid_price()
                or self.initial_price
            )
            wealth_row = {}
            for a in self.agents:
                w = a.wealth(mark)
                wealth_row[a.agent_id] = w
                if isinstance(a, _WealthMemoryMixin):
                    a.record_wealth(w)
            self.wealth_log.append(wealth_row)

            if verbose:
                cp = f"{result.clearing_price:.2f}" if result.clearing_price else "  N/A"
                kl_s = f"{kl:.4f}" if kl is not None else "  N/A "
                am_s = f"{amihud:.2e}" if amihud is not None else "  N/A "
                event = sc_state.event_name if sc_state else "baseline"
                print(
                    f"R{round_no:>2} clr={cp:>6} vol={result.volume:>5.1f} "
                    f"trades={result.num_trades:>2} "
                    f"λ={kl_s:>8} amihud={am_s:>9} | {event}"
                )

        # Terminal redemption.
        for a in self.agents:
            a.receive_redemption(self.redemption_value)
        return self.results
