"""
supply_chain.py — structured supply chain shock system for the LOB simulation.

SUPPLY_CHAIN_EVENTS catalogues four real-world shock archetypes, each with:
  affected_sectors       — which industries bear the primary impact
  severity               — 0–1 scale (0 = negligible, 1 = systemic)
  narrative              — immediate shock description (one sentence)
  multi_sentence_narrative — narrative + chained second-order effects

SupplyChainPulse is a drop-in replacement for FactorPulse: it picks one event
randomly at tick 10 and exposes it through get_state() / to_context().
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

SUPPLY_CHAIN_EVENTS: dict[str, dict] = {
    "lithium_shortage": {
        "affected_sectors": ["EV manufacturers", "battery suppliers", "tech hardware"],
        "severity": 0.7,
        "narrative": (
            "A wildcat strike at the Atacama lithium mine has halted 18% of global "
            "lithium carbonate supply, forcing EV manufacturers to cut production guidance "
            "and battery suppliers to warn of margin compression."
        ),
        "multi_sentence_narrative": (
            "A wildcat strike at the Atacama lithium mine has halted 18% of global "
            "lithium carbonate supply, forcing EV manufacturers to cut production guidance "
            "and battery suppliers to warn of margin compression. "
            "This is expected to cascade into reduced EV sales forecasts and potential "
            "credit rating downgrades for affected manufacturers."
        ),
    },
    "semiconductor_shortage": {
        "affected_sectors": ["auto manufacturers", "consumer electronics", "defense contractors"],
        "severity": 0.6,
        "narrative": (
            "A fire at a TSMC fab in Taiwan has disrupted 3nm chip production, with lead "
            "times extending 8–12 weeks across automotive and consumer electronics supply chains."
        ),
        "multi_sentence_narrative": (
            "A fire at a TSMC fab in Taiwan has disrupted 3nm chip production, with lead "
            "times extending 8–12 weeks across automotive and consumer electronics supply chains. "
            "Extended lead times are expected to stall auto production schedules, delay consumer "
            "electronics product launches, and widen cost overruns for defense contractors "
            "operating on fixed-price contracts."
        ),
    },
    "energy_shock": {
        "affected_sectors": ["industrials", "airlines", "chemical manufacturers"],
        "severity": 0.8,
        "narrative": (
            "A geopolitical disruption in the Strait of Hormuz has spiked oil futures 12%, "
            "triggering margin warnings across energy-intensive industries and forcing airline "
            "capacity cuts."
        ),
        "multi_sentence_narrative": (
            "A geopolitical disruption in the Strait of Hormuz has spiked oil futures 12%, "
            "triggering margin warnings across energy-intensive industries and forcing airline "
            "capacity cuts. "
            "Prolonged elevated energy costs are expected to accelerate pass-through inflation "
            "in consumer goods, trigger a wave of airline route suspensions, and compress "
            "operating margins across chemical and industrial sectors through higher feedstock "
            "prices."
        ),
    },
    "credit_crunch": {
        "affected_sectors": ["financial sector", "real estate", "small caps"],
        "severity": 0.9,
        "narrative": (
            "Three regional banks have disclosed elevated commercial real estate exposure, "
            "triggering a broad credit tightening as counterparties reassess interbank "
            "lending terms."
        ),
        "multi_sentence_narrative": (
            "Three regional banks have disclosed elevated commercial real estate exposure, "
            "triggering a broad credit tightening as counterparties reassess interbank "
            "lending terms. "
            "Tighter credit conditions are expected to increase funding costs for real estate "
            "developers, restrict small-cap access to revolving credit facilities, and reduce "
            "overall market liquidity as banks shift toward shorter-duration and "
            "higher-grade collateral."
        ),
    },
}

_SHOCK_TICK = 10


@dataclass
class SupplyChainState:
    event_name: str
    affected_sectors: list
    severity: float
    narrative: str
    multi_sentence_narrative: str
    tick: int


class SupplyChainPulse:
    """
    Randomly selects one supply chain event at tick 10 and holds it for the
    remainder of the simulation.  Pre-shock ticks return None / empty context.
    """

    def __init__(self) -> None:
        self._active_key: str | None = None

    def get_state(self, tick: int) -> SupplyChainState | None:
        if tick < _SHOCK_TICK:
            return None
        if self._active_key is None:
            self._active_key = random.choice(list(SUPPLY_CHAIN_EVENTS.keys()))
        ev = SUPPLY_CHAIN_EVENTS[self._active_key]
        return SupplyChainState(
            event_name=self._active_key,
            affected_sectors=list(ev["affected_sectors"]),
            severity=ev["severity"],
            narrative=ev["narrative"],
            multi_sentence_narrative=ev["multi_sentence_narrative"],
            tick=tick,
        )

    def to_context(self, state: SupplyChainState | None) -> dict:
        """Return a JSON-serialisable context dict; empty dict for pre-shock ticks."""
        if state is None:
            return {}
        return asdict(state)
