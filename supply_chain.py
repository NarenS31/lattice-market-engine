"""
supply_chain.py — structured supply chain shock system for the LOB simulation.

SUPPLY_CHAIN_EVENTS catalogues ten real-world shock archetypes, each with:
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
    # ---------------------------------------------------------------------
    # Families 5-10, added to take the 2x2 design from n=4 to n=10 families.
    #
    # Same authoring rules as the first four: a named real operator or place, a
    # documented event a model could date, and a magnitude taken from the event
    # itself. Six distinct causal classes were chosen deliberately — port
    # closure, export restriction, pipeline outage, flood, rail embargo, dam
    # failure — so the added families widen the causal space rather than
    # re-sampling the four already present. Severities span 0.55-0.85.
    #
    # Diacritics are avoided (Corrego, Feijao): check_shock_balance.py tokenises
    # on [A-Za-z], so an accented character silently splits one word into two
    # and shows up as a spurious length imbalance against the fictional twin.
    # ---------------------------------------------------------------------
    "port_closure": {
        "affected_sectors": ["consumer goods importers", "electronics assemblers", "freight forwarders"],
        "severity": 0.65,
        "narrative": (
            "A COVID outbreak at the Meishan terminal of the Port of Ningbo-Zhoushan "
            "has idled 28% of the port's container throughput, forcing consumer goods "
            "importers to rebook freight and electronics assemblers to warn of delivery slippage."
        ),
        "multi_sentence_narrative": (
            "A COVID outbreak at the Meishan terminal of the Port of Ningbo-Zhoushan "
            "has idled 28% of the port's container throughput, forcing consumer goods "
            "importers to rebook freight and electronics assemblers to warn of delivery slippage. "
            "Idled berth capacity is expected to lengthen ocean transit times for consumer "
            "goods importers, push electronics assemblers into costly air freight substitution, "
            "and inflate demurrage charges for freight forwarders operating on fixed-rate contracts."
        ),
    },
    "wheat_export_ban": {
        "affected_sectors": ["food processors", "livestock producers", "commodity traders"],
        "severity": 0.55,
        "narrative": (
            "India's ban on wheat exports has withdrawn 10 million tonnes from global "
            "grain supply, forcing food processors to reprice contracts and livestock "
            "producers to warn of feed cost inflation."
        ),
        "multi_sentence_narrative": (
            "India's ban on wheat exports has withdrawn 10 million tonnes from global "
            "grain supply, forcing food processors to reprice contracts and livestock "
            "producers to warn of feed cost inflation. "
            "Restricted export volumes are expected to raise input costs for food processors, "
            "compress margins for livestock producers, and widen basis risk for commodity "
            "traders holding forward positions."
        ),
    },
    "pipeline_outage": {
        "affected_sectors": ["fuel distributors", "petrochemical refiners", "logistics operators"],
        "severity": 0.75,
        "narrative": (
            "A ransomware attack on the Colonial Pipeline has halted 2.5 million barrels "
            "per day of refined fuel flow, forcing fuel distributors to invoke allocation "
            "and petrochemical refiners to cut run rates."
        ),
        "multi_sentence_narrative": (
            "A ransomware attack on the Colonial Pipeline has halted 2.5 million barrels "
            "per day of refined fuel flow, forcing fuel distributors to invoke allocation "
            "and petrochemical refiners to cut run rates. "
            "Suspended pipeline flow is expected to strand product inventories for fuel "
            "distributors, force run-rate cuts at petrochemical refiners, and escalate spot "
            "haulage rates for logistics operators rerouting by tanker truck."
        ),
    },
    "flood_inundation": {
        "affected_sectors": ["PC manufacturers", "storage vendors", "contract assemblers"],
        "severity": 0.7,
        "narrative": (
            "Monsoon flooding across Thailand's Ayutthaya industrial estates has submerged "
            "plants producing 30% of global hard drive output, forcing PC manufacturers to "
            "ration allocations and storage vendors to suspend price guidance."
        ),
        "multi_sentence_narrative": (
            "Monsoon flooding across Thailand's Ayutthaya industrial estates has submerged "
            "plants producing 30% of global hard drive output, forcing PC manufacturers to "
            "ration allocations and storage vendors to suspend price guidance. "
            "Submerged production capacity is expected to delay build schedules for PC "
            "manufacturers, extend order backlogs for storage vendors, and idle assembly "
            "lines at contract assemblers dependent on single-source components."
        ),
    },
    "rail_embargo": {
        "affected_sectors": ["grain shippers", "fertilizer producers", "auto parts suppliers"],
        "severity": 0.6,
        "narrative": (
            "A labor dispute across US Class I freight railroads has triggered an embargo "
            "cutting 15% of weekly carload volume, forcing grain shippers to divert to truck "
            "and fertilizer producers to halt outbound loading."
        ),
        "multi_sentence_narrative": (
            "A labor dispute across US Class I freight railroads has triggered an embargo "
            "cutting 15% of weekly carload volume, forcing grain shippers to divert to truck "
            "and fertilizer producers to halt outbound loading. "
            "Reduced carload capacity is expected to strand harvest volumes for grain shippers, "
            "force production curtailment at fertilizer producers, and interrupt just-in-time "
            "deliveries for auto parts suppliers running minimal buffer stock."
        ),
    },
    "dam_failure": {
        "affected_sectors": ["steel producers", "ore traders", "mining equipment suppliers"],
        "severity": 0.85,
        "narrative": (
            "A tailings dam collapse at Vale's Corrego do Feijao mine has removed 40 million "
            "tonnes of annual iron ore capacity, forcing steel producers to secure alternate "
            "offtake and ore traders to reprice term contracts."
        ),
        "multi_sentence_narrative": (
            "A tailings dam collapse at Vale's Corrego do Feijao mine has removed 40 million "
            "tonnes of annual iron ore capacity, forcing steel producers to secure alternate "
            "offtake and ore traders to reprice term contracts. "
            "Lost ore capacity is expected to raise input costs for steel producers, tighten "
            "seaborne availability for ore traders, and defer capital orders at mining "
            "equipment suppliers exposed to the affected operator."
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
