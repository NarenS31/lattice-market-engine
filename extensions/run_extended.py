"""
extensions/run_extended.py

Drives a full simulation built on the Lopez-Lira (2025) market mechanism
(see `base_market.py`) with all three research extensions enabled:

  1. Supply-chain causal narrative shock at round 10.
  2. Per-round microstructure metrics (Kyle's lambda, Amihud, spread split).
  3. Each agent's prompt carries its rolling 5-round wealth memory.

Invocation:

    # From the project root:
    python -m extensions.run_extended

The script prints, for every round:
  * the engine summary line (clearing price, volume, microstructure metrics,
    and the active supply-chain event),
  * each agent's natural-language reasoning string,
and after the run prints the final wealth table plus a side-by-side
"how each archetype responded to the shock narrative" panel that
demonstrates the three extensions are working.
"""
from __future__ import annotations

import os
import sys

# When invoked as `python extensions/run_extended.py` (i.e. NOT via -m), we
# need the project root on sys.path so that `extensions.*` and
# `supply_chain.*` resolve. This is a one-line ergonomic shim.
if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    __package__ = "extensions"

from extensions.base_market import DividendModel  # noqa: E402
from extensions.supply_chain_extensions import (  # noqa: E402
    ExtendedContrarian,
    ExtendedMarketMaker,
    ExtendedMomentumTrader,
    ExtendedSimulation,
    ExtendedValueInvestor,
)


def _build_agents() -> list:
    """Two of each archetype, with the cash/share endowments from the paper."""
    initial_cash = 1000.0
    initial_shares = 10
    agents = []
    for i in range(2):
        agents.append(ExtendedValueInvestor(
            agent_id=f"Value-{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
        ))
    for i in range(2):
        agents.append(ExtendedMomentumTrader(
            agent_id=f"Momentum-{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
        ))
    for i in range(2):
        agents.append(ExtendedMarketMaker(
            agent_id=f"MarketMaker-{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
        ))
    for i in range(2):
        agents.append(ExtendedContrarian(
            agent_id=f"Contrarian-{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
        ))
    return agents


def _print_reasoning(reasoning_log: list[list[str]], agent_ids: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Per-round agent reasoning")
    print("=" * 80)
    for round_idx, row in enumerate(reasoning_log, start=1):
        print(f"\nRound {round_idx}:")
        for r in row:
            print(f"  {r}")


def _print_microstructure_table(sim: ExtendedSimulation) -> None:
    print("\n" + "=" * 80)
    print("Microstructure metrics")
    print("=" * 80)
    print(f"{'Round':>5}  {'Event':<22}  {'Sev':>4}  {'λ':>8}  "
          f"{'Amihud':>10}  {'AdvSel':>7}  {'InvCost':>8}")
    for r in sim.extended_results:
        ev = r.active_event or "baseline"
        sev = f"{r.active_severity:.1f}" if r.active_severity is not None else " — "
        kl = f"{r.kyles_lambda:.4f}" if r.kyles_lambda is not None else " N/A"
        am = f"{r.amihud_illiquidity:.2e}" if r.amihud_illiquidity is not None else "  N/A"
        adv = f"{r.spread_adverse_selection:.3f}" if r.spread_adverse_selection is not None else " N/A"
        inv = f"{r.spread_inventory_cost:.3f}" if r.spread_inventory_cost is not None else " N/A"
        print(f"{r.round_number:>5}  {ev:<22}  {sev:>4}  {kl:>8}  "
              f"{am:>10}  {adv:>7}  {inv:>8}")


def _print_shock_response(sim: ExtendedSimulation) -> None:
    """Show the round-by-round buy/sell action of each archetype before vs.
    after the shock fires, demonstrating that agents respond to the narrative."""
    if not any(s for s in sim.shock_states):
        print("\n(No shock fired during the simulation — nothing to compare.)")
        return

    first_shock = next(i for i, s in enumerate(sim.shock_states, start=1) if s)
    active = next(s for s in sim.shock_states if s)

    print("\n" + "=" * 80)
    print(f"Supply-chain shock fired at round {first_shock}: "
          f"{active.event_name.upper().replace('_', ' ')} "
          f"(severity {active.severity:.1f})")
    print("Affected sectors:", ", ".join(active.affected_sectors))
    print("Narrative:", active.multi_sentence_narrative)
    print("=" * 80)

    print("\nAction direction by archetype, by round (B=buy, S=sell, .=hold):")
    archetypes: dict[str, list[str]] = {}
    for round_idx, row in enumerate(sim.reasoning_log, start=1):
        for r in row:
            agent_id = r.split(":", 1)[0].strip()
            archetype = agent_id.split("-")[0]
            archetypes.setdefault(archetype, [])
        # collect one symbol per agent in this round
        per_arch: dict[str, list[str]] = {k: [] for k in archetypes}
        for r in row:
            agent_id = r.split(":", 1)[0].strip()
            archetype = agent_id.split("-")[0]
            tag = "."
            up = r.upper()
            if " BUY " in up:
                tag = "B"
            elif " SELL " in up:
                tag = "S"
            per_arch[archetype].append(tag)
        for arch, tags in per_arch.items():
            archetypes[arch].append("/".join(tags))

    header = "  Round  " + "  ".join(f"{a:<14}" for a in archetypes)
    print(header)
    for round_idx in range(len(sim.reasoning_log)):
        marker = " * " if round_idx + 1 == first_shock else "   "
        cells = "  ".join(f"{archetypes[a][round_idx]:<14}" for a in archetypes)
        print(f"  {round_idx + 1:>3} {marker} {cells}")

    print("\n  '*' marks the round when the shock first fires.")
    print("  Direction divergence between archetypes after the shock is the")
    print("  signature that LLM agents are reacting to the narrative differently.")


def _print_final_wealth(sim: ExtendedSimulation) -> None:
    print("\n" + "=" * 80)
    print("Final wealth (after terminal redemption)")
    print("=" * 80)
    for a in sim.agents:
        w = a.wealth(sim.redemption_value)
        print(f"  {a.agent_id:<14}  cash=${a.cash:>10.2f}  shares={a.shares:>4}  "
              f"div=${a.dividend_cash:>7.2f}  wealth=${w:>10.2f}")


def main() -> None:
    agents = _build_agents()
    sim = ExtendedSimulation(
        agents=agents,
        num_rounds=15,
        initial_price=28.0,
        redemption_value=28.0,
        interest_rate=0.05,
        dividend_model=DividendModel(high=1.6, low=1.2, prob_high=0.5),
        shock_round=10,
    )

    print("\n" + "#" * 80)
    print("# Lattice-market-engine — extensions/run_extended.py")
    print("# Lopez-Lira (2025) market mechanism + 3 research extensions")
    print("#" * 80)

    sim.run(verbose=True)

    _print_reasoning(sim.reasoning_log, [a.agent_id for a in sim.agents])
    _print_microstructure_table(sim)
    _print_shock_response(sim)
    _print_final_wealth(sim)


if __name__ == "__main__":
    main()
