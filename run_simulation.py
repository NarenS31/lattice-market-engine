

#!/usr/bin/env python3
"""
run_simulation.py

Creates 5 RetailAgents, 2 InstitutionalAgents, 2 QuantAgents.
Seeds the book at fair value 100, runs 25 ticks (supply-chain shock fires at
tick 10), then prints:
  - per-tick wealth table with the active supply chain event
  - a banner when the shock first fires showing the event name, affected
    sectors, severity, and full multi-sentence narrative
  - final wealth + P&L summary
"""
import argparse

import llm_backends
from engine import OrderBook
from agents import (
    Simulation,
    RetailAgent,
    InstitutionalAgent,
    QuantAgent,
    MarketMakerAgent,
)
from supply_chain import SupplyChainPulse

INITIAL_CASH = {
    "Retail": 10_000.0,
    "Inst":   100_000.0,
    "Quant":  50_000.0,
}

_SHOCK_TICK = 10   # first tick at which the supply chain event activates


def print_research_overview() -> None:
    banner = "RESEARCH CONTRIBUTION SUMMARY"
    line = "=" * len(banner)
    print(line)
    print(banner)
    print(line)
    print("- Fully original limit-order book engine implemented from scratch in engine.py")
    print("- LLM-market simulation with Retail, Institutional, Quant, and Market Maker archetypes")
    print("- Natural-language supply-chain narrative shocks injected into each agent prompt")
    print("- Per-round microstructure analytics: Kyle's lambda, Amihud illiquidity, spread decomposition")
    print("- Rolling agent P&L memory for realistic short-run adaptation")
    print(line + "\n")


def _event_cols(macro: dict) -> tuple[str, str]:
    """Return (event_label, severity_str) formatted for the per-tick table."""
    event = macro.get("event_name", "")
    if not event:
        return "BASELINE        ", "  — "
    label = event.upper().replace("_", "-")
    label = f"{label:<16}"[:16]
    return label, f"{macro.get('severity', 0.0):.1f}"


def _print_shock_banner(macro: dict) -> None:
    """Print a formatted banner the first time the supply chain shock fires."""
    event     = macro.get("event_name", "unknown").replace("_", " ").upper()
    severity  = macro.get("severity", 0.0)
    sectors   = ", ".join(macro.get("affected_sectors", []))
    narrative = macro.get("multi_sentence_narrative", "")

    width = 70
    print("\n" + "!" * width)
    print(f"  SUPPLY CHAIN SHOCK ACTIVATED — TICK {_SHOCK_TICK}")
    print("!" * width)
    print(f"  Event    : {event}")
    print(f"  Severity : {severity:.1f} / 1.0")
    print(f"  Sectors  : {sectors}")
    print(f"  Narrative: {narrative[:width - 13]}")
    if len(narrative) > width - 13:
        # wrap remaining narrative text at word boundaries
        remainder = narrative[width - 13:]
        words = remainder.split()
        line, lines = [], []
        for w in words:
            if len(" ".join(line + [w])) > width - 13:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for ln in lines:
            print(f"             {ln}")
    print("!" * width + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the LOB + LLM-agent simulation with a supply-chain shock."
    )
    llm_backends.add_backend_cli_args(ap)
    llm_backends.apply_cli_args(ap.parse_args())

    book  = OrderBook(symbol="SIM")
    pulse = SupplyChainPulse()

    agents = (
        [RetailAgent(cash=INITIAL_CASH["Retail"], name=f"Retail-{i+1}") for i in range(5)]
        + [InstitutionalAgent(cash=INITIAL_CASH["Inst"], name=f"Inst-{i+1}") for i in range(2)]
        + [QuantAgent(cash=INITIAL_CASH["Quant"], name=f"Quant-{i+1}") for i in range(2)]
        + [MarketMakerAgent(name="MM-1")]
    )

    print_research_overview()
    sim = Simulation(book, agents)
    sim.seed_book(fair_value=100.0)
    summaries = sim.run(n_ticks=25, factor_pulse=pulse, verbose=True)

    # ------------------------------------------------------------------ #
    # Per-tick wealth table with supply chain state
    # ------------------------------------------------------------------ #
    COL   = 14
    names = [a.name for a in agents]

    event_header  = f"{'Event':<16}  {'Sev':>3}"
    agent_header  = "  ".join(f"{n:>{COL}}" for n in names)
    header  = f"{'Tick':>4}  {event_header}  {agent_header}"
    divider = "-" * len(header)

    print(f"\n{'Per-Tick Agent Wealth with Supply Chain State':^{len(header)}}")
    print(divider)
    print(header)
    print(divider)

    shock_printed = False
    for tick_idx, (row, macro) in enumerate(zip(sim.wealth_log, sim.macro_log)):
        # Print the shock banner exactly once when the event first appears.
        if not shock_printed and macro.get("event_name"):
            _print_shock_banner(macro)
            shock_printed = True

        event_label, sev_str = _event_cols(macro)
        wealth_str = "  ".join(f"{w:>{COL}.2f}" for w in row)
        print(f"{tick_idx + 1:>4}  {event_label}  {sev_str:>3}  {wealth_str}")

    print(divider)

    # ------------------------------------------------------------------ #
    # Tick summary (clearing prices)
    # ------------------------------------------------------------------ #
    print(f"\n{'Tick Summary (engine)':}")
    print("-" * 50)
    for s in summaries:
        cp = f"{s.clearing_price:.2f}" if s.clearing_price else "  N/A"
        print(f"  tick={s.tick:>3}  clr={cp:>7}  vol={s.volume:>7.2f}  trades={s.num_trades:>3}")

    # ------------------------------------------------------------------ #
    # Final wealth + P&L
    # ------------------------------------------------------------------ #
    print(f"\n{'=== Final Wealth ==='}")
    final = sim.wealth_log[-1]
    for agent, w in zip(agents, final):
        archetype = agent.name.split("-")[0]
        initial   = INITIAL_CASH[archetype]
        pnl       = w - initial
        pct       = pnl / initial * 100
        print(
            f"  {agent.name:<12}  "
            f"initial={initial:>10.2f}  "
            f"final={w:>12.2f}  "
            f"P&L={pnl:>+10.2f}  ({pct:>+6.2f}%)"
        )

    # ------------------------------------------------------------------ #
    # Active supply chain event summary (post-run)
    # ------------------------------------------------------------------ #
    active = next((m for m in sim.macro_log if m.get("event_name")), None)
    if active:
        print(f"\n{'=== Active Supply Chain Event ==='}")
        print(f"  Event    : {active['event_name'].replace('_', ' ').upper()}")
        print(f"  Severity : {active['severity']:.1f}")
        print(f"  Sectors  : {', '.join(active['affected_sectors'])}")
        print(f"  Narrative: {active['multi_sentence_narrative']}")


if __name__ == "__main__":
    main()
