# lattice-market-engine

A research codebase with two layers:

| Layer | Purpose | Built on |
| --- | --- | --- |
| **Top level** (`engine.py`, `agents.py`, `run_simulation.py`, …) | Original limit-order book matching engine, written from scratch. Price-time priority, partial fills, market and limit orders, JSON serialisation, plus an LLM-agent layer (Retail / Institutional / Quant / Market Maker) and a supply-chain factor pulse. | — |
| **`extensions/`** | Three novel research contributions, layered on a faithful adaptation of the LLM market in Lopez-Lira (2025). | [Lopez-Lira 2025](https://ssrn.com/abstract=5217340) — *Can Large Language Models Trade?* |

## Top level — original LOB

The root of the repo is a self-contained limit-order book and LLM-agent
simulation that I built from scratch. Run it with:

```
python3 run_simulation.py
```

Key files:

- `engine.py` — `Order`, `Trade`, `OrderBook`, `TickSummary`. Price-time
  priority matching with partial fills; per-tick microstructure metrics
  (Kyle's lambda, Amihud illiquidity, spread decomposition).
- `agents.py` — `RetailAgent`, `InstitutionalAgent`, `QuantAgent`,
  `MarketMakerAgent`, plus the `Simulation` driver. Each LLM agent calls
  the Anthropic API with a cached cognitive-archetype system prompt.
- `supply_chain.py` — the structured supply-chain shock catalogue
  (`lithium_shortage`, `semiconductor_shortage`, `energy_shock`,
  `credit_crunch`) used by the extensions and by `run_simulation.py`.
- `run_simulation.py`, `analysis.py`, `visualize.py`, `factor_pulse.py`,
  `test_engine.py` — drivers, post-run analytics, plots, tests.

## `extensions/` — novel research on top of Lopez-Lira (2025)

The `extensions/` package re-creates the core market mechanism from
[`alejandroll10/llm_trading_sim`](https://github.com/alejandroll10/llm_trading_sim)
in a single readable file (persistent LOB, stochastic dividends, interest,
terminal redemption, fundamental-value calibration, and the canonical
agent archetypes from the paper) — and then adds three contributions
that aren't in the original:

1. **Supply-chain causal narrative shocks.** At round 10, one of four
   real-world archetype events fires (lithium / semiconductor / energy /
   credit) and a *multi-sentence* narrative chaining the shock to its
   second-order effects is injected into every agent's user prompt as
   plain English. The LLM observes news as language, not as a numeric
   severity score.

2. **Microstructure metrics in the per-round summary.** Each round we
   attach Kyle's lambda, Amihud illiquidity, and a spread decomposition
   (adverse-selection vs. inventory / order-processing components) to
   the standard market result.

3. **Per-agent rolling P&L memory.** Each agent gets a `deque[5]` of
   recent wealth marks; the prompt builder reads it back so the LLM
   observes its own short-run P&L trajectory each round.

### Layout

```
extensions/
  __init__.py
  base_market.py                # Lopez-Lira market mechanism (single file)
  supply_chain_extensions.py    # the three contributions, layered as mixins
  run_extended.py               # driver — runs and prints results
```

### Run it

```
python3 -m extensions.run_extended
```

The driver prints, for every round: the engine summary line (clearing
price, volume, microstructure metrics, and the active supply-chain
event), each agent's natural-language reasoning string, and a
side-by-side panel showing how each archetype responded to the shock
narrative.

### Citation

This code accompanies / extends:

```bibtex
@article{lopez2025llm,
  title  = {Can Large Language Models Trade? Testing Financial Theories
            with LLM Agents in Market Simulations},
  author = {Lopez-Lira, Alejandro},
  year   = {2025},
  url    = {https://ssrn.com/abstract=5217340},
  doi    = {10.2139/ssrn.5217340}
}
```
