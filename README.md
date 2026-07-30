# lattice-market-engine

A research codebase with two layers:

| Layer | Purpose | Built on |
| --- | --- | --- |
| **Top level** (`engine.py`, `agents.py`, `run_simulation.py`, …) | Original limit-order book matching engine, written from scratch. Price-time priority, partial fills, market and limit orders, JSON serialisation, plus an LLM-agent layer (Retail / Institutional / Quant / Market Maker) and a supply-chain factor pulse. | — |
| **`extensions/`** | Novel research contributions layered on a faithful adaptation of the LLM market in Lopez-Lira (2025). | [Lopez-Lira 2025](https://ssrn.com/abstract=5217340) — *Can Large Language Models Trade?* |

## Why this project is impressive

This repository is built to demonstrate both technical originality and research impact.

- The core **limit-order book engine is fully original** and implemented from scratch in `engine.py`.
- The simulation integrates **LLM-based trading agents** with a market microstructure engine.
- The project adds **new research contributions** on top of the LLM trading market literature.
- Everything is end-to-end reproducible with `python3 run_simulation.py`.

## Research contributions

1. **Original LOB engine.** `engine.py` implements a complete limit-order book with:
   - price-time priority matching,
   - partial fills,
   - market and limit orders,
   - persistent order history,
   - trade logging,
   - tick-level price/volume summaries.

2. **Supply-chain narrative shocks.** At a configurable impact tick, one of four realistic supply-chain events fires and a multi-sentence natural-language narrative is injected into every agent prompt.
   - This makes the LLMs respond to news text instead of abstract numeric shock vectors.
   - It models how macro shocks propagate through supply chains and affect different market archetypes.

3. **Market microstructure metrics.** The simulation computes per-round liquidity metrics such as:
   - Kyle’s lambda,
   - Amihud illiquidity,
   - spread decomposition into adverse-selection and inventory/order-processing components.

4. **Per-agent prompt memory.** Each agent stores a rolling history of recent wealth and includes that history in its prompt.
   - This lets the LLM observe short-run P&L trends and adapt behavior across rounds.

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
that are not present in the original framework.

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
