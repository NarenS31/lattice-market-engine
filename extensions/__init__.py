"""extensions/ — novel research contributions on top of Lopez-Lira (2025).

This package layers three additions onto a faithful re-implementation of the
upstream LLM-trading-sim market mechanism:

  1. Supply-chain causal narrative shocks (`supply_chain.py`)
  2. Microstructure metrics: Kyle's lambda, Amihud illiquidity, spread
     decomposition (added in `supply_chain_extensions.py`)
  3. Per-agent rolling wealth deque feeding the LLM prompt
     (added in `supply_chain_extensions.py`)
"""
