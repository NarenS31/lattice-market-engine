"""
factor_pulse.py — macro environment layer for the LOB simulation.

Now delegates to supply_chain.SupplyChainPulse.  The legacy FactorPulse class
and MacroState dataclass are kept for backward compatibility with any code that
imports them directly; FactorPulse is aliased to SupplyChainPulse so existing
call sites receive supply-chain events without modification.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from supply_chain import SUPPLY_CHAIN_EVENTS, SupplyChainPulse, SupplyChainState  # noqa: F401


# ---------------------------------------------------------------------------
# Legacy classes (retained for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class MacroState:
    fed_rate: float
    inflation: float
    vix: float
    sentiment: float   # bounded [-1, 1]; positive = risk-on
    tick: int


class _LegacyFactorPulse:
    """Original numeric macro-factor pulse — kept but no longer the default."""

    _BASELINE: dict = dict(fed_rate=5.25, inflation=3.1, vix=18.0, sentiment=0.1)

    _SHOCKS: list[dict] = [
        {"fed_rate": 5.75, "vix": 28.0},
        {"sentiment": -0.8, "vix": 35.0},
        {"vix": 45.0, "sentiment": -0.5},
    ]

    def __init__(self) -> None:
        self._shock: dict | None = None

    def get_state(self, tick: int) -> MacroState:
        params = dict(self._BASELINE)
        if tick >= 10:
            if self._shock is None:
                self._shock = random.choice(self._SHOCKS)
            params.update(self._shock)
        return MacroState(**params, tick=tick)

    def to_context(self, state: MacroState) -> dict:
        return asdict(state)


# SupplyChainPulse is the active implementation; FactorPulse is an alias so
# existing `from factor_pulse import FactorPulse` imports work unchanged.
FactorPulse = SupplyChainPulse
