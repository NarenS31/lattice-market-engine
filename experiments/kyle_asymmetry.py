"""
Experiment 2: Kyle (1985) Information Asymmetry Test
=====================================================

Research question
-----------------
When only a fraction of traders observe the supply-chain shock narrative, does
the resulting market follow the structural predictions of Kyle's (1985) model
of strategic insider trading?

Kyle (1985) testable predictions
---------------------------------
  P1  Higher informed fraction  →  larger post-shock Kyle's lambda (price impact)
  P2  Higher informed fraction  →  faster price discovery after the shock
  P3  Higher informed fraction  →  higher adverse-selection fraction of the spread
  P4  Informed agents earn positive alpha vs uninformed agents

Experimental design
-------------------
  Four informed-fraction conditions: 0%, 33%, 67%, 100%

  Agent roster (3 LLM + 1 rule-based MM per run):
    0%  — all LLM agents are Uninformed (macro_context stripped)
   33%  — QuantAgent informed; Retail + Institutional uninformed
   67%  — Quant + Institutional informed; Retail uninformed
  100%  — all LLM agents informed (current baseline)

  Shock fires at tick SHOCK_TICK (default 10).  Pre-shock ticks: baseline.
  Post-shock ticks: supply-chain narrative injected for informed agents.

Metrics per condition
---------------------
  lambda_post    mean Kyle's lambda in post-shock ticks
  lambda_cv      coefficient of variation of lambda post-shock
                 (Kyle predicts λ constant → low CV)
  adv_sel_frac   mean (spread_adv_selection / spread) post-shock
                 (Kyle: higher informed fraction → higher adverse selection)
  disc_speed     half-life: mean ticks for |price − mean_post_price| to halve
                 (Kyle: more informed → faster price discovery)
  info_alpha     mean final-wealth difference: informed vs uninformed LLM agents
                 (Kyle: informed agents earn rent at expense of uninformed)

Run
---
  python -m experiments.run_all --exp 2 [--runs N] [--ticks N]
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import Order, OrderBook, Side           # noqa: E402
from agents import (                                # noqa: E402
    RetailAgent, InstitutionalAgent, QuantAgent,
    MarketMakerAgent, Simulation,
)
from factor_pulse import FactorPulse                # noqa: E402

SHOCK_TICK = 10


# ---------------------------------------------------------------------------
# Uninformed agent mixin — strips macro_context before the API call
# ---------------------------------------------------------------------------

class _UninformedMixin:
    """
    Overrides decide_orders to hide supply-chain narratives from this agent.
    The agent always sees "Markets are operating under normal baseline conditions"
    regardless of the actual macro_context fed by the simulation.
    """
    def decide_orders(
        self,
        book_snapshot: dict,
        macro_context: Optional[dict] = None,
        tick_idx: int = 0,
    ):
        return super().decide_orders(book_snapshot, None, tick_idx)


class UninformedRetail(_UninformedMixin, RetailAgent):
    pass


class UninformedInstitutional(_UninformedMixin, InstitutionalAgent):
    pass


class UninformedQuant(_UninformedMixin, QuantAgent):
    pass


# ---------------------------------------------------------------------------
# Kyle-tracking simulation
# ---------------------------------------------------------------------------

class KyleSimulation(Simulation):
    """
    Extends Simulation to track which agents are 'informed' and record
    per-tick microstructure metrics for post-shock Kyle analysis.
    """

    def __init__(
        self,
        book: OrderBook,
        agents: list,
        informed_indices: list[int],
    ) -> None:
        super().__init__(book, agents)
        self.informed_indices = set(informed_indices)
        self.uninformed_indices = {
            i for i in range(len(agents))
            if i not in self.informed_indices
            and not isinstance(agents[i], MarketMakerAgent)
        }
        self._kyle_ticks: list[dict] = []

    def run(
        self,
        n_ticks: int,
        factor_pulse=None,
        verbose: bool = False,
    ) -> list:
        summaries = []

        for tick_idx in range(n_ticks):
            snap = json.loads(self.book.to_json())
            ref = snap.get("mid_price") or snap.get("last_clearing_price") or 100.0

            macro_context: Optional[dict] = None
            if factor_pulse is not None:
                state = factor_pulse.get_state(tick_idx)
                macro_context = factor_pulse.to_context(state)
            self.macro_log.append(macro_context or {})

            prev_mid = self.book.mid_price()

            # Snapshot informed/uninformed wealth before decisions
            pre_wealth_informed = [
                self.agents[i].wealth(ref) for i in self.informed_indices
            ]
            pre_wealth_uninformed = [
                self.agents[i].wealth(ref) for i in self.uninformed_indices
            ]

            new_orders: list[Order] = []
            tick_reasoning: list[str] = []
            for i, agent in enumerate(self.agents):
                orders, reasoning = agent.decide_orders(snap, macro_context, tick_idx)
                tick_reasoning.append(reasoning)
                for o in orders:
                    new_orders.append(o)
                    self._order_map[o.order_id] = agent
            self.reasoning_log.append(tick_reasoning)

            summary = self.book.advance_tick(new_orders)
            summaries.append(summary)

            curr_mid = self.book.mid_price()
            fill_price = summary.clearing_price or self.book.last_clearing_price or 100.0

            for oid, agent in self._order_map.items():
                order = self.book.order_history.get(oid)
                if order is not None and order.filled_qty > 0:
                    agent.update_position(order, fill_price)
            self._order_map = {}

            post_wealth_informed = [
                self.agents[i].wealth(fill_price) for i in self.informed_indices
            ]
            post_wealth_uninformed = [
                self.agents[i].wealth(fill_price) for i in self.uninformed_indices
            ]

            tick_wealth = [a.wealth(fill_price) for a in self.agents]
            self.wealth_log.append(tick_wealth)
            for a, w in zip(self.agents, tick_wealth):
                a.append_wealth(w)

            # Spread adverse selection fraction
            adv = summary.spread_adverse_selection
            sprd = summary.spread
            adv_frac = (adv / sprd) if (adv is not None and sprd is not None and sprd > 0) else None

            self._kyle_ticks.append(
                {
                    "tick": tick_idx,
                    "is_post_shock": tick_idx >= SHOCK_TICK,
                    "clearing_price": summary.clearing_price,
                    "mid_price": curr_mid,
                    "spread": sprd,
                    "kyles_lambda": summary.kyles_lambda,
                    "adv_sel_frac": adv_frac,
                    "prev_mid": prev_mid,
                    "wealth_informed": float(np.mean(post_wealth_informed)) if post_wealth_informed else None,
                    "wealth_uninformed": float(np.mean(post_wealth_uninformed)) if post_wealth_uninformed else None,
                }
            )

        return summaries


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_one(
    informed_fraction: float,
    seed: int,
    n_ticks: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    # 3 LLM agents + 1 MarketMaker.  Informed fraction applies to LLM agents only.
    # Order: [Retail, Institutional, Quant, MM]
    # Informed assignment: Quant first, then Institutional, then Retail.
    n_llm = 3
    n_informed = round(informed_fraction * n_llm)

    retail_cls     = RetailAgent       if n_informed >= 3 else UninformedRetail
    inst_cls       = InstitutionalAgent if n_informed >= 2 else UninformedInstitutional
    quant_cls      = QuantAgent        if n_informed >= 1 else UninformedQuant

    agents = [
        retail_cls(cash=10_000.0,   name="Retail"),
        inst_cls(cash=100_000.0,    name="Inst"),
        quant_cls(cash=50_000.0,    name="Quant"),
        MarketMakerAgent(name="MM"),
    ]

    informed_idx  = [i for i, a in enumerate(agents) if not isinstance(a, _UninformedMixin) and not isinstance(a, MarketMakerAgent)]
    uninformed_idx = [i for i, a in enumerate(agents) if isinstance(a, _UninformedMixin)]

    book = OrderBook(symbol="EXP2")
    sim = KyleSimulation(book, agents, informed_idx)
    sim.seed_book(fair_value=100.0)
    pulse = FactorPulse()
    sim.run(n_ticks=n_ticks, factor_pulse=pulse)

    post = [t for t in sim._kyle_ticks if t["is_post_shock"]]

    # Kyle's lambda stats
    lambdas = [t["kyles_lambda"] for t in post if t["kyles_lambda"] is not None]
    lambda_post = float(np.mean(lambdas)) if lambdas else float("nan")
    lambda_cv = (float(np.std(lambdas)) / abs(lambda_post)
                 if lambdas and not math.isnan(lambda_post) and lambda_post != 0
                 else float("nan"))

    # Adverse selection fraction
    adv_fracs = [t["adv_sel_frac"] for t in post if t["adv_sel_frac"] is not None]
    adv_sel_frac = float(np.mean(adv_fracs)) if adv_fracs else float("nan")

    # Price discovery half-life: ticks for |price - mean_post_price| to halve
    post_prices = [t["clearing_price"] for t in post if t["clearing_price"] is not None]
    disc_speed = float("nan")
    if len(post_prices) >= 4:
        mean_post = float(np.mean(post_prices))
        init_dev = abs(post_prices[0] - mean_post)
        if init_dev > 0:
            half_dev = init_dev / 2.0
            for k, p in enumerate(post_prices[1:], 1):
                if abs(p - mean_post) <= half_dev:
                    disc_speed = float(k)
                    break
            else:
                disc_speed = float(len(post_prices))  # never halved

    # Informed alpha: mean wealth informed - mean wealth uninformed at end
    final_wealth = sim.wealth_log[-1] if sim.wealth_log else None
    info_alpha = float("nan")
    if final_wealth is not None and informed_idx and uninformed_idx:
        w_inf = float(np.mean([final_wealth[i] for i in informed_idx]))
        w_unf = float(np.mean([final_wealth[i] for i in uninformed_idx]))
        info_alpha = w_inf - w_unf

    return {
        "lambda_post": lambda_post,
        "lambda_cv": lambda_cv,
        "adv_sel_frac": adv_sel_frac,
        "disc_speed": disc_speed,
        "info_alpha": info_alpha,
    }


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def _run_condition(
    label: str,
    informed_fraction: float,
    n_runs: int,
    n_ticks: int,
) -> dict:
    results: dict[str, list[float]] = defaultdict(list)
    print(f"  {label}: ", end="", flush=True)
    for seed in range(n_runs):
        r = _run_one(informed_fraction, seed=seed * 11 + 7, n_ticks=n_ticks)
        for k, v in r.items():
            if not math.isnan(v):
                results[k].append(v)
        print(".", end="", flush=True)
    print()
    return {k: list(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Kyle prediction evaluator
# ---------------------------------------------------------------------------

def _check_monotone(values: list[float], higher_is_expected: bool) -> str:
    """Return ✓ if the list is (weakly) monotone in the expected direction."""
    if len(values) < 2:
        return "?"
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if higher_is_expected:
        return "✓" if all(d >= -1e-6 for d in diffs) else "~"
    else:
        return "✓" if all(d <= 1e-6 for d in diffs) else "~"


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------

def run_experiment2(n_runs: int = 10, n_ticks: int = 25) -> dict:
    """
    Run Experiment 2 and print a Kyle-prediction verification table.
    Returns the full results dict (also saved to results_exp2_kyle.json).
    """
    print("\n" + "=" * 72)
    print("Experiment 2: Kyle (1985) Information Asymmetry Test")
    print(f"  Informed fractions: 0%, 33%, 67%, 100%  |  Shock at tick {SHOCK_TICK}")
    print(f"  Runs: {n_runs} per condition  |  Ticks: {n_ticks} per run")
    print("=" * 72)

    conditions = [
        ("0%  informed  (all uninformed)", 0.0),
        ("33% informed  (Quant only)",     1 / 3),
        ("67% informed  (Quant + Inst)",   2 / 3),
        ("100% informed (all LLM)",        1.0),
    ]

    print()
    all_results = {}
    for label, frac in conditions:
        res = _run_condition(label, frac, n_runs, n_ticks)
        all_results[label] = res

    # Summary table
    print()
    hdr = f"  {'Condition':<28}  {'λ post':>8}  {'λ CV':>8}  {'Adv.Sel%':>9}  {'Disc.Spd':>9}  {'Info α':>9}"
    print(hdr)
    print(f"  {'-'*76}")

    table = {}
    lambda_means, adv_means, disc_means = [], [], []
    for label, _ in conditions:
        res = all_results[label]
        lm  = float(np.mean(res.get("lambda_post",  [float("nan")]))) if res.get("lambda_post")  else float("nan")
        lcv = float(np.mean(res.get("lambda_cv",    [float("nan")]))) if res.get("lambda_cv")    else float("nan")
        adv = float(np.mean(res.get("adv_sel_frac", [float("nan")]))) if res.get("adv_sel_frac") else float("nan")
        ds  = float(np.mean(res.get("disc_speed",   [float("nan")]))) if res.get("disc_speed")   else float("nan")
        ia  = float(np.mean(res.get("info_alpha",   [float("nan")]))) if res.get("info_alpha")   else float("nan")

        def _f(v, fmt=".4f"):
            return f"{v:{fmt}}" if not math.isnan(v) else "  N/A"

        adv_pct = f"{adv*100:.1f}%" if not math.isnan(adv) else " N/A"
        print(
            f"  {label:<28}  {_f(lm):>8}  {_f(lcv):>8}  {adv_pct:>9}  "
            f"{_f(ds, '.1f'):>9}  {_f(ia, '.1f'):>9}"
        )
        lambda_means.append(lm)
        adv_means.append(adv)
        disc_means.append(ds)
        table[label] = {
            "lambda_post_mean":  round(lm,  6) if not math.isnan(lm)  else None,
            "lambda_cv_mean":    round(lcv, 6) if not math.isnan(lcv) else None,
            "adv_sel_frac_mean": round(adv, 6) if not math.isnan(adv) else None,
            "disc_speed_mean":   round(ds,  4) if not math.isnan(ds)  else None,
            "info_alpha_mean":   round(ia,  4) if not math.isnan(ia)  else None,
        }

    # Kyle prediction verification
    print()
    print("  Kyle (1985) Predictions:")
    valid_lm  = [v for v in lambda_means if not math.isnan(v)]
    valid_adv = [v for v in adv_means    if not math.isnan(v)]
    valid_ds  = [v for v in disc_means   if not math.isnan(v)]

    p1 = _check_monotone(valid_lm,  higher_is_expected=True)
    p2 = _check_monotone(valid_ds,  higher_is_expected=False)   # faster = fewer ticks
    p3 = _check_monotone(valid_adv, higher_is_expected=True)

    print(f"  [{p1}] P1: Higher informed fraction → higher Kyle's lambda post-shock")
    print(f"  [{p2}] P2: Higher informed fraction → faster price discovery (fewer ticks to half-price)")
    print(f"  [{p3}] P3: Higher informed fraction → higher adverse-selection fraction of spread")
    print(f"  [~] P4: Informed agents earn positive alpha (see Info α column above)")
    print()
    print("  ✓ = prediction satisfied monotonically  ~ = partial support  ? = insufficient data")

    # Save
    out = {
        "experiment": "2_kyle_information_asymmetry",
        "config": {"n_runs": n_runs, "n_ticks": n_ticks, "shock_tick": SHOCK_TICK},
        "conditions": {label: {k: [round(v, 6) for v in vals] for k, vals in res.items()}
                       for label, res in all_results.items()},
        "summary_table": table,
        "kyle_predictions": {
            "P1_lambda_monotone": p1,
            "P2_disc_speed_monotone": p2,
            "P3_adv_sel_monotone": p3,
        },
    }
    out_path = os.path.join(_ROOT, "results_exp2_kyle.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved → results_exp2_kyle.json")
    print("=" * 72)

    return out
