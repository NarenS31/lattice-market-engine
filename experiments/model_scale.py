"""
Experiment 3: LLM Capability Level vs. Financial Rationality
=============================================================

Research question
-----------------
Does a more capable LLM produce more informationally efficient markets, lower
crash rates, and higher trading participation than a less capable LLM, holding
everything else constant?

Motivation
----------
Scaling laws in NLP suggest larger models reason better.  If this translates
to market settings, more capable models should:
  • price assets closer to fundamental value  (better reasoning about FV)
  • participate more  (fewer spurious holds)
  • produce fewer liquidity events  (flash crashes, wide spreads)
  • but possibly generate *more* wealth inequality  (smarter agents exploit
    dumber counterparts — a systemic-risk implication)

Models tested
-------------
  Haiku   claude-haiku-4-5-20251001   (fast, cheapest)
  Sonnet  claude-sonnet-4-6           (mid-tier)

  Opus (claude-opus-4-7) is left as an optional extension; add its name to
  MODELS below and re-run.  API costs are ~10× Sonnet per call.

Metrics
-------
  price_eff     Mean squared deviation of clearing price from initial FV (100.0)
                Lower is better — tighter pricing around fundamentals.

  sharpe        Annualised Sharpe ratio of per-tick returns across the simulation.
                Higher is better — smoother, more rational price path.

  crash_rate    Fraction of runs containing at least one >2% single-tick drop.
                Lower is better.

  hold_rate     Fraction of agent-ticks where the LLM submits no order.
                Lower is better — more rational agents should always trade.

  gini          Gini coefficient of final mark-to-market wealth across all agents.
                Purely descriptive — higher = more unequal final distribution.

  spread_post   Mean bid-ask spread in post-shock ticks.
                Lower spread = more liquid market.

Run
---
  python -m experiments.run_all --exp 3 [--runs N] [--ticks N]
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

from engine import Order, OrderBook                 # noqa: E402
from agents import (                                # noqa: E402
    RetailAgent, InstitutionalAgent, QuantAgent,
    MarketMakerAgent, Simulation,
)
import agents as _agents_module                     # noqa: E402
from factor_pulse import FactorPulse                # noqa: E402

INITIAL_FV = 100.0
SHOCK_TICK  = 10

# Models to compare.  Add "claude-opus-4-7" here if you want to include Opus.
MODELS = [
    ("Haiku",  "claude-haiku-4-5-20251001"),
    ("Sonnet", "claude-sonnet-4-6"),
]



# ---------------------------------------------------------------------------
# Scale-tracking simulation
# ---------------------------------------------------------------------------

class ScaleSimulation(Simulation):
    """
    Extends Simulation to record all data needed for the scale metrics.
    The model name must be set on the module-level before constructing agents.
    """

    def run(
        self,
        n_ticks: int,
        factor_pulse=None,
        verbose: bool = False,
    ) -> list:
        summaries = []
        self._llm_hold_ticks = 0
        self._llm_total_ticks = 0
        llm_agents = [a for a in self.agents if not isinstance(a, MarketMakerAgent)]

        for tick_idx in range(n_ticks):
            snap = json.loads(self.book.to_json())

            macro_context: Optional[dict] = None
            if factor_pulse is not None:
                state = factor_pulse.get_state(tick_idx)
                macro_context = factor_pulse.to_context(state)
            self.macro_log.append(macro_context or {})

            new_orders: list[Order] = []
            tick_reasoning: list[str] = []
            for agent in self.agents:
                orders, reasoning = agent.decide_orders(snap, macro_context, tick_idx)
                tick_reasoning.append(reasoning)
                # Count hold rate for LLM agents only
                if not isinstance(agent, MarketMakerAgent):
                    self._llm_total_ticks += 1
                    if not orders:
                        self._llm_hold_ticks += 1
                for o in orders:
                    new_orders.append(o)
                    self._order_map[o.order_id] = agent
            self.reasoning_log.append(tick_reasoning)

            summary = self.book.advance_tick(new_orders)
            summaries.append(summary)

            fill_price = summary.clearing_price or self.book.last_clearing_price or INITIAL_FV
            for oid, agent in self._order_map.items():
                order = self.book.order_history.get(oid)
                if order is not None and order.filled_qty > 0:
                    agent.update_position(order, fill_price)
            self._order_map = {}

            tick_wealth = [a.wealth(fill_price) for a in self.agents]
            self.wealth_log.append(tick_wealth)
            for a, w in zip(self.agents, tick_wealth):
                a.append_wealth(w)

        return summaries


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_one(model_name: str, seed: int, n_ticks: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    original = _agents_module._MODEL
    _agents_module._MODEL = model_name
    try:
        agents = [
            RetailAgent(cash=10_000.0,   name="Retail"),
            InstitutionalAgent(cash=100_000.0, name="Inst"),
            QuantAgent(cash=50_000.0,    name="Quant"),
            MarketMakerAgent(name="MM"),
        ]
        book = OrderBook(symbol="EXP3")
        sim = ScaleSimulation(book, agents)
        sim.seed_book(fair_value=INITIAL_FV)
        pulse = FactorPulse()
        summaries = sim.run(n_ticks=n_ticks, factor_pulse=pulse)
    finally:
        _agents_module._MODEL = original

    clearing = [s.clearing_price for s in summaries]
    valid_px  = [p for p in clearing if p is not None]

    # Price efficiency: MSE vs INITIAL_FV
    price_eff = float(np.mean([(p - INITIAL_FV) ** 2 for p in valid_px])) if valid_px else float("nan")

    # Sharpe: annualised on per-tick returns
    rets: list[float] = []
    for i in range(1, len(valid_px)):
        if valid_px[i - 1] > 0:
            rets.append((valid_px[i] - valid_px[i - 1]) / valid_px[i - 1])
    sharpe = float("nan")
    if len(rets) >= 2:
        mu = float(np.mean(rets))
        sigma = float(np.std(rets, ddof=1))
        sharpe = (mu / sigma * math.sqrt(250)) if sigma > 0 else float("nan")

    # Flash crash: any tick with > 2% single-tick drop
    had_crash = False
    for i in range(1, len(valid_px)):
        if valid_px[i - 1] > 0 and (valid_px[i] - valid_px[i - 1]) / valid_px[i - 1] < -0.02:
            had_crash = True
            break

    # Hold rate (LLM agents only)
    hold_rate = (
        sim._llm_hold_ticks / sim._llm_total_ticks
        if sim._llm_total_ticks > 0 else float("nan")
    )

    # Gini coefficient of final wealth
    final_w = np.array(sim.wealth_log[-1]) if sim.wealth_log else np.array([])
    gini = float("nan")
    if len(final_w) > 1:
        sorted_w = np.sort(np.abs(final_w))
        n = len(sorted_w)
        idx = np.arange(1, n + 1)
        gini = float((2 * np.sum(idx * sorted_w) / (n * np.sum(sorted_w))) - (n + 1) / n)

    # Mean spread in post-shock ticks
    post_spreads = [
        s.spread for s in summaries[SHOCK_TICK:]
        if s.spread is not None
    ]
    spread_post = float(np.mean(post_spreads)) if post_spreads else float("nan")

    return {
        "price_eff": price_eff,
        "sharpe": sharpe,
        "crash": float(had_crash),
        "hold_rate": hold_rate,
        "gini": gini,
        "spread_post": spread_post,
    }


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def _run_condition(label: str, model_name: str, n_runs: int, n_ticks: int) -> dict:
    results: dict[str, list[float]] = defaultdict(list)
    print(f"  {label} ({model_name}): ", end="", flush=True)
    for seed in range(n_runs):
        r = _run_one(model_name, seed=seed * 17 + 3, n_ticks=n_ticks)
        for k, v in r.items():
            if not math.isnan(v):
                results[k].append(v)
        print(".", end="", flush=True)
    print()
    return {k: list(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Welch t-test (duplicate to keep module self-contained)
# ---------------------------------------------------------------------------

def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    a_arr, b_arr = np.array(a), np.array(b)
    m1, m2 = np.mean(a_arr), np.mean(b_arr)
    v1 = np.var(a_arr, ddof=1)
    v2 = np.var(b_arr, ddof=1)
    se = math.sqrt(v1 / len(a) + v2 / len(b))
    if se < 1e-12:
        return float("nan"), float("nan")
    t = float((m1 - m2) / se)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return t, p


def _sig(p: float) -> str:
    if math.isnan(p):
        return "  "
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "* "
    return "  "


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------

def run_experiment3(n_runs: int = 5, n_ticks: int = 20) -> dict:
    """
    Run Experiment 3 and print a capability-vs-rationality comparison table.
    Returns the full results dict (also saved to results_exp3_model_scale.json).
    """
    print("\n" + "=" * 72)
    print("Experiment 3: LLM Capability Level vs. Financial Rationality")
    model_str = "  vs  ".join(f"{lbl} ({m})" for lbl, m in MODELS)
    print(f"  Models: {model_str}")
    print(f"  Runs: {n_runs} per model  |  Ticks: {n_ticks} per run")
    print("=" * 72)

    print()
    all_results = {}
    for label, model_name in MODELS:
        all_results[label] = _run_condition(label, model_name, n_runs, n_ticks)

    metrics_meta = [
        ("price_eff",   "Price Efficiency (MSE vs FV=100)", "lower = better"),
        ("sharpe",      "Sharpe Ratio (annualised)",         "higher = better"),
        ("crash",       "Flash Crash Rate (fraction runs)",  "lower = better"),
        ("hold_rate",   "Hold Rate (LLM agents)",            "lower = better"),
        ("gini",        "Wealth Gini Coefficient",           "descriptive"),
        ("spread_post", "Mean Spread (post-shock ticks)",    "lower = better"),
    ]

    # Use first two models as the primary comparison pair
    lbl_a, lbl_b = MODELS[0][0], MODELS[1][0]
    res_a = all_results[lbl_a]
    res_b = all_results[lbl_b]

    print()
    col_w = max(len(lbl) for lbl, _ in MODELS) + 2
    hdr_models = "  ".join(f"{lbl:>{col_w}}" for lbl, _ in MODELS)
    print(f"  {'Metric':<36}  {hdr_models}  {'t':>6}  {'p':>7}  Sig  Direction")
    print(f"  {'-'*90}")

    table_rows = {}
    for key, label, note in metrics_meta:
        row_vals = []
        for lbl, _ in MODELS:
            v = all_results[lbl].get(key, [])
            row_vals.append(float(np.mean(v)) if v else float("nan"))

        va = res_a.get(key, [])
        vb = res_b.get(key, [])
        t, p = _welch_t(va, vb)
        sig = _sig(p)

        val_strs = "  ".join(
            f"{v:>{col_w}.4f}" if not math.isnan(v) else f"{'N/A':>{col_w}}"
            for v in row_vals
        )
        t_s = f"{t:+.2f}" if not math.isnan(t) else " N/A"
        p_s = f"{p:.3f}" if not math.isnan(p) else " N/A"

        # Determine which model "wins" on this metric
        if not math.isnan(row_vals[0]) and not math.isnan(row_vals[1]):
            if note == "lower = better":
                winner = lbl_a if row_vals[0] < row_vals[1] else lbl_b
            elif note == "higher = better":
                winner = lbl_a if row_vals[0] > row_vals[1] else lbl_b
            else:
                winner = "—"
        else:
            winner = "?"

        print(f"  {label:<36}  {val_strs}  {t_s:>6}  {p_s:>7}  {sig}  {winner} ({note})")
        table_rows[key] = {
            "means": {lbl: round(v, 6) if not math.isnan(v) else None for lbl, v in zip([l for l, _ in MODELS], row_vals)},
            "t_stat": round(t, 4) if not math.isnan(t) else None,
            "p_value": round(p, 4) if not math.isnan(p) else None,
            "significant_05": bool(not math.isnan(p) and p < 0.05),
        }

    print()
    print("  Significance: ** p<0.01  * p<0.05")
    print()

    # Overall verdict
    sonnet_wins = 0
    haiku_wins  = 0
    for key, _, note in metrics_meta:
        row = table_rows.get(key, {})
        means = row.get("means", {})
        va_m = means.get(lbl_a)
        vb_m = means.get(lbl_b)
        if va_m is None or vb_m is None:
            continue
        if note == "lower = better":
            if vb_m < va_m:
                sonnet_wins += 1
            else:
                haiku_wins += 1
        elif note == "higher = better":
            if vb_m > va_m:
                sonnet_wins += 1
            else:
                haiku_wins += 1

    print(f"  Score: {lbl_b} wins {sonnet_wins}/{len(metrics_meta)-1} rational-market metrics vs {lbl_a}.")
    if sonnet_wins > haiku_wins:
        print(f"  → {lbl_b} produces more informationally rational markets than {lbl_a}.")
        print(f"    Consistent with the hypothesis: capability ↑ → market rationality ↑.")
    elif haiku_wins > sonnet_wins:
        print(f"  → {lbl_a} produces more rational markets — higher capability does NOT")
        print(f"    translate to better market outcomes here.")
    else:
        print(f"  → Mixed evidence; no clear capability-rationality relationship.")

    # Save
    out = {
        "experiment": "3_model_scale_rationality",
        "config": {
            "n_runs": n_runs,
            "n_ticks": n_ticks,
            "models": {lbl: m for lbl, m in MODELS},
            "shock_tick": SHOCK_TICK,
        },
        "raw_results": {
            lbl: {k: [round(v, 6) for v in vals] for k, vals in res.items()}
            for lbl, res in all_results.items()
        },
        "comparison_table": table_rows,
    }
    out_path = os.path.join(_ROOT, "results_exp3_model_scale.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Saved → results_exp3_model_scale.json")
    print("=" * 72)

    return out
