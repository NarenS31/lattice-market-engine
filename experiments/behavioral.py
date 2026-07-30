"""
Experiment 1: Behavioral Finance Anomaly Recovery Without Explicit Prompting
=============================================================================

Research question
-----------------
Do LLMs spontaneously exhibit canonical behavioral finance biases — disposition
effect, anchoring, loss aversion — even when agent prompts contain NO explicit
behavioral descriptors?

Experimental design
-------------------
  Condition A  Archetype prompts  (explicit: "you are loss-averse", "follow momentum", …)
  Condition B  Neutral prompts    (role only: "you are a retail investor", no heuristics)

Both conditions run identical markets with identical macro shocks.

Metrics
-------
  DE    Disposition Effect Ratio = PGR / PLR  (Odean 1998)
          PGR = P(sell | unrealized gain)
          PLR = P(sell | unrealized loss)
          DE > 1  →  selling winners, holding losers  (classic bias)

  ANC   Anchoring Score = mean( (limit_price − 100) / (mid_price − 100) )
        computed on ticks where |mid − 100| > 0.5 so denominator is stable.
          ANC ≈ 1  →  fully adjusted to current mid  (no anchoring)
          ANC ≈ 0  →  prices cluster near initial anchor  (strong anchoring)

  LA    Loss Aversion Proxy = mean_qty_after_loss / mean_qty_after_gain
          LA < 1  →  trade less after losses  (paralysis / loss aversion)

Statistical comparison: Welch's t-test per metric, Condition A vs B.

Interpretation
--------------
  If both conditions show comparable biases →  LLMs have internalised these from
  pre-training; the archetype prompt is NOT the causal driver.

  If only Condition A shows the bias →  the prompt IS load-bearing; the model
  does not spontaneously generate the behaviour.

Run
---
  python -m experiments.run_all --exp 1 [--runs N] [--ticks N]
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


# ---------------------------------------------------------------------------
# Neutral-prompt agent subclasses
# ---------------------------------------------------------------------------

class NeutralRetailAgent(RetailAgent):
    """Retail investor role — no explicit loss aversion / emotional descriptors."""
    _SYSTEM_PROMPT = (
        "You are a retail investor making trading decisions in a simulated stock market. "
        "Trade small quantities, 1 to 5 shares per tick. "
        "Always respond with ONLY a JSON object, no preamble:\n"
        '{"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one sentence>"}'
    )


class NeutralInstitutionalAgent(InstitutionalAgent):
    """Fund manager role — no explicit VWAP / mean-reversion descriptors."""
    _SYSTEM_PROMPT = (
        "You are a fund manager making trading decisions in a simulated stock market. "
        "Trade medium quantities, 10 to 30 shares per tick. "
        "Always respond with ONLY a JSON object, no preamble:\n"
        '{"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one sentence>"}'
    )


class NeutralQuantAgent(QuantAgent):
    """Quantitative trader role — no explicit momentum / signal descriptors."""
    _SYSTEM_PROMPT = (
        "You are a quantitative trader making decisions in a simulated stock market. "
        "Trade medium quantities, 5 to 15 shares per tick. "
        "Always respond with ONLY a JSON object, no preamble:\n"
        '{"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one sentence>"}'
    )


# ---------------------------------------------------------------------------
# BehavioralSimulation — extends Simulation with per-tick tracking
# ---------------------------------------------------------------------------

class BehavioralSimulation(Simulation):
    """
    Adds cost-basis tracking and per-tick behavioral snapshots to Simulation.

    After run(), call compute_metrics() to extract DE, ANC, LA ratios
    aggregated across all LLM agents and all ticks.
    """

    INITIAL_REF = 100.0

    def __init__(self, book: OrderBook, agents: list) -> None:
        super().__init__(book, agents)
        self._cost_bases: list[float] = [0.0] * len(agents)
        self._ticks: list[list[dict]] = []

    def run(
        self,
        n_ticks: int,
        factor_pulse=None,
        verbose: bool = False,
    ) -> list:
        summaries = []

        for tick_idx in range(n_ticks):
            snap = json.loads(self.book.to_json())
            ref = (
                snap.get("mid_price")
                or snap.get("last_clearing_price")
                or self.INITIAL_REF
            )

            macro_context: Optional[dict] = None
            if factor_pulse is not None:
                state = factor_pulse.get_state(tick_idx)
                macro_context = factor_pulse.to_context(state)
            self.macro_log.append(macro_context or {})

            # Snapshot agent state BEFORE decisions (needed for DE)
            pre_states = [
                {
                    "position": a.position,
                    "cost_basis": self._cost_bases[i],
                    "wealth": a.wealth(ref),
                }
                for i, a in enumerate(self.agents)
            ]

            # Collect orders
            new_orders: list[Order] = []
            tick_reasoning: list[str] = []
            agent_orders: list[list[Order]] = [[] for _ in self.agents]

            for i, agent in enumerate(self.agents):
                orders, reasoning = agent.decide_orders(snap, macro_context, tick_idx)
                tick_reasoning.append(reasoning)
                for o in orders:
                    new_orders.append(o)
                    self._order_map[o.order_id] = agent
                    agent_orders[i].append(o)
            self.reasoning_log.append(tick_reasoning)

            summary = self.book.advance_tick(new_orders)
            summaries.append(summary)

            fill_price = (
                summary.clearing_price
                or self.book.last_clearing_price
                or self.INITIAL_REF
            )

            # Apply fills; update cost basis sequentially per fill
            for oid, agent in self._order_map.items():
                order = self.book.order_history.get(oid)
                if order is None or order.filled_qty <= 0:
                    continue
                idx = self.agents.index(agent)
                if order.side == Side.BUY:
                    old_pos = agent.position      # pre-update position
                    old_basis = self._cost_bases[idx]
                    agent.update_position(order, fill_price)
                    new_pos = old_pos + order.filled_qty
                    if new_pos > 0:
                        self._cost_bases[idx] = (
                            old_pos * old_basis + order.filled_qty * fill_price
                        ) / new_pos
                else:
                    agent.update_position(order, fill_price)
                    # cost basis unchanged on sells

            self._order_map = {}

            post_wealth = [a.wealth(fill_price) for a in self.agents]
            self.wealth_log.append(post_wealth)
            for a, w in zip(self.agents, post_wealth):
                a.append_wealth(w)

            # Record behavioral snapshot for this tick
            tick_snaps = []
            for i, agent in enumerate(self.agents):
                orders = agent_orders[i]
                lp_list = [o.price for o in orders if o.price is not None]
                tick_snaps.append(
                    {
                        "pre_position": pre_states[i]["position"],
                        "pre_cost_basis": pre_states[i]["cost_basis"],
                        "pre_wealth": pre_states[i]["wealth"],
                        "post_wealth": post_wealth[i],
                        "ref_price": ref,
                        "fill_price": fill_price,
                        "has_sell": any(o.side == Side.SELL for o in orders),
                        "has_buy": any(o.side == Side.BUY for o in orders),
                        "total_qty": sum(o.qty for o in orders),
                        "limit_prices": lp_list,
                    }
                )
            self._ticks.append(tick_snaps)

        return summaries

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_metrics(self, llm_agent_indices: list[int]) -> dict:
        """
        Compute DE, ANC, LA across all ticks for the given agent indices.
        Returns a dict with scalar metric values (NaN if insufficient data).
        """
        # --- Disposition Effect (Odean 1998) ---
        gain_sold = gain_held = loss_sold = loss_held = 0
        for tick in self._ticks:
            for i in llm_agent_indices:
                snap = tick[i]
                pos = snap["pre_position"]
                basis = snap["pre_cost_basis"]
                price = snap["ref_price"]
                if pos <= 0 or basis <= 0:
                    continue
                is_gain = price > basis
                is_loss = price < basis
                sold = snap["has_sell"]
                if is_gain:
                    if sold:
                        gain_sold += 1
                    else:
                        gain_held += 1
                if is_loss:
                    if sold:
                        loss_sold += 1
                    else:
                        loss_held += 1

        pgr = gain_sold / (gain_sold + gain_held) if (gain_sold + gain_held) > 0 else float("nan")
        plr = loss_sold / (loss_sold + loss_held) if (loss_sold + loss_held) > 0 else float("nan")
        de = pgr / plr if (not math.isnan(pgr) and not math.isnan(plr) and plr > 0) else float("nan")

        # --- Anchoring Score ---
        anchor_fracs: list[float] = []
        for tick in self._ticks:
            for i in llm_agent_indices:
                snap = tick[i]
                mid = snap["ref_price"]
                denom = mid - self.INITIAL_REF
                if abs(denom) < 0.5:  # skip ticks where mid is near initial anchor
                    continue
                for lp in snap["limit_prices"]:
                    if lp is not None:
                        anchor_fracs.append((lp - self.INITIAL_REF) / denom)

        anc = float(np.mean(anchor_fracs)) if anchor_fracs else float("nan")

        # --- Loss Aversion Proxy ---
        loss_qtys: list[float] = []
        gain_qtys: list[float] = []
        for tick_idx in range(1, len(self._ticks)):
            for i in llm_agent_indices:
                prev = self._ticks[tick_idx - 1][i]
                curr = self._ticks[tick_idx][i]
                wealth_delta = prev["post_wealth"] - prev["pre_wealth"]
                qty = curr["total_qty"]
                if wealth_delta < -1e-6:
                    loss_qtys.append(qty)
                elif wealth_delta > 1e-6:
                    gain_qtys.append(qty)

        la = (
            float(np.mean(loss_qtys)) / float(np.mean(gain_qtys))
            if loss_qtys and gain_qtys and np.mean(gain_qtys) > 0
            else float("nan")
        )

        # --- Hold rate (fraction of agent ticks that submitted nothing) ---
        n_hold = sum(
            1
            for tick in self._ticks
            for i in llm_agent_indices
            if self._ticks[self._ticks.index(tick)][i]["total_qty"] == 0
        )
        n_total = len(self._ticks) * len(llm_agent_indices)
        hold_rate = n_hold / n_total if n_total > 0 else float("nan")

        return {
            "de_ratio": de,
            "pgr": pgr,
            "plr": plr,
            "anchoring_score": anc,
            "la_ratio": la,
            "hold_rate": hold_rate,
        }


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def _run_one(agent_factories: list, seed: int, n_ticks: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    agents = [f(i) for i, f in enumerate(agent_factories)]
    # LLM agents are indices 0..n-2; last agent is MarketMaker
    llm_indices = list(range(len(agents) - 1))

    book = OrderBook(symbol="EXP1")
    sim = BehavioralSimulation(book, agents)
    sim.seed_book(fair_value=100.0)
    pulse = FactorPulse()
    sim.run(n_ticks=n_ticks, factor_pulse=pulse)

    return sim.compute_metrics(llm_indices)


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def _run_condition(label: str, agent_factories: list, n_runs: int, n_ticks: int) -> dict:
    results: dict[str, list[float]] = defaultdict(list)
    print(f"  Condition {label}: ", end="", flush=True)
    for seed in range(n_runs):
        r = _run_one(agent_factories, seed=seed * 7 + 13, n_ticks=n_ticks)
        for k, v in r.items():
            if not math.isnan(v):
                results[k].append(v)
        print(".", end="", flush=True)
    print()
    return {k: list(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test; p-value via normal approximation (adequate for n >= 8)."""
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
    # two-sided p via normal approximation
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

def run_experiment1(n_runs: int = 10, n_ticks: int = 20) -> dict:
    """
    Run Experiment 1 and print a comparison table.
    Returns the full results dict (also saved to results_exp1_behavioral.json).
    """
    print("\n" + "=" * 68)
    print("Experiment 1: Behavioral Finance Anomaly Recovery")
    print(f"  Conditions: A (archetype prompts) vs B (neutral prompts)")
    print(f"  Runs: {n_runs} per condition  |  Ticks: {n_ticks} per run")
    print("=" * 68)

    def _archetype_factories(idx):
        return [
            lambda i: RetailAgent(cash=10_000.0, name=f"Retail-{i}"),
            lambda i: InstitutionalAgent(cash=100_000.0, name=f"Inst-{i}"),
            lambda i: QuantAgent(cash=50_000.0, name=f"Quant-{i}"),
            lambda i: MarketMakerAgent(name=f"MM-{i}"),
        ]

    def _neutral_factories(idx):
        return [
            lambda i: NeutralRetailAgent(cash=10_000.0, name=f"NRetail-{i}"),
            lambda i: NeutralInstitutionalAgent(cash=100_000.0, name=f"NInst-{i}"),
            lambda i: NeutralQuantAgent(cash=50_000.0, name=f"NQuant-{i}"),
            lambda i: MarketMakerAgent(name=f"MM-{i}"),
        ]

    print("\nRunning condition A (archetype prompts)...")
    cond_a = _run_condition("A", _archetype_factories(0), n_runs, n_ticks)

    print("Running condition B (neutral prompts)...")
    cond_b = _run_condition("B", _neutral_factories(0), n_runs, n_ticks)

    # Print comparison table
    metrics = [
        ("de_ratio",        "Disposition Effect Ratio (DE)",      "1.0 = no bias; > 1 = selling winners",  ">1 in A & B?"),
        ("anchoring_score", "Anchoring Score (ANC)",               "1.0 = fully adjusted; 0 = anchored",    "A > B?"),
        ("la_ratio",        "Loss Aversion Proxy (LA)",            "< 1 = trade less after losses",         "<1 in A & B?"),
        ("hold_rate",       "Hold Rate",                           "fraction of agent ticks with no trade", ""),
    ]

    print()
    print(f"  {'Metric':<35}  {'Cond A':>8}  {'Cond B':>8}  {'Diff':>8}  {'t':>6}  {'p':>7}  Sig")
    print(f"  {'-'*80}")

    table_rows = {}
    for key, label, note, _ in metrics:
        va = cond_a.get(key, [])
        vb = cond_b.get(key, [])
        if not va or not vb:
            print(f"  {label:<35}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>6}  {'N/A':>7}")
            continue
        ma, mb = np.mean(va), np.mean(vb)
        diff = ma - mb
        t, p = _welch_t(va, vb)
        sig = _sig(p)
        p_str = f"{p:.3f}" if not math.isnan(p) else " N/A"
        t_str = f"{t:+.2f}" if not math.isnan(t) else " N/A"
        print(
            f"  {label:<35}  {ma:>8.4f}  {mb:>8.4f}  {diff:>+8.4f}"
            f"  {t_str:>6}  {p_str:>7}  {sig}  ({note})"
        )
        table_rows[key] = {
            "cond_a_mean": round(float(ma), 6),
            "cond_b_mean": round(float(mb), 6),
            "diff": round(float(diff), 6),
            "t_stat": round(t, 4) if not math.isnan(t) else None,
            "p_value": round(p, 4) if not math.isnan(p) else None,
            "significant_05": bool(not math.isnan(p) and p < 0.05),
        }

    print()
    print("  Significance: ** p<0.01  * p<0.05")
    print()

    # Interpretation
    de_a = np.mean(cond_a.get("de_ratio", [float("nan")]))
    de_b = np.mean(cond_b.get("de_ratio", [float("nan")]))
    anc_key = table_rows.get("anchoring_score", {})
    la_key  = table_rows.get("la_ratio", {})

    print("  Interpretation:")
    if not math.isnan(de_a) and not math.isnan(de_b):
        if de_a > 1 and de_b > 1:
            print(f"  [DE]  BOTH conditions DE > 1 (A={de_a:.3f}, B={de_b:.3f}). LLMs exhibit")
            print(f"        disposition effect regardless of prompt — likely internalised")
            print(f"        from pre-training on financial text.")
        elif de_a > 1 and de_b <= 1:
            print(f"  [DE]  Only Condition A shows DE > 1. Archetype prompt IS driving the bias.")
        else:
            print(f"  [DE]  Neither condition shows clear DE > 1.")

    if anc_key.get("significant_05"):
        winner = "A" if anc_key["cond_a_mean"] > anc_key["cond_b_mean"] else "B"
        print(f"  [ANC] Significant difference (p={anc_key['p_value']:.3f}). Condition {winner}")
        print(f"        shows higher anchoring adjustment — archetype prompts affect anchoring.")
    else:
        print(f"  [ANC] No significant prompt effect on anchoring (p={anc_key.get('p_value', 'N/A')}).")

    if la_key:
        la_a = la_key.get("cond_a_mean", float("nan"))
        la_b = la_key.get("cond_b_mean", float("nan"))
        if la_a < 1 and la_b < 1:
            print(f"  [LA]  BOTH conditions LA < 1 (A={la_a:.3f}, B={la_b:.3f}). Agents trade less")
            print(f"        after losses in both conditions — consistent with loss aversion.")

    # Save results
    out = {
        "experiment": "1_behavioral_bias_recovery",
        "config": {"n_runs": n_runs, "n_ticks": n_ticks},
        "condition_a_archetype": {k: [round(v, 6) for v in vals] for k, vals in cond_a.items()},
        "condition_b_neutral":   {k: [round(v, 6) for v in vals] for k, vals in cond_b.items()},
        "comparison": table_rows,
    }
    out_path = os.path.join(_ROOT, "results_exp1_behavioral.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Saved → results_exp1_behavioral.json")
    print("=" * 68)

    return out
