"""
extensions/shock_experiment.py

Controlled A/B experiment: do narrative shocks produce different market
dynamics than numerical shocks describing the same underlying event?

Both versions use the Lopez-Lira (2025) market structure (`base_market.py`),
identical agents, identical fundamentals, identical random seeds. The only
difference is how the shock at round 10 is described to the agents.

  Version A — Numerical:
    "VIX: 45, sentiment: -0.80, fed_rate: 5.25, inflation: 3.8"

  Version B — Narrative:
    "A wildcat strike at the Atacama lithium mine has halted 18% of global
     lithium carbonate supply, forcing EV manufacturers to cut production
     guidance and battery suppliers to warn of margin compression. This is
     expected to cascade into reduced EV sales forecasts and potential
     credit rating downgrades for affected manufacturers."

Per run we record clearing prices, spreads, volumes, Kyle's lambda, and
each agent's reasoning string. Across runs we compute four post-shock
comparison metrics and a Welch's t-test verdict.

Usage:
    python -m extensions.shock_experiment --seeds 20

Output:
    extensions/experiment_results.json   (full per-run series + aggregates)
    stdout                               (clean comparison table + verdict)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    __package__ = "extensions"

from extensions.base_market import (  # noqa: E402
    BaseAgent,
    DividendModel,
    LLMAgent,
    Order,
    OrderBook,
    OrderType,
    Side,
    Trade,
    fundamental_value,
)
from extensions.supply_chain_extensions import (  # noqa: E402
    _WealthMemoryMixin,
    compute_microstructure_metrics,
)


# ---------------------------------------------------------------------------
# Shock contexts (the only thing that differs between A and B)
# ---------------------------------------------------------------------------

NUMERICAL_SHOCK = (
    "VIX: 45, sentiment: -0.80, fed_rate: 5.25, inflation: 3.8"
)

NARRATIVE_SHOCK = (
    "A wildcat strike at the Atacama lithium mine has halted 18% of global "
    "lithium carbonate supply, forcing EV manufacturers to cut production "
    "guidance and battery suppliers to warn of margin compression. This is "
    "expected to cascade into reduced EV sales forecasts and potential "
    "credit rating downgrades for affected manufacturers."
)


# ---------------------------------------------------------------------------
# Experimental agent classes (retail vs institutional, with wealth memory)
# ---------------------------------------------------------------------------

_RETAIL_PROMPT = (
    "You are a retail investor making real-time trading decisions in a "
    "simulated stock market. Your cognitive archetype: highly loss-averse "
    "(you dread losses far more than gains), you rely on recent price memory "
    "rather than deep analysis, you anxiously follow the most recent price "
    "direction with an emotionally-coloured signal, and you size positions "
    "small (1-5 shares per round)."
)

_INSTITUTIONAL_PROMPT = (
    "You are an institutional fund manager making real-time trading decisions "
    "in a simulated stock market. Your cognitive archetype: disciplined and "
    "systematic, you apply VWAP mean-reversion logic — buy when the price is "
    "below the recent volume-weighted average, sell when it is above. Trade "
    "in larger blocks (10-30 shares per round) and read the order book depth "
    "for secondary signals."
)

_JSON_TAIL = (
    "\n\nAlways respond with ONLY a single JSON object — no markdown, no "
    "preamble. Format: "
    '{"action": "buy"|"sell"|"hold", "quantity": <int>, '
    '"limit_price": <number or null>, "reasoning": "<one sentence>"}'
)


class _ExperimentAgent(_WealthMemoryMixin, LLMAgent):
    """LLM agent with rolling wealth memory + an experimental shock context.

    The simulation calls `set_shock_text(text)` each round; the agent injects
    that string into its user prompt under a neutral "Macro context:" frame
    so Versions A and B receive structurally identical prompts up to the
    macro string itself.
    """

    archetype: str = "Experiment"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._shock_text: Optional[str] = None

    def set_shock_text(self, text: Optional[str]) -> None:
        self._shock_text = text

    def _extra_user_lines(self, market_state: dict) -> list[str]:
        lines: list[str] = []
        if self._wealth_memory:
            hist = ", ".join(f"${w:.2f}" for w in self._wealth_memory)
            lines.append(
                f"Recent P&L (last {len(self._wealth_memory)} rounds wealth): {hist}."
            )
        if self._shock_text:
            lines.append(f"Macro context: {self._shock_text}")
        return lines


class ExperimentRetailAgent(_ExperimentAgent):
    _SYSTEM_PROMPT = _RETAIL_PROMPT + _JSON_TAIL
    archetype = "Retail"


class ExperimentInstitutionalAgent(_ExperimentAgent):
    _SYSTEM_PROMPT = _INSTITUTIONAL_PROMPT + _JSON_TAIL
    archetype = "Institutional"


class StaticMarketMaker(BaseAgent):
    """Deterministic two-sided liquidity provider — no LLM call.

    Posts a 1% wide quote around the mid each round so the LOB always has
    crossable liquidity for the LLM-driven retail and institutional agents.
    """

    archetype: str = "MM"

    def decide(self, market_state: dict) -> tuple[list[Order], str]:
        ref = (
            market_state.get("mid")
            or market_state.get("last_clearing_price")
            or market_state.get("initial_price", 28.0)
        )
        bid = Order(
            Side.BUY, OrderType.LIMIT, 30.0,
            round(ref * 0.99, 2), agent_id=self.agent_id,
        )
        ask = Order(
            Side.SELL, OrderType.LIMIT, 30.0,
            round(ref * 1.01, 2), agent_id=self.agent_id,
        )
        return [bid, ask], (
            f"{self.agent_id}: quote {ref * 0.99:.2f}/{ref * 1.01:.2f}"
        )


# ---------------------------------------------------------------------------
# Per-run record + helpers
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    seed: int
    version: str  # "A_numerical" | "B_narrative"
    clearing_prices: list[Optional[float]] = field(default_factory=list)
    spreads: list[Optional[float]] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    kyles_lambdas: list[Optional[float]] = field(default_factory=list)
    # reasoning_log[round_idx] = [{"agent_id", "archetype", "text"}, ...]
    reasoning_log: list[list[dict]] = field(default_factory=list)


def _parse_action(text: str) -> str:
    """Pull 'buy' / 'sell' / 'hold' out of an `LLMAgent.decide` reasoning string.

    Accepted formats (set by `LLMAgent.decide` and `StaticMarketMaker.decide`):
      "<agent_id>: hold — ..."
      "<agent_id>: BUY 5@28.50 — ..."
      "<agent_id>: SELL 10@27.90 — ..."
      "<agent_id>: API unavailable, holding."
      "<agent_id>: quote 27.72/28.28"
    """
    parts = text.split(":", 1)
    if len(parts) < 2:
        return "hold"
    body = parts[1].strip().upper()
    if body.startswith("BUY"):
        return "buy"
    if body.startswith("SELL"):
        return "sell"
    return "hold"


def _build_agents() -> list[BaseAgent]:
    return [
        ExperimentRetailAgent("Retail-1",
                              initial_cash=10_000.0, initial_shares=10),
        ExperimentRetailAgent("Retail-2",
                              initial_cash=10_000.0, initial_shares=10),
        ExperimentInstitutionalAgent("Inst-1",
                                     initial_cash=100_000.0, initial_shares=20),
        ExperimentInstitutionalAgent("Inst-2",
                                     initial_cash=100_000.0, initial_shares=20),
        StaticMarketMaker("MM-1",
                          initial_cash=1_000_000.0, initial_shares=200),
    ]


# ---------------------------------------------------------------------------
# Single-run simulator
# ---------------------------------------------------------------------------

def _run_one(
    seed: int,
    version: str,
    shock_text: str,
    num_rounds: int = 15,
    shock_round: int = 10,
    initial_price: float = 28.0,
    redemption_value: float = 28.0,
    interest_rate: float = 0.05,
    parallel_agents: bool = True,
) -> RunResult:
    """Run a single simulation with the given shock string injected at
    `shock_round`. The seed governs the dividend stream so Versions A and B
    see identical fundamentals for a given seed."""
    random.seed(seed)

    agents = _build_agents()
    book = OrderBook()
    dividend_model = DividendModel(high=1.6, low=1.2, prob_high=0.5)

    result = RunResult(seed=seed, version=version)
    recent_clearing: list[float] = []

    for round_no in range(1, num_rounds + 1):
        # 1. Inject the shock text on every LLM agent for this round.
        active_text = shock_text if round_no >= shock_round else None
        for a in agents:
            if isinstance(a, _ExperimentAgent):
                a.set_shock_text(active_text)

        # 2. Snapshot mid BEFORE this round's orders arrive (for Kyle's lambda).
        prev_mid = book.mid_price()

        market_state = {
            **book.book_snapshot(),
            "round_number": round_no,
            "num_rounds": num_rounds,
            "rounds_remaining": num_rounds - round_no + 1,
            "initial_price": initial_price,
            "expected_dividend": dividend_model.expected,
            "interest_rate": interest_rate,
            "redemption_value": redemption_value,
            "fundamental_value": fundamental_value(
                num_rounds, dividend_model.expected,
                interest_rate, redemption_value, round_no,
            ),
            "recent_clearing_prices": list(recent_clearing[-5:]),
        }

        # 3. Solicit decisions. Within a round agents are independent, so we
        #    fan out the LLM calls in parallel.
        if parallel_agents:
            with ThreadPoolExecutor(max_workers=len(agents)) as ex:
                futures = [ex.submit(a.decide, market_state) for a in agents]
                decisions = [f.result() for f in futures]
        else:
            decisions = [a.decide(market_state) for a in agents]

        new_orders: list[Order] = []
        per_round_reasoning: list[dict] = []
        for (orders, reasoning), agent in zip(decisions, agents):
            new_orders.extend(orders)
            per_round_reasoning.append({
                "agent_id": agent.agent_id,
                "archetype": getattr(agent, "archetype", "Unknown"),
                "text": reasoning,
                "action": _parse_action(reasoning),
            })
        result.reasoning_log.append(per_round_reasoning)

        # 4. Match, settle, log.
        market_result = book.match_round(new_orders, round_no)
        by_id = {a.agent_id: a for a in agents}
        for tr in market_result.trades:
            if tr.buyer_id in by_id:
                by_id[tr.buyer_id].apply_trade(tr)
            if tr.seller_id in by_id:
                by_id[tr.seller_id].apply_trade(tr)

        if market_result.clearing_price is not None:
            recent_clearing.append(market_result.clearing_price)

        curr_mid = book.mid_price()
        kl, _, _, _ = compute_microstructure_metrics(
            prev_mid, curr_mid, new_orders, market_result
        )

        result.clearing_prices.append(market_result.clearing_price)
        result.spreads.append(market_result.spread)
        result.volumes.append(market_result.volume)
        result.kyles_lambdas.append(kl)

        # 5. Dividend, interest, wealth memory.
        d = dividend_model.realise()
        for a in agents:
            a.receive_dividend(d)
            a.receive_interest(interest_rate)
        mark = (
            market_result.clearing_price
            or book.mid_price()
            or initial_price
        )
        for a in agents:
            if isinstance(a, _WealthMemoryMixin):
                a.record_wealth(a.wealth(mark))

    return result


# ---------------------------------------------------------------------------
# Aggregation: 4 post-shock comparison metrics
# ---------------------------------------------------------------------------

def _per_run_metrics(
    runs: list[RunResult],
    shock_round: int,
    post_window: int = 5,
) -> dict[str, list[float]]:
    """Return one number per run for each of the 4 metrics."""
    pre_idx = list(range(0, shock_round - 1))
    n_rounds_available = min((len(r.spreads) for r in runs), default=0)
    post_idx = [
        i for i in range(shock_round - 1, shock_round - 1 + post_window)
        if i < n_rounds_available
    ]

    spread_widening: list[float] = []
    volume_drop: list[float] = []
    price_impact: list[float] = []
    divergence_fraction: list[float] = []

    for run in runs:
        # Spread widening (post avg − pre avg).
        pre_s = [s for i in pre_idx for s in [run.spreads[i]] if s is not None]
        post_s = [s for i in post_idx for s in [run.spreads[i]] if s is not None]
        if pre_s and post_s:
            spread_widening.append(
                sum(post_s) / len(post_s) - sum(pre_s) / len(pre_s)
            )

        # Volume drop (pre avg − post avg).
        pre_v = [run.volumes[i] for i in pre_idx]
        post_v = [run.volumes[i] for i in post_idx]
        if pre_v and post_v:
            volume_drop.append(
                sum(pre_v) / len(pre_v) - sum(post_v) / len(post_v)
            )

        # Price impact (mean |Kyle's lambda| post-shock).
        post_lams = [
            abs(run.kyles_lambdas[i])
            for i in post_idx
            if run.kyles_lambdas[i] is not None
        ]
        if post_lams:
            price_impact.append(sum(post_lams) / len(post_lams))

        # Direction divergence between retail and institutional, post-shock.
        divergent = 0
        for i in post_idx:
            row = run.reasoning_log[i]
            retail = {r["action"] for r in row if r["archetype"] == "Retail"}
            inst = {r["action"] for r in row if r["archetype"] == "Institutional"}
            opposed = (
                ("buy" in retail and "sell" in inst)
                or ("sell" in retail and "buy" in inst)
            )
            divergent += int(opposed)
        divergence_fraction.append(divergent / len(post_idx))

    return {
        "spread_widening": spread_widening,
        "volume_drop": volume_drop,
        "price_impact": price_impact,
        "divergence_fraction": divergence_fraction,
    }


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _welch_t(a: list[float], b: list[float]) -> tuple[float, str]:
    """Welch's two-sample t-statistic and a coarse verdict at α=0.05."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), "insufficient data"
    ma, sa = _mean_std(a)
    mb, sb = _mean_std(b)
    se = math.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b))
    if se == 0:
        return float("nan"), "zero variance"
    t = (mb - ma) / se  # B - A: positive means narrative > numerical
    verdict = (
        "different (|t|>1.96, p<0.05)"
        if abs(t) > 1.96 else "not significantly different"
    )
    return t, verdict


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "spread_widening": ("Avg spread widening (post − pre)", "$"),
    "volume_drop":     ("Avg volume drop (pre − post)",     "shares"),
    "price_impact":    ("Avg |Kyle's λ| post-shock",         "$/share"),
    "divergence_fraction": (
        "Retail/Institutional opposite-direction fraction", "of ticks"
    ),
}


def _print_table(per_run_a: dict, per_run_b: dict, n_seeds: int) -> None:
    print("\n" + "=" * 84)
    print(f"  Controlled A/B comparison — {n_seeds} seeds × 2 versions, "
          f"5-tick post-shock window")
    print("=" * 84)
    print(f"  {'Metric':<48}  {'A (numerical)':>14}  {'B (narrative)':>14}  Verdict")
    print("-" * 84)
    for key, (label, unit) in _METRIC_LABELS.items():
        a_vals = per_run_a[key]
        b_vals = per_run_b[key]
        ma, sa = _mean_std(a_vals)
        mb, sb = _mean_std(b_vals)
        t, verdict = _welch_t(a_vals, b_vals)
        a_cell = f"{ma:>+8.4f}±{sa:.3f}"
        b_cell = f"{mb:>+8.4f}±{sb:.3f}"
        full_label = f"{label} [{unit}]"
        print(f"  {full_label:<48}  {a_cell:>14}  {b_cell:>14}  "
              f"t={t:+.2f} → {verdict}")
    print("=" * 84)
    print("\n  Verdict legend: t-statistic compares narrative B vs. numerical A.")
    print("  Positive t = narrative produces larger post-shock effect than numerical.")


def _serialize_run(r: RunResult) -> dict:
    return {
        "seed": r.seed,
        "version": r.version,
        "clearing_prices": r.clearing_prices,
        "spreads": r.spreads,
        "volumes": r.volumes,
        "kyles_lambdas": r.kyles_lambdas,
        "reasoning_log": r.reasoning_log,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=20,
        help="Number of seeds (each seed produces one run of A and one of B).",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=15,
        help="Total trading rounds per simulation.",
    )
    parser.add_argument(
        "--shock-round", type=int, default=10,
        help="Round at which the shock context is first injected into prompts.",
    )
    parser.add_argument(
        "--post-window", type=int, default=5,
        help="Number of post-shock rounds used for the comparison metrics.",
    )
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "experiment_results.json"),
        help="Where to write the full per-run + aggregate JSON.",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable per-round agent parallelism (debug option).",
    )
    args = parser.parse_args()

    print("\n" + "#" * 84)
    print("# extensions/shock_experiment.py — narrative vs numerical shock A/B test")
    print("#" * 84)
    print(f"# seeds={args.seeds}  rounds={args.num_rounds}  shock_round={args.shock_round}"
          f"  post_window={args.post_window}")

    runs_a: list[RunResult] = []
    runs_b: list[RunResult] = []
    t0 = time.time()
    for seed in range(args.seeds):
        elapsed = time.time() - t0
        print(f"\n[seed {seed + 1}/{args.seeds}] elapsed={elapsed:.0f}s")
        print("  Version A (numerical)...", flush=True)
        runs_a.append(_run_one(
            seed=seed, version="A_numerical", shock_text=NUMERICAL_SHOCK,
            num_rounds=args.num_rounds, shock_round=args.shock_round,
            parallel_agents=not args.no_parallel,
        ))
        print("  Version B (narrative)...", flush=True)
        runs_b.append(_run_one(
            seed=seed, version="B_narrative", shock_text=NARRATIVE_SHOCK,
            num_rounds=args.num_rounds, shock_round=args.shock_round,
            parallel_agents=not args.no_parallel,
        ))

    per_run_a = _per_run_metrics(runs_a, args.shock_round, args.post_window)
    per_run_b = _per_run_metrics(runs_b, args.shock_round, args.post_window)

    payload = {
        "config": {
            "seeds": args.seeds,
            "num_rounds": args.num_rounds,
            "shock_round": args.shock_round,
            "post_window": args.post_window,
            "numerical_shock": NUMERICAL_SHOCK,
            "narrative_shock": NARRATIVE_SHOCK,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "per_run_metrics": {
            "A_numerical": per_run_a,
            "B_narrative": per_run_b,
        },
        "aggregates": {
            "A_numerical": {
                k: dict(zip(("mean", "std"), _mean_std(v)))
                for k, v in per_run_a.items()
            },
            "B_narrative": {
                k: dict(zip(("mean", "std"), _mean_std(v)))
                for k, v in per_run_b.items()
            },
            "welch_t": {
                k: dict(zip(("t", "verdict"),
                            _welch_t(per_run_a[k], per_run_b[k])))
                for k in per_run_a
            },
        },
        "raw_runs": {
            "A_numerical": [_serialize_run(r) for r in runs_a],
            "B_narrative": [_serialize_run(r) for r in runs_b],
        },
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {out_path}")

    _print_table(per_run_a, per_run_b, args.seeds)


if __name__ == "__main__":
    main()
