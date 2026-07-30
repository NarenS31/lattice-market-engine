"""
Experiment 4: Framing x Realism (the 2x2 design)
================================================

Research question
-----------------
Does the *framing* of a supply chain shock (natural-language narrative vs. a
numeric parameter vector) change how LLM agents price it — and does that
framing effect depend on whether the shock describes a real-looking event or an
invented one?

If agents respond to shock *structure*, the framing effect should be the same
for real and fictional shocks (no interaction). If agents are partly recalling
how real events actually played out, the narrative framing should do more work
for real shocks than fictional ones (a positive interaction).

Design
------
  10 event families x 2 realism x 2 framing x N repetitions

Every family appears in all four cells (see shocks.json), so family is crossed
with both factors. Within a family, the same repetition index uses the *same
random seed* in all four cells — same market-maker walk, same warm-up book,
same agent starting state — so the shock text is the only thing that differs.
That pairing is what makes the interaction contrast well-powered per family.

    seed(family, rep) is identical across all 4 cells
    => cell differences within a family are attributable to the shock condition

The seed does NOT depend on the model, so the same family+rep runs against an
identical market on every model. Cross-model differences are therefore
attributable to the model, not to a different random draw.

Cross-model replication
-----------------------
The whole sweep runs once per model (see MODELS). Two models is a *replication*
check, not a generalisation claim: it asks whether the interaction reproduces
on a second local model, which is a much weaker question than whether it holds
across model families. Both possible answers are reportable — an effect that
appears only on one model is a real, publishable finding about model-specific
behaviour, not a failure. analyze_2x2.py --compare-models is deliberately
written not to prefer either outcome.

Cost
----
  runs = 10 families x 2 x 2 x reps x n_models
  calls = runs x ticks x n_llm_agents

  Defaults (reps=5, ticks=20, 3 LLM agents, 2 models):
    200 runs / 12,000 calls per model  ->  400 runs / 24,000 calls total.
  Measured on an M4/16GB at ~4-5 s/call that is roughly 30 hours serial. Note
  --workers buys far less than it looks: on that box 4 workers moved throughput
  only 5.2 -> 4.2 s/call (1.24x), because a single Ollama server sharing one GPU
  is the bottleneck, not client concurrency. Per-run elapsed_s in stored results
  is inflated by exactly that contention and should not be read as per-call cost.
  These are local calls: no API cost, only wall-clock. Use --workers to
  parallelise (applied per model), --reps/--ticks to shrink, and note the
  runner is resumable — completed runs are skipped on a re-run, per model.

Run
---
  python -m experiments.design_2x2 --smoke              # 1 rep, 6 ticks, plumbing
  python -m experiments.design_2x2 --reps 5             # one model (--model)
  python -m experiments.design_2x2 --reps 5 --workers 4
  python -m experiments.design_2x2 --model mistral:7b --reps 5
  python -m experiments.design_2x2 --all-models --reps 5 --workers 4
  python -m experiments.design_2x2 --all-models --estimate-only

Backend
-------
This runner is Ollama-only: --model names an Ollama tag and every LLM agent in
a run is given an explicit OllamaBackend for that tag. That is a deliberate
narrowing — results are filed per model, so the model has to be a property of
the run rather than of ambient LATTICE_BACKEND environment state.

Layout
------
  results_2x2/{model}/{framing}__{realism}/{event}__rep{NN}.json

`{model}` is the model tag with ':' and '/' replaced by '_' (llama3.1_8b), so
the path stays portable. The unmodified tag is stored in each result's "model"
field, and that field — not the directory name — is what the analysis filters
on.

Then: python3 analyze_2x2.py --compare-models
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import Order, OrderBook                                    # noqa: E402
from agents import (                                                   # noqa: E402
    Simulation, RetailAgent, InstitutionalAgent, QuantAgent, MarketMakerAgent,
)
from llm_backends import OllamaBackend                                 # noqa: E402

SHOCKS_PATH = os.path.join(_ROOT, "shocks.json")
RESULTS_DIR = os.path.join(_ROOT, "results_2x2")

DEFAULT_MODEL = "llama3.1:8b"

# Models swept by run_all_models(). Order is the reporting order; the first is
# the primary and the rest are replication checks.
MODELS = ["llama3.1:8b", "mistral:7b"]

SHOCK_TICK = 10
INITIAL_FV = 100.0
INITIAL_CASH = {"retail": 10_000.0, "institutional": 100_000.0, "quant": 50_000.0}


# ---------------------------------------------------------------------------
# Fixed-shock pulse
# ---------------------------------------------------------------------------

class FixedShockPulse:
    """Serves one specific shock from `shock_tick` onward.

    Mirrors SupplyChainPulse's (get_state, to_context) interface so Simulation
    needs no modification. The shock's `text` is placed in the
    multi_sentence_narrative slot, which is the field the agent prompt builder
    reads — so narrative and numeric conditions differ in *content only*, not
    in prompt structure. That is the intended control: we are manipulating what
    the agent is told, not how the prompt is assembled.
    """

    def __init__(self, shock: dict, shock_tick: int = SHOCK_TICK) -> None:
        self.shock = shock
        self.shock_tick = shock_tick

    def get_state(self, tick: int) -> Optional[dict]:
        if tick < self.shock_tick:
            return None
        return {
            "event_name": self.shock["event_key"],
            "shock_id": self.shock["shock_id"],
            "realism": self.shock["realism"],
            "framing": self.shock["framing"],
            "severity": self.shock["severity"],
            "affected_sectors": list(self.shock["affected_sectors"]),
            "multi_sentence_narrative": self.shock["text"],
            "tick": tick,
        }

    def to_context(self, state: Optional[dict]) -> dict:
        return state or {}


# ---------------------------------------------------------------------------
# Trajectory-recording simulation
# ---------------------------------------------------------------------------

class TrajectorySimulation(Simulation):
    """Simulation that records each agent's structured decision per tick.

    The base Simulation keeps only reasoning strings; the 2x2 analysis needs
    the actual actions too. Order-book and matching behaviour are untouched —
    this only adds recording.
    """

    def __init__(self, book: OrderBook, agents: list) -> None:
        super().__init__(book, agents)
        self.decision_log: list = []   # [tick][agent] -> dict

    def run(self, n_ticks: int, factor_pulse=None, verbose: bool = False) -> list:
        summaries = []

        for tick_idx in range(n_ticks):
            snap = json.loads(self.book.to_json())

            macro_context = None
            if factor_pulse is not None:
                macro_context = factor_pulse.to_context(factor_pulse.get_state(tick_idx))
            self.macro_log.append(macro_context or {})

            new_orders: list = []
            tick_reasoning: list = []
            tick_decisions: list = []

            for agent in self.agents:
                orders, reasoning = agent.decide_orders(snap, macro_context, tick_idx)
                tick_reasoning.append(reasoning)
                tick_decisions.append({
                    "agent": agent.name,
                    "archetype": getattr(agent, "_ARCHETYPE", None) or "market_maker",
                    "reasoning": reasoning,
                    "orders": [
                        {"side": o.side.name, "qty": o.qty, "price": o.price}
                        for o in orders
                    ],
                    "held": len(orders) == 0,
                    # A coin-flip fallback order produces held=False and is
                    # otherwise indistinguishable from a real decision; a
                    # backend failure produces held=True and looks like
                    # caution. Both are degeneracy — record them separately.
                    "forced_random": bool(getattr(agent, "_last_forced_random", False)),
                    "api_failed": bool(getattr(agent, "_last_api_failed", False)),
                })
                for o in orders:
                    new_orders.append(o)
                    self._order_map[o.order_id] = agent

            self.reasoning_log.append(tick_reasoning)
            self.decision_log.append(tick_decisions)

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
# Metrics
# ---------------------------------------------------------------------------

def _mean(vals: list) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _decision_rates(decision_log: list) -> dict:
    """Fractions of LLM-agent decisions by kind, market maker excluded.

    Three ways a "decision" can fail to be one, and they look different
    downstream, so they are counted apart:

      held          — the model chose hold (or the backend failed; see below)
      forced_random — 3 consecutive holds triggered a coin-flip order. Produces
                      held=False, so without this it counts as participation.
      api_failed    — the backend returned nothing. Produces held=True, so
                      without this it counts as a cautious hold.

    A sweep that is mostly forced_random is noise wearing a trade's clothes;
    one that is mostly api_failed is a broken pipeline. Neither is a result.
    """
    total = held = forced = failed = 0
    for tick in decision_log:
        for d in tick:
            if d["archetype"] == "market_maker":
                continue
            total += 1
            held += bool(d.get("held"))
            forced += bool(d.get("forced_random"))
            failed += bool(d.get("api_failed"))
    if not total:
        return {"hold_rate": float("nan"), "forced_random_rate": float("nan"),
                "api_failure_rate": float("nan"), "genuine_hold_rate": float("nan")}
    return {
        "hold_rate": held / total,
        "forced_random_rate": forced / total,
        "api_failure_rate": failed / total,
        # Holds that were an actual choice, not a backend failure.
        "genuine_hold_rate": (held - failed) / total,
    }


def hold_rate(decision_log: list) -> float:
    """Fraction of LLM-agent decisions that produced no order.

    Retained for backwards compatibility with runs and code that predate the
    split; prefer _decision_rates.
    """
    return _decision_rates(decision_log)["hold_rate"]


def compute_metrics(summaries: list, agents: list, wealth_log: list,
                    shock_tick: int) -> dict:
    """Per-run metrics, split around the shock tick."""
    pre = summaries[:shock_tick]
    post = summaries[shock_tick:]

    spread_pre = _mean([s.spread for s in pre])
    spread_post = _mean([s.spread for s in post])

    kyle_pre = _mean([abs(s.kyles_lambda) for s in pre if s.kyles_lambda is not None])
    kyle_post = _mean([abs(s.kyles_lambda) for s in post if s.kyles_lambda is not None])

    amihud_post = _mean([s.amihud_illiquidity for s in post])
    adverse_post = _mean([s.spread_adverse_selection for s in post])

    # Archetype divergence: dispersion of end-of-run returns across the three
    # LLM archetypes. Market maker excluded — it does not use the shock text.
    final = wealth_log[-1] if wealth_log else []
    by_arch: dict = {}
    for agent, w in zip(agents, final):
        arch = getattr(agent, "_ARCHETYPE", None)
        if arch is None:
            continue
        initial = INITIAL_CASH.get(arch)
        if initial:
            by_arch.setdefault(arch, []).append((w - initial) / initial)

    arch_returns = {a: float(np.mean(v)) for a, v in by_arch.items()}
    divergence = (
        float(np.std(list(arch_returns.values()), ddof=1))
        if len(arch_returns) > 1 else float("nan")
    )

    return {
        "spread_pre": spread_pre,
        "spread_post": spread_post,
        "spread_widening": spread_post - spread_pre,
        "kyle_lambda_pre": kyle_pre,
        "kyle_lambda_post": kyle_post,
        "kyle_lambda_delta": kyle_post - kyle_pre,
        "amihud_post": amihud_post,
        "spread_adverse_post": adverse_post,
        "archetype_divergence": divergence,
        "archetype_returns": arch_returns,
    }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def make_seed(family: str, rep: int) -> int:
    """Deterministic seed, identical across all four cells of a family+rep.

    Deliberately independent of the model: every model sees the same market for
    a given family+rep, so a cross-model difference cannot be a seeding
    artefact.
    """
    h = 0
    for ch in family:
        h = (h * 131 + ord(ch)) % (2 ** 31)
    return (h + rep * 7919) % (2 ** 31)


# One backend per model per process. Backends are stateless between calls but
# hold an HTTP client, so rebuilding one per run would be pure overhead. Keyed
# by model because a --all-models sweep touches more than one.
_BACKENDS: dict = {}


def backend_for_model(model: str) -> OllamaBackend:
    if model not in _BACKENDS:
        _BACKENDS[model] = OllamaBackend(model=model)
    return _BACKENDS[model]


def run_one(shock: dict, rep: int, n_ticks: int, shock_tick: int,
            model: str = DEFAULT_MODEL) -> dict:
    """Execute a single simulation for one shock condition on one model."""
    family = shock["matched_pair_id"]
    seed = make_seed(family, rep)

    random.seed(seed)
    np.random.seed(seed)

    # All three LLM archetypes share one backend instance, so the model is a
    # property of the run rather than of ambient env state. The market maker is
    # rule-based and takes no backend.
    backend = backend_for_model(model)
    agents = [
        RetailAgent(cash=INITIAL_CASH["retail"], name="Retail", backend=backend),
        InstitutionalAgent(cash=INITIAL_CASH["institutional"], name="Inst",
                           backend=backend),
        QuantAgent(cash=INITIAL_CASH["quant"], name="Quant", backend=backend),
        MarketMakerAgent(name="MM"),
    ]
    book = OrderBook(symbol="X2X")
    sim = TrajectorySimulation(book, agents)
    sim.seed_book(fair_value=INITIAL_FV)

    pulse = FixedShockPulse(shock, shock_tick=shock_tick)
    t0 = time.time()
    summaries = sim.run(n_ticks=n_ticks, factor_pulse=pulse)
    elapsed = time.time() - t0

    trajectory = [
        {
            "tick": s.tick,
            "clearing_price": s.clearing_price,
            "mid_price": s.mid_price,
            "spread": s.spread,
            "volume": s.volume,
            "num_trades": s.num_trades,
            "kyles_lambda": s.kyles_lambda,
            "amihud_illiquidity": s.amihud_illiquidity,
            "spread_adverse_selection": s.spread_adverse_selection,
            "spread_inventory_cost": s.spread_inventory_cost,
        }
        for s in summaries
    ]

    metrics = compute_metrics(summaries, agents, sim.wealth_log, shock_tick)
    metrics.update(_decision_rates(sim.decision_log))

    return {
        "model": model,
        "backend": backend.describe(),
        "shock_id": shock["shock_id"],
        "matched_pair_id": family,
        "event_key": shock["event_key"],
        "realism": shock["realism"],
        "framing": shock["framing"],
        "rep": rep,
        "seed": seed,
        "n_ticks": n_ticks,
        "shock_tick": shock_tick,
        "shock_text": shock["text"],
        "elapsed_s": round(elapsed, 2),
        "metrics": metrics,
        "trajectory": trajectory,
        "decisions": sim.decision_log,
        "wealth_log": sim.wealth_log,
        "agent_names": [a.name for a in agents],
        "agent_archetypes": [
            getattr(a, "_ARCHETYPE", None) or "market_maker" for a in agents
        ],
    }


def model_slug(model: str) -> str:
    """Filesystem-safe form of an Ollama tag ('llama3.1:8b' -> 'llama3.1_8b').

    Colons are legal in POSIX filenames but are displayed as '/' by the macOS
    Finder and need quoting in plenty of tooling, so they are replaced rather
    than tolerated. The unmodified tag is preserved in each result's "model"
    field, which is what the analysis reads — this slug is only a directory
    name.
    """
    return model.replace(":", "_").replace("/", "_")


def model_dir(model: str) -> str:
    return os.path.join(RESULTS_DIR, model_slug(model))


def cell_dir(model: str, framing: str, realism: str) -> str:
    return os.path.join(model_dir(model), f"{framing}__{realism}")


def run_path(model: str, shock: dict, rep: int) -> str:
    return os.path.join(
        cell_dir(model, shock["framing"], shock["realism"]),
        f"{shock['event_key']}__rep{rep:02d}.json",
    )


def _worker(payload: tuple) -> tuple:
    """Top-level so ProcessPoolExecutor can pickle it."""
    shock, rep, n_ticks, shock_tick, model, path = payload
    try:
        result = run_one(shock, rep, n_ticks, shock_tick, model=model)
    except Exception as exc:  # keep the sweep alive; record the failure
        return path, False, f"{type(exc).__name__}: {exc}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    return path, True, ""


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def load_shocks() -> list:
    with open(SHOCKS_PATH, encoding="utf-8") as fh:
        return json.load(fh)["shocks"]


def preflight(model: str = DEFAULT_MODEL) -> bool:
    """Confirm this model actually returns a decision.

    Without this, a misconfigured or unpulled model yields 80 runs of pure
    HOLDs — a pipeline failure that looks exactly like a null result once
    analysed. Run per model: `ollama pull mistral:7b` being forgotten is
    precisely the failure this catches, and it would otherwise show up as
    "the interaction did not replicate on mistral".
    """
    probe = RetailAgent(name="preflight", backend=backend_for_model(model))
    backend = probe.backend
    print(f"\n  Preflight: probing {backend.describe()} ...", end="", flush=True)
    # NOTE: this probes the raw backend, deliberately bypassing decide()'s
    # coin-flip fallback — a fallback order would make a dead backend look
    # alive. The in-sweep degeneracy checks cover the fallback itself.
    decision = backend.get_decision(
        probe._SYSTEM_PROMPT,
        "Market Tick 0\nMacro environment: Markets are calm.\n\n"
        "Your current state:\n  Cash: $10000.00\n  Position: 0.0 shares\n\n"
        "Market data:\n  Reference price: $100.00\n  Best bid: $99.95  |  Best ask: $100.05\n\n"
        'Respond with ONLY a JSON object, no other text:\n'
        '{"action": "buy" or "sell" or "hold", "quantity": <number of shares>, '
        '"limit_price": <price as number, or null if hold>, "reasoning": "<one sentence>"}',
    )
    print(" OK" if decision is not None else " FAILED")
    return decision is not None


def report_sweep_health(model: str) -> dict:
    """Post-sweep degeneracy check over this model's completed runs.

    Preflight can only prove the backend answers once; it cannot see the
    coin-flip fallback, which by construction only fires after three
    consecutive holds mid-run. This is where that shows up.
    """
    import glob

    paths = [p for p in glob.glob(os.path.join(model_dir(model), "*", "*.json"))
             if os.path.basename(p) != "manifest.json"]
    rates: dict = {"hold_rate": [], "forced_random_rate": [], "api_failure_rate": []}
    legacy = 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                m = json.load(fh).get("metrics", {})
        except (json.JSONDecodeError, OSError):
            continue
        if "forced_random_rate" not in m:
            legacy += 1
        for k in rates:
            v = m.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rates[k].append(float(v))

    if not any(rates.values()):
        return {}

    means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in rates.items()}
    print(f"\n  Decision health for {model} ({len(paths)} runs)")
    print(f"    hold rate            : {means['hold_rate']:.1%}")
    print(f"    forced coin-flips    : {means['forced_random_rate']:.1%}")
    print(f"    backend failures     : {means['api_failure_rate']:.1%}")
    if legacy:
        print(f"    ({legacy} run(s) predate this telemetry and are excluded)")

    fr, af, hr = (means["forced_random_rate"], means["api_failure_rate"],
                  means["hold_rate"])
    if not np.isnan(fr) and fr > 0.20:
        print("\n    ** STOP: over a fifth of decisions are coin-flip fallbacks.")
        print("       These are random orders, not decisions. They enter the")
        print("       order book and the metrics as if they were real trades,")
        print("       so a sweep dominated by them measures the RNG, not the")
        print("       models. Do not analyse this.")
    elif not np.isnan(fr) and fr > 0.05:
        print("\n    ** NOTE: elevated coin-flip rate; some trades are random.")
    if not np.isnan(af) and af > 0.05:
        print("\n    ** WARNING: backend failures present. These count as holds,")
        print("       which understates participation and can look like caution.")
    if not np.isnan(hr) and hr > 0.5:
        print("\n    ** STOP: over half of decisions produced no order.")

    return means


def run_experiment4(
    reps: int = 5,
    n_ticks: int = 20,
    shock_tick: int = SHOCK_TICK,
    workers: int = 1,
    resume: bool = True,
    estimate_only: bool = False,
    skip_preflight: bool = False,
    model: str = DEFAULT_MODEL,
) -> dict:
    shocks = load_shocks()
    n_llm_agents = 3

    tasks = []
    skipped = 0
    for shock in shocks:
        for rep in range(reps):
            path = run_path(model, shock, rep)
            if resume and os.path.exists(path):
                skipped += 1
                continue
            tasks.append((shock, rep, n_ticks, shock_tick, model, path))

    total_runs = len(shocks) * reps
    calls = len(tasks) * n_ticks * n_llm_agents

    print("=" * 78)
    print(f"Experiment 4: Framing x Realism (2x2) — model {model}")
    print("=" * 78)
    print(f"  model            : {model}  -> results_2x2/{model_slug(model)}/")
    print(f"  shocks           : {len(shocks)}  ({len(shocks)//4} families x 4 cells)")
    print(f"  reps per shock   : {reps}")
    print(f"  ticks per run    : {n_ticks}   (shock fires at tick {shock_tick})")
    print(f"  total runs       : {total_runs}")
    print(f"  already complete : {skipped}")
    print(f"  to execute       : {len(tasks)}")
    print(f"  LLM calls needed : {calls:,}  ({n_llm_agents} agents x {n_ticks} ticks)")
    print(f"  workers          : {workers}")
    if calls:
        print(f"  rough wall-clock : {calls * 7 / 3600 / max(1, workers):.1f} h "
              f"at 7 s/call")
    print("=" * 78)

    if estimate_only:
        print("\n  --estimate-only: nothing executed.")
        return {"model": model, "planned": len(tasks), "calls": calls}

    if tasks and not skip_preflight and not preflight(model):
        print(f"\n  ABORTED — {model} returned no usable decision.")
        print("  Running anyway would produce a sweep of all-HOLD runs, which")
        print("  analyses as a clean null result and is indistinguishable from")
        print("  a genuine 'no interaction' finding — and in a cross-model")
        print("  comparison it would masquerade as 'the effect is specific to")
        print("  the other model'. Fix the backend first:")
        print("    • start the Ollama server: `ollama serve`")
        print(f"    • pull the model:           `ollama pull {model}`")
        print("    • non-default host:         set LATTICE_OLLAMA_HOST")
        print("  Override with --skip-preflight if you know what you are doing.")
        return {"model": model, "planned": 0, "calls": 0, "aborted": "preflight"}

    if not tasks:
        print(f"\n  Nothing to do — all runs present for {model}. "
              f"Delete results_2x2/{model_slug(model)}/ to re-run.")
        return {"model": model, "planned": 0, "calls": 0}

    t0 = time.time()
    done = failed = 0
    failures = []

    if workers > 1:
        # Process-based: each run needs its own global RNG state, so threads
        # would break the per-run seeding guarantee.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                path, ok, err = fut.result()
                done += 1
                if not ok:
                    failed += 1
                    failures.append((path, err))
                print(f"\r  {done}/{len(tasks)} runs  ({failed} failed)",
                      end="", flush=True)
    else:
        for t in tasks:
            path, ok, err = _worker(t)
            done += 1
            if not ok:
                failed += 1
                failures.append((path, err))
            print(f"\r  {done}/{len(tasks)} runs  ({failed} failed)",
                  end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n\n  Completed in {elapsed/60:.1f} min. {failed} failures.")
    for path, err in failures[:10]:
        print(f"    FAIL {os.path.basename(path)}: {err}")

    report_sweep_health(model)

    manifest = {
        "model": model, "reps": reps, "n_ticks": n_ticks, "shock_tick": shock_tick,
        "n_shocks": len(shocks), "total_runs": total_runs,
        "executed": len(tasks), "failed": failed,
        "elapsed_s": round(elapsed, 1),
    }
    # Manifest lives beside that model's cells, so each sweep documents itself
    # rather than overwriting the previous model's record.
    os.makedirs(model_dir(model), exist_ok=True)
    with open(os.path.join(model_dir(model), "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n  Results → {model_dir(model)}/")
    print(f"  Next: python3 analyze_2x2.py --model {model}")
    return manifest


def run_all_models(
    models: Optional[list] = None,
    reps: int = 5,
    n_ticks: int = 20,
    shock_tick: int = SHOCK_TICK,
    workers: int = 1,
    resume: bool = True,
    estimate_only: bool = False,
    skip_preflight: bool = False,
) -> dict:
    """Run the full sweep once per model, for cross-model replication.

    Sweeps run sequentially — they contend for the same local Ollama server, so
    interleaving two models would just thrash the GPU and make per-model
    timings meaningless. `workers` still applies *within* each model's sweep,
    so parallelism is unchanged; it is the models that are serialised.

    A model whose preflight fails is skipped, not fatal: one working model's
    results are worth keeping, and the skip is reported loudly at the end so it
    can never be mistaken for a model that ran and produced a null.
    """
    models = list(models if models is not None else MODELS)

    print("=" * 78)
    print(f"CROSS-MODEL SWEEP — {len(models)} models: {', '.join(models)}")
    print("=" * 78)
    print("  Each model runs the complete 2x2 design on identical seeds, so a")
    print("  difference between models is attributable to the model.")
    print(f"  Total budget: {len(models)} x {len(load_shocks()) * reps} runs "
          f"= {len(models) * len(load_shocks()) * reps} runs "
          f"({len(models) * len(load_shocks()) * reps * n_ticks * 3:,} calls, "
          f"local — no API cost).")

    t0 = time.time()
    manifests = {}
    for i, model in enumerate(models, 1):
        print(f"\n\n{'#' * 78}")
        print(f"# MODEL {i}/{len(models)}: {model}")
        print(f"{'#' * 78}")
        manifests[model] = run_experiment4(
            reps=reps, n_ticks=n_ticks, shock_tick=shock_tick,
            workers=workers, resume=resume, estimate_only=estimate_only,
            skip_preflight=skip_preflight, model=model,
        )

    elapsed = time.time() - t0

    print("\n" + "=" * 78)
    print("CROSS-MODEL SWEEP — ESTIMATE" if estimate_only
          else "CROSS-MODEL SWEEP COMPLETE")
    print("=" * 78)
    aborted = []
    for model, m in manifests.items():
        if m.get("aborted"):
            aborted.append(model)
            print(f"  {model:<20} SKIPPED ({m['aborted']})")
        elif estimate_only:
            print(f"  {model:<20} {m.get('planned', 0):>4} runs planned, "
                  f"{m.get('calls', 0):,} calls")
        else:
            print(f"  {model:<20} {m.get('executed', 0):>4} runs executed, "
                  f"{m.get('failed', 0)} failed")
    if not estimate_only:
        print(f"\n  Total wall-clock: {elapsed/60:.1f} min")

    if aborted:
        print(f"\n  ** {len(aborted)} model(s) did not run: {', '.join(aborted)}")
        print("     Do NOT compare models until these are fixed and re-run. A")
        print("     missing model is not a negative replication result.")

    if not estimate_only:
        print("\n  Next: python3 analyze_2x2.py --compare-models")
    return manifests


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the 2x2 framing x realism sweep.")
    ap.add_argument("--reps", type=int, default=5,
                    help="Repetitions per shock (default 5).")
    ap.add_argument("--ticks", type=int, default=20,
                    help="Ticks per run (default 20).")
    ap.add_argument("--shock-tick", type=int, default=SHOCK_TICK,
                    help=f"Tick at which the shock fires (default {SHOCK_TICK}).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel processes (default 1; processes, not threads, "
                         "so per-run seeding stays deterministic).")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-run even if a result file already exists.")
    ap.add_argument("--estimate-only", action="store_true",
                    help="Print the cost estimate and exit.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="Skip the backend probe (not recommended).")
    ap.add_argument("--smoke", action="store_true",
                    help="1 rep, 6 ticks — plumbing check only.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama model tag for all LLM agents "
                         f"(default {DEFAULT_MODEL}).")
    ap.add_argument("--all-models", action="store_true",
                    help=f"Run the full sweep once per model in {MODELS} "
                         f"(sequentially; --workers still applies per model). "
                         f"Overrides --model.")
    args = ap.parse_args(argv)

    reps, ticks, shock_tick = args.reps, args.ticks, args.shock_tick
    if args.smoke:
        reps, ticks, shock_tick = 1, 6, 3

    common = dict(
        reps=reps, n_ticks=ticks, shock_tick=shock_tick,
        workers=args.workers, resume=not args.no_resume,
        estimate_only=args.estimate_only, skip_preflight=args.skip_preflight,
    )

    if args.all_models:
        run_all_models(models=MODELS, **common)
    else:
        run_experiment4(model=args.model, **common)
    return 0


if __name__ == "__main__":
    sys.exit(main())
