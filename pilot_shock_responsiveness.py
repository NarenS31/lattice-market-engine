"""
pilot_shock_responsiveness.py — do agent decisions actually respond to the shock?
================================================================================

Why this exists
---------------
The 2x2 design asks whether shock *framing* changes how agents price a shock.
That question is only answerable if the shock text reaches the decision at all.

It did not. The original archetype system prompts handed each agent a complete,
deterministic, price-only rule — institutional: "if current price is below the
average of the last 3 clearing prices, buy; if above, sell"; quant: "any upward
movement means buy, any downward movement means sell"; retail: buy the dip or
sell before it gets worse. None of them referenced the macro context. The shock
narrative was in the user prompt, but nothing told the agent it was relevant.

That matters because of how the design is seeded. At the shock tick all four
cells share a seed and therefore face an *identical* order book. An agent
following its prescribed rule returns the same decision in every cell, so the
four cells evolve identically and the interaction is zero by construction. The
experiment could only ever have measured how far the model *drifts* from the
rule it was given — sampling noise, not response to framing.

The fix reframes the price rule as each archetype's DEFAULT and tells the agent
that macro context is decision-relevant and may justify deviating. It stops
short of "react to the shock regardless of price", which would just relocate
the confound: the agent would then respond to the presence of a shock rather
than to its content, and the framing contrast would still be uninformative.

What this measures
------------------
Identical market path in every condition; only the macro context changes. That
is the same control the sweep relies on, so any variation in the decision is
attributable to the shock text and nothing else.

  responsiveness = does (direction, quantity) vary across shock conditions
                   when price, position and tick are all held fixed?

Baseline (pre-fix) prompts are embedded below verbatim, so both arms can be
re-measured from this one file rather than from git history.

Reading the result
------------------
A flat baseline arm and a responsive revised arm means the prompts were the
binding constraint. Two flat arms means the fix was insufficient and the sweep
still cannot detect anything — do not run it. Two responsive arms would mean
the original diagnosis was wrong.

Note what this does NOT establish: that agents respond *differently* to
narrative vs numeric framing, which is the actual experimental hypothesis.
It only establishes that the shock reaches the decision at all. Responsiveness
is necessary for the 2x2 to be meaningful, not sufficient for it to find
anything.

Run
---
  python3 pilot_shock_responsiveness.py                      # both arms, both models
  python3 pilot_shock_responsiveness.py --variant revised
  python3 pilot_shock_responsiveness.py --models llama3.1:8b --ticks 2
  python3 pilot_shock_responsiveness.py --out pilot_shock.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Optional

import agents as agents_mod
from agents import RetailAgent, InstitutionalAgent, QuantAgent
from llm_backends import OllamaBackend

_HERE = os.path.dirname(os.path.abspath(__file__))
SHOCKS_PATH = os.path.join(_HERE, "shocks.json")

DEFAULT_MODELS = ["llama3.1:8b", "mistral:7b"]


# ---------------------------------------------------------------------------
# Baseline (pre-fix) prompts, verbatim
# ---------------------------------------------------------------------------
# Kept literal so the "before" arm is reproducible without checking out an old
# revision. If these are edited, the comparison stops meaning anything.

BASELINE_SYSTEM = {
    "retail": (
        "You are a retail investor making real-time trading decisions in a simulated stock market. "
        "Your cognitive archetype: you are highly loss-averse (you dread losses far more than you "
        "enjoy gains), you rely on recent price memory rather than deep analysis, you get anxious "
        "when volatility is high, and you tend to follow the most recent price direction with a "
        "noisy, emotionally-coloured signal. You trade small quantities — 1 to 5 shares per tick. "
        "Your decisions feel driven by gut instinct and recent price memory.\n\n"
        "You feel uncomfortable holding cash when prices are moving. If the price has changed "
        "from the previous tick, you have a strong urge to either buy the dip or sell before it "
        "gets worse. You rarely hold for more than 2 ticks in a row.\n\n"
        "Always respond with ONLY a JSON object — no explanation, no preamble, no markdown. "
        'Exact format: {"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one concise sentence>"}'
    ),
    "institutional": (
        "You are an institutional fund manager making real-time trading decisions in a simulated "
        "stock market. Your cognitive archetype: disciplined and systematic, you apply VWAP "
        "mean-reversion logic — buy when price is below the recent volume-weighted average, sell "
        "when it is above. You have moderate risk aversion and trade in larger blocks — 10 to 30 "
        "shares per tick. You read the order book depth for secondary signals but your primary "
        "anchor is always the VWAP mean-reversion thesis.\n\n"
        "You must place an order every tick unless your position is already at maximum. If "
        "current price is below the average of the last 3 clearing prices, buy. If above, sell. "
        "You do not hold unless there is no price history at all.\n\n"
        "Always respond with ONLY a JSON object — no explanation, no preamble, no markdown. "
        'Exact format: {"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one concise sentence>"}'
    ),
    "quant": (
        "You are a quantitative momentum trader making real-time trading decisions in a simulated "
        "stock market. Your cognitive archetype: you follow price momentum relentlessly. "
        "You have very low risk aversion and act with conviction on clear signals. You trade "
        "medium-sized blocks — 5 to 15 shares per tick. Speed and signal conviction define your "
        "archetype.\n\n"
        "You trade on every tick where there is price history. Any upward movement means buy, "
        "any downward movement means sell, flat price means small buy. You never hold when you "
        "have data.\n\n"
        "Always respond with ONLY a JSON object — no explanation, no preamble, no markdown. "
        'Exact format: {"action": "buy"|"sell"|"hold", "quantity": <number>, '
        '"limit_price": <number or null>, "reasoning": "<one concise sentence>"}'
    ),
}

BASELINE_CLOSING = (
    "IMPORTANT: You must return buy or sell in almost all cases. Only return hold "
    "if you have absolutely no information to act on. The market needs your participation."
)

_CLASSES = {"retail": RetailAgent, "institutional": InstitutionalAgent, "quant": QuantAgent}


class prompt_variant:
    """Context manager swapping the archetype prompts for one arm of the pilot."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self._saved_system: dict = {}
        self._saved_closing: Optional[str] = None

    def __enter__(self) -> "prompt_variant":
        if self.variant != "baseline":
            return self
        for arch, cls in _CLASSES.items():
            self._saved_system[arch] = cls._SYSTEM_PROMPT
            cls._SYSTEM_PROMPT = BASELINE_SYSTEM[arch]
        self._saved_closing = agents_mod._CLOSING_INSTRUCTION
        agents_mod._CLOSING_INSTRUCTION = BASELINE_CLOSING
        return self

    def __exit__(self, *exc) -> None:
        for arch, prompt in self._saved_system.items():
            _CLASSES[arch]._SYSTEM_PROMPT = prompt
        if self._saved_closing is not None:
            agents_mod._CLOSING_INSTRUCTION = self._saved_closing


# ---------------------------------------------------------------------------
# Fixed market path — identical in every condition, by design
# ---------------------------------------------------------------------------

PRICES = [99.40, 98.10, 96.75, 95.90]
VOLS = [42.0, 55.0, 61.0, 58.0]
POSITIONS = [("short", -30.0), ("long", 45.0)]

INITIAL_CASH = {"retail": 10_000.0, "institutional": 100_000.0, "quant": 50_000.0}


def load_conditions() -> list:
    with open(SHOCKS_PATH, encoding="utf-8") as fh:
        shocks = json.load(fh)["shocks"]

    def pick(fr: str, rl: str) -> dict:
        return next(s for s in shocks if s["framing"] == fr and s["realism"] == rl)

    conds = [("calm", {})]
    for fr in ("narrative", "numeric"):
        s = pick(fr, "real")
        conds.append((f"{fr}/real", {
            "event_name": s["event_key"], "severity": s["severity"],
            "affected_sectors": list(s["affected_sectors"]),
            "multi_sentence_narrative": s["text"],
        }))
    return conds


def snapshot(tick: int) -> dict:
    p = PRICES[tick]
    spread = 0.12 + 0.05 * tick
    bid, ask = round(p - spread / 2, 2), round(p + spread / 2, 2)
    hist = [{"clearing_price": q, "volume": v}
            for q, v in zip([100.2, 99.9] + PRICES[:tick + 1],
                            [30.0, 35.0] + VOLS[:tick + 1])]
    return {
        "mid_price": round((bid + ask) / 2, 2),
        "last_clearing_price": p,
        "order_book": {"best_bid": bid, "best_ask": ask,
                       "bids": [{"price": bid, "qty": 12.0}],
                       "asks": [{"price": ask, "qty": 12.0}]},
        "tick_history": hist,
    }


def make_agent(arch: str, pos: float):
    ag = _CLASSES[arch](cash=INITIAL_CASH[arch], name=arch[:1].upper())
    ag.position = pos
    # Accumulated P&L, as a position carried into a decline would show.
    ag._wealth_history.clear()
    base = ag.cash + pos * 100.0
    for d in (0.0, -180.0, -410.0):
        ag._wealth_history.append(base + d)
    return ag


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def signed_qty(action: Optional[str], qty) -> Optional[float]:
    """Direction and size as one number: buy +q, sell -q, hold 0."""
    if action == "hold":
        return 0.0
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return None
    if action == "buy":
        return q
    if action == "sell":
        return -q
    return None


def run_arm(variant: str, model: str, conditions: list, archetypes: list,
            n_ticks: int, on_progress=None) -> list:
    """One (variant, model) arm: every archetype x position x condition x tick."""
    rows = []
    with prompt_variant(variant):
        backend = OllamaBackend(model=model)
        for arch in archetypes:
            for pos_name, pos in POSITIONS:
                for cond_name, macro in conditions:
                    for tick in range(n_ticks):
                        ag = make_agent(arch, pos)
                        prompt = ag._build_user_prompt(snapshot(tick), macro, 10 + tick)
                        t0 = time.time()
                        d = backend.get_decision(ag._SYSTEM_PROMPT, prompt)
                        el = time.time() - t0
                        action = None if d is None else str(d.get("action", "?")).lower()
                        qty = None if d is None else d.get("quantity")
                        rows.append({
                            "variant": variant, "model": model, "archetype": arch,
                            "position": pos_name, "condition": cond_name, "tick": tick,
                            "action": action, "quantity": qty,
                            "signed": signed_qty(action, qty),
                            "parsed": d is not None,
                            "seconds": round(el, 1),
                            "reasoning": "" if d is None else str(d.get("reasoning", ""))[:200],
                        })
                        if on_progress:
                            on_progress()
    return rows


def responsiveness(rows: list, conditions: list) -> dict:
    """How much does the decision move across conditions, holding all else fixed?

    Each (archetype, position, tick) is a cell in which price, inventory and
    history are identical and ONLY the shock text differs. Within a cell:

      direction_varies — the buy/sell/hold choice is not the same everywhere
      signed_spread    — max minus min signed quantity, i.e. how far apart the
                         most bullish and most bearish conditions ended up
      shock_vs_calm    — mean signed quantity under shock minus under calm;
                         sign and size of the shift the shock produces

    Reported as a mean over cells. Zero on all three is the degenerate case the
    2x2 cannot survive: the shock text changed nothing.
    """
    cond_names = [c[0] for c in conditions]
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["archetype"], r["position"], r["tick"]), {})[r["condition"]] = r

    n_cells = 0
    n_dir_varies = 0
    spreads = []
    shifts = []
    for _key, by_cond in cells.items():
        vals = {c: by_cond[c]["signed"] for c in cond_names
                if c in by_cond and by_cond[c]["signed"] is not None}
        acts = {c: by_cond[c]["action"] for c in cond_names if c in by_cond}
        if len(vals) < 2:
            continue
        n_cells += 1
        if len(set(acts.values())) > 1:
            n_dir_varies += 1
        spreads.append(max(vals.values()) - min(vals.values()))
        shocked = [v for c, v in vals.items() if c != "calm"]
        if "calm" in vals and shocked:
            shifts.append(statistics.mean(shocked) - vals["calm"])

    return {
        "n_cells": n_cells,
        "direction_varies_frac": (n_dir_varies / n_cells) if n_cells else float("nan"),
        "mean_signed_spread": statistics.mean(spreads) if spreads else float("nan"),
        "mean_shock_vs_calm": statistics.mean(shifts) if shifts else float("nan"),
        "parse_rate": (sum(1 for r in rows if r["parsed"]) / len(rows)) if rows else float("nan"),
    }


def _fmt(v: float, w: int = 9) -> str:
    import math
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'n/a':>{w}}"
    return f"{v:>{w}.2f}"


def report(summary: dict, models: list, variants: list, conditions: list) -> None:
    print("\n" + "=" * 84)
    print("  SHOCK RESPONSIVENESS — decision variation across conditions")
    print("=" * 84)
    print("  Price, inventory and tick are identical within each cell; only the")
    print("  shock text differs. Movement here is caused by the shock text.\n")
    print(f"    {'model':<16} {'variant':<10} {'dir varies':>11} "
          f"{'spread':>9} {'shock-calm':>11} {'parse':>7}")
    for model in models:
        for variant in variants:
            s = summary.get((variant, model))
            if not s:
                continue
            dv = s["direction_varies_frac"]
            print(f"    {model:<16} {variant:<10} "
                  f"{dv:>10.0%} {_fmt(s['mean_signed_spread'])} "
                  f"{_fmt(s['mean_shock_vs_calm'], 11)} {s['parse_rate']:>6.0%}")

    print("\n  dir varies  : share of cells where the buy/sell/hold choice changed")
    print("  spread      : mean gap between most bullish and most bearish condition")
    print("  shock-calm  : mean signed-quantity shift from calm to shocked")

    if len(variants) < 2:
        return

    print("\n" + "-" * 84)
    print("  BEFORE / AFTER")
    print("-" * 84)
    for model in models:
        b = summary.get(("baseline", model))
        r = summary.get(("revised", model))
        if not b or not r:
            continue
        print(f"\n    {model}")
        print(f"      direction varies : {b['direction_varies_frac']:>6.0%}  ->  "
              f"{r['direction_varies_frac']:.0%}")
        print(f"      signed spread    : {b['mean_signed_spread']:>6.2f}  ->  "
              f"{r['mean_signed_spread']:.2f}")
        print(f"      shock vs calm    : {b['mean_shock_vs_calm']:>6.2f}  ->  "
              f"{r['mean_shock_vs_calm']:.2f}")

    print("\n" + "=" * 84)
    print("  READING")
    print("=" * 84)
    flat_before = all(
        summary[("baseline", m)]["mean_signed_spread"] < 1.0
        for m in models if ("baseline", m) in summary
    )
    responsive_after = all(
        summary[("revised", m)]["mean_signed_spread"] >= 1.0
        for m in models if ("revised", m) in summary
    )
    if flat_before and responsive_after:
        print("""
  The baseline arm is flat and the revised arm is not: the price-only prompts
  were the binding constraint, and the shock text now reaches the decision.
  That makes the 2x2 answerable. It does NOT predict which way it will come
  out — responsiveness to a shock is not the same as sensitivity to how the
  shock is framed, which is the actual hypothesis.""")
    elif flat_before and not responsive_after:
        print("""
  ** Both arms are flat. The prompt change was not sufficient. The shock still
  does not move the decision, so the 2x2 would measure sampling noise and
  return a null by construction. Do not run the sweep on this. Diagnose
  further before spending the compute.""")
    elif not flat_before:
        print("""
  ** The baseline arm is NOT flat, which contradicts the original diagnosis.
  Re-examine before trusting either arm — the earlier single-snapshot probe
  may have hit an unrepresentative market state.""")
    print("""
  Caveats: one market path, one shock family, few ticks. This is a plumbing
  check on whether the shock is wired into the decision — it is not evidence
  about the size or direction of any framing effect.""")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Pilot: does the shock reach the decision?")
    ap.add_argument("--variant", choices=["baseline", "revised", "both"], default="both",
                    help="Which prompt arm(s) to measure (default both).")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"Ollama models (default {' '.join(DEFAULT_MODELS)}).")
    ap.add_argument("--ticks", type=int, default=2,
                    help="Ticks per condition (default 2).")
    ap.add_argument("--archetypes", nargs="+",
                    default=["retail", "institutional", "quant"],
                    help="Archetypes to probe.")
    ap.add_argument("--out", default="pilot_shock_responsiveness.json",
                    help="Where to write the raw rows.")
    args = ap.parse_args(argv)

    variants = ["baseline", "revised"] if args.variant == "both" else [args.variant]
    conditions = load_conditions()
    n_ticks = min(args.ticks, len(PRICES))

    total = (len(variants) * len(args.models) * len(args.archetypes)
             * len(POSITIONS) * len(conditions) * n_ticks)
    print("=" * 84)
    print("  PILOT — shock responsiveness before/after the prompt fix")
    print("=" * 84)
    print(f"  models     : {', '.join(args.models)}")
    print(f"  variants   : {', '.join(variants)}")
    print(f"  archetypes : {', '.join(args.archetypes)}")
    print(f"  conditions : {', '.join(c[0] for c in conditions)}")
    print(f"  positions  : {', '.join(p[0] for p in POSITIONS)}   ticks: {n_ticks}")
    print(f"  LLM calls  : {total}")
    print("=" * 84)

    done = {"n": 0}
    t0 = time.time()

    def tick_progress() -> None:
        done["n"] += 1
        if done["n"] % 10 == 0:
            el = (time.time() - t0) / 60
            rate = el / done["n"]
            print(f"    {done['n']}/{total}  ({el:.1f} min, "
                  f"~{rate * (total - done['n']):.0f} min left)", flush=True)

    all_rows: list = []
    summary: dict = {}
    for variant in variants:
        for model in args.models:
            print(f"\n  {variant} / {model}")
            rows = run_arm(variant, model, conditions, args.archetypes,
                           n_ticks, on_progress=tick_progress)
            all_rows.extend(rows)
            summary[(variant, model)] = responsiveness(rows, conditions)
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(all_rows, fh, indent=1)

    report(summary, args.models, variants, conditions)
    print(f"\n  Raw rows → {args.out}  ({len(all_rows)} decisions, "
          f"{(time.time()-t0)/60:.1f} min)")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
