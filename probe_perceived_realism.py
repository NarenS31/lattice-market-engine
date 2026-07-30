#!/usr/bin/env python3
"""
Mediator probe — measure PERCEIVED realism (M) directly, per existing run.

Causal-mediation Step 1. The 2x2 design assumes agents perceive "real" vs
"fictional" shocks as intended; this was never measured. This module adds a
distinct, lightweight follow-up call that asks each agent — replaying the shock
scenario it was actually given — whether it believes that scenario is real or
hypothetical. The answer is the mediator M (binary) per decision record, joined
to the already-collected behavioral data by run id.

This does NOT re-run any market simulation. It reuses the shock/rep/model grid
already in results_2x2/ and only adds the probe call. The follow-up is sent as a
SEPARATE call (not folded into the trading-decision prompt) so it cannot
contaminate the decisions — which are, in any case, already collected.

Design notes
------------
* Persona is preserved but the JSON-forcing tail of each archetype system prompt
  is stripped, so the probe returns a bare one-word answer instead of a trade.
* The probe input is (archetype persona, shock_text). Order-book microstructure
  is deliberately excluded: perceived realism is a property of the scenario, not
  of the current spread. A consequence is that within one run the post-shock
  ticks share an identical probe prompt per archetype, so probing every tick is
  repeated sampling of M under the decision temperature (0.7).
* Pre-shock ticks carry no shock scenario (macro = "normal baseline"), identical
  across real and fictional runs, so they are uninformative for the manipulation
  check. --scope defaults to post-shock only. Use --scope all to match the
  literal 9,600-record budget.

Outputs one file per run, keyed to the original run id:
    results_2x2_mediator/{model_slug}/{shock_id}__rep{NN}.json

Usage:
    python3 probe_perceived_realism.py --benchmark 12      # time a few calls; write nothing
    python3 probe_perceived_realism.py --model llama3.1:8b  # probe one model
    python3 probe_perceived_realism.py --all-models         # probe both, sequentially
    python3 probe_perceived_realism.py --report             # manipulation check from results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict
from typing import Optional

import agents
from llm_backends import OllamaBackend

try:
    from experiments.design_2x2 import model_slug, DEFAULT_MODEL
except Exception:  # pragma: no cover - fallback if import path differs
    DEFAULT_MODEL = "llama3.1:8b"
    def model_slug(m: str) -> str:
        return m.replace(":", "_").replace(".", "").replace("/", "_")

RESULTS_DIR = "results_2x2"
MEDIATOR_DIR = "results_2x2_mediator"
ALL_MODELS = ["llama3.1:8b", "mistral:7b"]
LLM_ARCHETYPES = ("retail", "institutional", "quant")

# The fixed follow-up prompt (verbatim from the design spec).
FOLLOW_UP = (
    "Separately from your trading decision: do you believe the scenario just "
    "described refers to an event that has actually occurred in the real world, "
    "or a hypothetical/synthetic one? Answer with exactly one word: real or "
    "hypothetical."
)
PROBE_VERSION = "v1"

_ARCH_CLASS = {
    "retail": agents.RetailAgent,
    "institutional": agents.InstitutionalAgent,
    "quant": agents.QuantAgent,
}


def _persona(archetype: str) -> str:
    """Archetype system prompt with its JSON-forcing tail removed."""
    full = _ARCH_CLASS[archetype]._SYSTEM_PROMPT
    cut = full.split("Always respond with ONLY a JSON object")[0].strip()
    return cut


def build_probe_prompt(shock_text: str) -> str:
    return (
        "The following macro scenario was presented to you as the current "
        f"market environment:\n\n\"{shock_text}\"\n\n{FOLLOW_UP}"
    )


def classify(raw: str) -> str:
    """Map a raw model answer to real / hypothetical / unclear."""
    t = (raw or "").strip().lower()
    if not t:
        return "unclear"
    # tokenise; check the first informative word, then fall back to containment
    toks = re.findall(r"[a-z]+", t)
    for w in toks:
        if w in ("hypothetical", "synthetic", "fictional", "hypothetically"):
            return "hypothetical"
        if w in ("real", "actual", "actually", "genuine"):
            return "real"
    if "hypothetical" in t or "synthetic" in t or "fictional" in t:
        return "hypothetical"
    if "real" in t:
        return "real"
    return "unclear"


def _make_backend(model: str) -> OllamaBackend:
    # max_attempts=1: a probe is a single lightweight call, no JSON retry loop.
    # timeout=300: a call that lands right after a model swap (Ollama holds one
    # model in VRAM under -np 1) can be slow; give it room before giving up.
    return OllamaBackend(model=model, max_tokens=8, temperature=0.7,
                         json_mode=False, max_attempts=1, timeout=300.0)


def _one_probe(backend: OllamaBackend, archetype: str, shock_text: str,
               retries: int = 1) -> tuple:
    """Return (perceived, raw, elapsed_s). Retries once on a transport error."""
    system = _persona(archetype)
    user = build_probe_prompt(shock_text)
    t0 = time.time()
    last = ""
    for attempt in range(retries + 1):
        try:
            raw = backend._one_call(system, user)
            return classify(raw), (raw or "").strip()[:120], time.time() - t0
        except Exception as exc:  # transport / server error
            last = f"{type(exc).__name__}: {exc}"
    return "error", last, time.time() - t0


def _run_files(model: str) -> list:
    slug = model_slug(model)
    return sorted(
        p for p in glob.glob(os.path.join(RESULTS_DIR, slug, "*", "*.json"))
        if os.path.basename(p) != "manifest.json"
    )


def _mediator_path(model: str, shock_id: str, rep: int) -> str:
    return os.path.join(MEDIATOR_DIR, model_slug(model),
                        f"{shock_id}__rep{rep:02d}.json")


def _target_records(run: dict, scope: str) -> list:
    """[(tick, archetype)] to probe for this run under the given scope."""
    st = run.get("shock_tick", 10)
    out = []
    for tick_idx, tick in enumerate(run.get("decisions", [])):
        if scope == "postshock" and tick_idx < st:
            continue
        for a in tick:
            arch = a.get("archetype")
            if arch in LLM_ARCHETYPES:
                out.append((tick_idx, arch))
    return out


def count_probes(models: list, scope: str) -> dict:
    out = {}
    for m in models:
        n = 0
        for p in _run_files(m):
            try:
                run = json.load(open(p, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            n += len(_target_records(run, scope))
        out[m] = n
    return out


def probe_model(model: str, scope: str = "postshock", resume: bool = True,
                limit: Optional[int] = None) -> dict:
    backend = _make_backend(model)
    files = _run_files(model)
    os.makedirs(os.path.join(MEDIATOR_DIR, model_slug(model)), exist_ok=True)

    done_runs = 0
    done_probes = 0
    t0 = time.time()
    print(f"\n  Probing {model}  (scope={scope}, {len(files)} runs)")

    for p in files:
        run = json.load(open(p, encoding="utf-8"))
        out_path = _mediator_path(model, run["shock_id"], run["rep"])
        if resume and os.path.exists(out_path):
            done_runs += 1
            continue

        targets = _target_records(run, scope)
        probes = []
        for (tick_idx, arch) in targets:
            perceived, raw, _ = _one_probe(backend, arch, run["shock_text"])
            probes.append({"tick": tick_idx, "archetype": arch,
                           "perceived": perceived, "raw": raw})
            done_probes += 1
            if limit is not None and done_probes >= limit:
                break

        record = {
            "model": model, "shock_id": run["shock_id"],
            "event_key": run["event_key"], "realism": run["realism"],
            "framing": run["framing"], "rep": run["rep"],
            "shock_tick": run.get("shock_tick", 10), "scope": scope,
            "probe_version": PROBE_VERSION, "follow_up": FOLLOW_UP,
            "probes": probes,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        done_runs += 1
        print(f"\r    {done_runs}/{len(files)} runs, {done_probes} probes, "
              f"{(time.time()-t0)/60:.1f} min", end="", flush=True)
        if limit is not None and done_probes >= limit:
            break

    print(f"\n  Done {model}: {done_probes} new probes in "
          f"{(time.time()-t0)/60:.1f} min -> {MEDIATOR_DIR}/{model_slug(model)}/")
    return {"model": model, "new_probes": done_probes, "elapsed_s": time.time() - t0}


def benchmark(models: list, n_each: int) -> None:
    """Time a handful of real probe calls per model to calibrate the estimate."""
    print("=" * 78)
    print(f"  PROBE BENCHMARK — {n_each} calls per model (nothing persisted)")
    print("=" * 78)
    for model in models:
        backend = _make_backend(model)
        files = _run_files(model)
        times, sample = [], []
        for p in files:
            if len(times) >= n_each:
                break
            run = json.load(open(p, encoding="utf-8"))
            if run.get("realism") not in ("real", "fictional"):
                continue
            perceived, raw, dt = _one_probe(backend, "retail", run["shock_text"])
            times.append(dt)
            sample.append((run["realism"], run["framing"], perceived, raw[:40]))
        if times:
            times.sort()
            mean = sum(times) / len(times)
            med = times[len(times) // 2]
            print(f"\n  {model}: n={len(times)}  mean={mean:.2f}s  median={med:.2f}s "
                  f"min={times[0]:.2f}s max={times[-1]:.2f}s")
            for rl, fr, perc, raw in sample[:6]:
                print(f"      [{rl}/{fr}] -> {perc:<12} raw={raw!r}")
    print("\n" + "=" * 78)


# ---------------------------------------------------------------------------
# Manipulation check (Step 1, requirement 4)
# ---------------------------------------------------------------------------

def _iter_mediator(models: list):
    for model in models:
        for p in sorted(glob.glob(os.path.join(MEDIATOR_DIR, model_slug(model),
                                               "*.json"))):
            try:
                yield json.load(open(p, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue


def manipulation_check(models: list) -> None:
    # counts[(model, framing, realism)] = [n_real_perceived, n_total, n_unclear]
    counts = defaultdict(lambda: [0, 0, 0])
    have = set()
    for rec in _iter_mediator(models):
        have.add(rec["model"])
        key = (rec["model"], rec["framing"], rec["realism"])
        for pr in rec["probes"]:
            perc = pr["perceived"]
            if perc == "error":
                continue
            counts[key][1] += 1
            if perc == "real":
                counts[key][0] += 1
            elif perc == "unclear":
                counts[key][2] += 1

    if not counts:
        print("  No mediator results found. Run the probe first.")
        return

    print("=" * 78)
    print("  MANIPULATION CHECK — does perceived realism (M) track true realism?")
    print("=" * 78)
    print("  Fraction of probes answering 'real', by cell.")
    print("  Manipulation PASSES if real shocks -> high, fictional -> low.\n")
    print(f"  {'model':14s} {'framing':10s} {'realism':10s} "
          f"{'P(real)':>9} {'n':>6} {'unclear':>8}")
    for model in sorted(have):
        for fr in ("narrative", "numeric"):
            for rl in ("real", "fictional"):
                real_n, tot, unclear = counts[(model, fr, rl)]
                if tot == 0:
                    continue
                print(f"  {model:14s} {fr:10s} {rl:10s} "
                      f"{real_n/tot:>9.1%} {tot:>6} {unclear:>8}")
        # per-framing separation = P(real|real) - P(real|fictional)
        for fr in ("narrative", "numeric"):
            rr = counts[(model, fr, "real")]
            ff = counts[(model, fr, "fictional")]
            if rr[1] and ff[1]:
                sep = rr[0] / rr[1] - ff[0] / ff[1]
                verdict = "PASS" if sep > 0.2 else ("WEAK" if sep > 0.05 else "FAIL")
                print(f"    -> {model} / {fr}: separation "
                      f"P(real|real)-P(real|fictional) = {sep:+.1%}  [{verdict}]")
        print()
    print("=" * 78)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Perceived-realism mediator probe.")
    ap.add_argument("--model", default=None, help="probe a single Ollama model")
    ap.add_argument("--all-models", action="store_true",
                    help="probe both models sequentially")
    ap.add_argument("--scope", choices=["postshock", "all"], default="postshock",
                    help="post-shock records only (default) or every decision record")
    ap.add_argument("--benchmark", type=int, default=0,
                    help="time N probe calls per model and exit (persists nothing)")
    ap.add_argument("--estimate-only", action="store_true",
                    help="print probe counts per model for the chosen scope and exit")
    ap.add_argument("--report", action="store_true",
                    help="print the manipulation check from existing results")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    models = ALL_MODELS if args.all_models else [args.model or DEFAULT_MODEL]

    if args.report:
        manipulation_check(models if (args.all_models or args.model) else ALL_MODELS)
        return 0
    if args.estimate_only:
        counts = count_probes(models, args.scope)
        for m, n in counts.items():
            print(f"  {m}: {n} probe calls (scope={args.scope})")
        print(f"  total: {sum(counts.values())}")
        return 0
    if args.benchmark:
        benchmark(models, args.benchmark)
        return 0

    for model in models:
        probe_model(model, scope=args.scope, resume=not args.no_resume)
    print("\n  Next: python3 probe_perceived_realism.py --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
