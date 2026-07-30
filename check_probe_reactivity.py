#!/usr/bin/env python3
"""
Probe-reactivity / reproducibility check (rigor-closing Step 2).

The stated worry: the mediator probe, appended after a trading decision, might
prime behavior on later ticks. In this codebase that is impossible at two levels:

  1. The probe was NEVER in the trading prompt — it is a separate, stateless,
     post-hoc Ollama call (verified: no probe phrasing in agents.py or any stored
     run). The trading agents never saw it.
  2. Agents are stateless per tick — Agent.decide() builds a fresh user prompt
     from the current book each tick and calls the backend with no conversation
     history, so nothing from tick t (probe or otherwise) can carry to t+1.

So a "probe-free replicate" cannot differ from the original by *probe* effects;
re-running the same (family, rep) at the SAME seed differs only by LLM sampling
(temperature 0.7). That difference is worth measuring on its own: it is the
run-to-run REPRODUCIBILITY / stochasticity floor of the whole pipeline, which
bounds how seriously to read any single trajectory and contextualizes the CIs.

This script re-runs a stratified subsample probe-free and compares, tick-by-tick,
to the first draw (results_2x2/ for models in the main sweep, else
results_2x2_reactivity_base/ -- see BASE_DIR_AUX).

Agreement is reported raw AND chance-corrected (Cohen's kappa). Raw agreement is
not comparable across models whose action mixes differ: mistral holds on 36.5% of
decisions and llama on 5.5%, which moves the chance floor from 45% to 67%. On the
raw number mistral looks the least stable of the three; corrected, it is more
stable than llama. Quote kappa when comparing models, raw when quoting the
absolute flip rate of one.

Usage:
  python3 check_probe_reactivity.py --run --all-models --per-model 10
  python3 check_probe_reactivity.py --report --all-models
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from experiments.design_2x2 import run_one, load_shocks, model_slug, SHOCK_TICK
from analyze_2x2 import load_runs

N_TICKS = 20
OUT_DIR = "results_2x2_reactivity"
BASE_DIR = "results_2x2"
# Where a first draw goes for a model that is NOT part of the main 2x2 sweep.
#
# It must not be BASE_DIR. A reactivity subsample is 10 runs over 2 families;
# dropped into results_2x2/ it is indistinguishable from a sweep, and
# analyze_2x2.py --compare-models and power_analysis.py both enumerate models by
# directory contents, so a 10-run partial would be reported next to the two
# 80-run sweeps as if it were a third arm of the study.
BASE_DIR_AUX = "results_2x2_reactivity_base"
# llama and mistral share a lineage; gemma2 is a separate architecture family,
# included so the run-to-run instability finding is tested ACROSS model families
# rather than only within one. It has no entries in the main 2x2 sweep, so its
# baseline is generated here (see _selection / _ensure_baseline).
ALL_MODELS = ["llama3.1:8b", "mistral:7b", "gemma2:9b"]
BASELINE_REF = "llama3.1:8b"
CELLS = ["narrative__real", "narrative__fictional", "numeric__real", "numeric__fictional"]
TRAJ_METRICS = ["spread", "kyles_lambda", "volume"]
RUN_METRICS = ["spread_widening", "kyle_lambda_delta", "archetype_divergence"]


def _shock_by_id() -> dict:
    return {s["shock_id"]: s for s in load_shocks()}


def _subsample(model_runs: list, per_model: int) -> list:
    """Stratified round-robin across cells until per_model selected."""
    bycell: dict = defaultdict(list)
    for r in model_runs:
        bycell[f"{r['framing']}__{r['realism']}"].append(r)
    for c in bycell:
        bycell[c].sort(key=lambda r: (r["event_key"], r["rep"]))
    picked, i = [], 0
    while len(picked) < per_model and any(i < len(bycell[c]) for c in CELLS):
        for c in CELLS:
            if i < len(bycell.get(c, [])) and len(picked) < per_model:
                picked.append(bycell[c][i])
        i += 1
    return picked


def out_path(model: str, r: dict) -> str:
    return os.path.join(OUT_DIR, model_slug(model),
                        f"{r['framing']}__{r['realism']}",
                        f"{r['event_key']}__rep{r['rep']:02d}.json")


def base_dir_for(model: str) -> str:
    """Main sweep if the model has one there, else the auxiliary tree."""
    d = os.path.join(BASE_DIR, model_slug(model))
    return BASE_DIR if os.path.isdir(d) and glob.glob(
        os.path.join(d, "*", "*.json")) else BASE_DIR_AUX


def base_path(model: str, r: dict) -> str:
    return os.path.join(base_dir_for(model), model_slug(model),
                        f"{r['framing']}__{r['realism']}",
                        f"{r['event_key']}__rep{r['rep']:02d}.json")


def _selection(model: str, per_model: int, ref_model: str = BASELINE_REF) -> list:
    """The (shock, rep) set to replicate for `model`.

    A model added only for this check has no entries in results_2x2/ to
    subsample, so its selection is copied from `ref_model`. Seeds are
    model-independent — make_seed(family, rep) — so the copied selection runs
    against an identical market draw. That is what makes cross-model flip rates
    comparable rather than merely similar: every model is re-deciding the same
    20 ticks of the same book.
    """
    runs = load_runs(BASE_DIR, model)
    if runs:
        return _subsample(runs, per_model)
    ref = load_runs(BASE_DIR, ref_model)
    if not ref:
        return []
    print(f"    (no {model} runs in {BASE_DIR}/ — selection copied from "
          f"{ref_model}, baselines generated below)")
    return _subsample(ref, per_model)


def _execute(shocks: dict, model: str, r: dict, path: str, resume: bool) -> bool:
    """Run one (shock, rep) to `path`. True if it executed, False if skipped."""
    if resume and os.path.exists(path):
        return False
    shock = shocks.get(r["shock_id"])
    if shock is None:
        return False
    res = run_one(shock, r["rep"], N_TICKS, SHOCK_TICK, model=model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"), indent=2)
    return True


def run(models: list, per_model: int, resume: bool = True) -> None:
    shocks = _shock_by_id()
    for model in models:
        sample = _selection(model, per_model)
        if not sample:
            print(f"  {model}: no selection available — skipping.")
            continue
        # Two same-seed draws are what the check compares. For llama/mistral the
        # first draw already exists in the main sweep and resume skips it; for a
        # model new to the study both draws are generated here, which is the same
        # comparison — two independent samplings of one fixed market.
        n_base = sum(1 for r in sample if not os.path.exists(base_path(model, r)))
        print(f"  {model}: {len(sample)} replicates "
              f"({n_base} baselines to generate first)")
        for k, r in enumerate(sample, 1):
            _execute(shocks, model, r, base_path(model, r), resume)
            _execute(shocks, model, r, out_path(model, r), resume)
            print(f"\r    {k}/{len(sample)} {r['shock_id']} rep{r['rep']}"
                  f"{' ' * 12}", end="", flush=True)
        print()
    print(f"  done -> {OUT_DIR}/")


def _match_original(model: str, rerun: dict) -> Optional[dict]:
    p = base_path(model, rerun)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def report(models: list) -> None:
    print("=" * 84)
    print("  PROBE-REACTIVITY / REPRODUCIBILITY CHECK")
    print("  Re-run vs original at the SAME seed (differ only by LLM sampling).")
    print("=" * 84)
    for model in models:
        reruns = []
        for p in glob.glob(os.path.join(OUT_DIR, model_slug(model), "*", "*.json")):
            reruns.append(json.load(open(p, encoding="utf-8")))
        if not reruns:
            print(f"\n  [{model}] no replicates yet.")
            continue
        run_deltas = defaultdict(list)
        traj_deltas = defaultdict(list)
        act_agree = []
        act_pairs = []
        for rr in reruns:
            orig = _match_original(model, rr)
            if orig is None:
                continue
            for m in RUN_METRICS:
                a, b = rr["metrics"].get(m), orig["metrics"].get(m)
                if a is not None and b is not None and not (np.isnan(a) or np.isnan(b)):
                    run_deltas[m].append(abs(a - b))
            # per-tick trajectory deltas
            for ta, tb in zip(rr["trajectory"], orig["trajectory"]):
                for tm in TRAJ_METRICS:
                    va, vb = ta.get(tm), tb.get(tm)
                    if va is not None and vb is not None:
                        traj_deltas[tm].append(abs(va - vb))
            # decision action agreement (LLM agents only)
            for da, db in zip(rr["decisions"], orig["decisions"]):
                for xa, xb in zip(da, db):
                    if xa.get("archetype") == "market_maker":
                        continue
                    aa = "hold" if xa.get("held") else (xa["orders"][0]["side"] if xa.get("orders") else "hold")
                    bb = "hold" if xb.get("held") else (xb["orders"][0]["side"] if xb.get("orders") else "hold")
                    act_agree.append(1.0 if aa == bb else 0.0)
                    act_pairs.append((aa, bb))

        print(f"\n  MODEL: {model}  ({len(reruns)} replicates)")
        print("    run-metric |Δ| (rerun vs original, same seed):")
        for m in RUN_METRICS:
            d = run_deltas[m]
            if d:
                print(f"      {m:<22} mean|Δ|={np.mean(d):.5f}  max|Δ|={np.max(d):.5f}")
        print("    per-tick trajectory mean|Δ|:")
        for tm in TRAJ_METRICS:
            d = traj_deltas[tm]
            if d:
                print(f"      {tm:<14} {np.mean(d):.5f}")
        if act_agree:
            agree = float(np.mean(act_agree))
            # Reported both ways: earlier write-ups quote the agreement rate,
            # the finding is usually stated as the flip rate. They are 1 - each
            # other, so quoting one alone invites reading it as the other.
            print(f"    LLM decision action agreement (same seed): {agree:.1%}"
                  f"   -> decision FLIP rate: {1 - agree:.1%}"
                  f"   (n={len(act_agree)} decisions)")
            # Raw agreement is confounded by how concentrated a model's action
            # distribution is: a model that answers BUY 82% of the time agrees
            # with itself ~70% by chance alone. That confound is tolerable when
            # comparing models with similar action mixes and NOT tolerable here,
            # where hold rates run from 4.5% (llama) to 36.7% (mistral) — the
            # whole point of adding a third architecture is a cross-family
            # comparison, so the chance floor has to be divided out.
            n_act = len(act_pairs)
            pa = Counter(a for a, _ in act_pairs)
            pb = Counter(b for _, b in act_pairs)
            p_chance = sum((pa[k] / n_act) * (pb[k] / n_act)
                           for k in set(pa) | set(pb))
            kappa = ((agree - p_chance) / (1 - p_chance)
                     if p_chance < 1 else float("nan"))
            mix = "  ".join(f"{k}={v / n_act:.1%}"
                            for k, v in sorted(pb.items(), key=lambda kv: -kv[1]))
            print(f"      chance agreement {p_chance:.1%} given action mix "
                  f"({mix})")
            print(f"      -> Cohen's kappa (chance-corrected stability): {kappa:.3f}")
    print("\n  READING: any difference here is pure LLM sampling noise at a fixed seed")
    print("  (market-maker walk + book warm-up are seed-identical). The probe was")
    print("  never inline AND agents are stateless per tick, so this is NOT probe")
    print("  reactivity — it is the pipeline's reproducibility floor. Compare mean|Δ|")
    print("  against each metric's MDE (power section): the stochastic floor is the")
    print("  irreducible noise the family bootstrap already integrates over.")
    print("=" * 84)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Probe reactivity / reproducibility.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--per-model", type=int, default=10)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    models = ALL_MODELS if args.all_models else [args.model or "llama3.1:8b"]
    if args.run:
        run(models, args.per_model, resume=not args.no_resume)
        print("  Next: python3 check_probe_reactivity.py --report --all-models")
    if args.report:
        report(models)
    if not (args.run or args.report):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
