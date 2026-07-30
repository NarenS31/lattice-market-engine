#!/usr/bin/env python3
"""
Causal-mediation Step 2 — natural direct / indirect effects of FRAMING,
mediated by perceived realism M.

Extends analyze_2x2.py. Treatment T = framing (narrative=1, numeric=0). Mediator
M = perceived realism, joined per run from results_2x2_mediator/ (fraction of a
run's probes answering "real"). Outcome Y = a primary behavioral metric. Realism
R (real=1, fictional=0) is a pre-treatment covariate, adjusted for in both models.

Standard linear (product-of-coefficients) mediation, no T*M interaction:

    mediator model :  M = a0 + a1*T + a2*R
    outcome model  :  Y = b0 + b1*T + b2*M + b3*R
    total  (TE)    :  coef of T in  Y = c0 + c1*T + c2*R
    NIE (indirect) :  a1 * b2   =  E[Y|T=1,M(1)] - E[Y|T=1,M(0)]
    NDE (direct)   :  b1        =  E[Y|T=1,M(0)] - E[Y|T=0,M(0)]
    (linear identity: TE = NDE + NIE)

Inference is the SAME family-cluster bootstrap as analyze_2x2.cluster_bootstrap:
resample the event families with replacement, then resample runs within each
drawn family, refit, 1000 iterations, percentile 95% CI. Family is the unit of
independent replication; this does not change here.

Read the manipulation check (probe_perceived_realism.py --report) FIRST. If M
does not vary with framing for a model (e.g. a model that calls everything
"real"), that model's a1 ~ 0 and its NIE is ~0 by construction — uninformative,
NOT evidence of "no mediation". This script prints each model's M spread so that
case is visible.

Usage:
    python3 analyze_mediation.py                 # both models, print
    python3 analyze_mediation.py --model llama3.1:8b
    python3 analyze_mediation.py --append        # also append section to RESULTS_2x2.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np

from analyze_2x2 import (
    load_runs, split_by_model, complete_families, index_runs, ci,
    DEFAULT_RESULTS, DEFAULT_ITERS, BOOTSTRAP_SEED, FRAMINGS, REALISMS,
)
try:
    from experiments.design_2x2 import model_slug
except Exception:  # pragma: no cover
    def model_slug(m: str) -> str:
        return m.replace(":", "_").replace(".", "").replace("/", "_")

MEDIATOR_DIR = "results_2x2_mediator"
SECTION_HEADER = "## Causal Mediation Extension (Steps 1–2)"

# Ordered so the printout matches analyze_2x2's primary metrics.
METRICS = [
    ("spread_widening", "Spread widening (post - pre)"),
    ("kyle_lambda_delta", "Kyle's lambda change (post - pre)"),
    ("archetype_divergence", "Archetype divergence (sd of returns)"),
]


# ---------------------------------------------------------------------------
# Join mediator M
# ---------------------------------------------------------------------------

def load_mediator_M(model: str) -> dict:
    """{(shock_id, rep): M} where M = fraction of the run's probes = 'real'."""
    out: dict = {}
    for p in glob.glob(os.path.join(MEDIATOR_DIR, model_slug(model), "*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        real = sum(1 for x in d["probes"] if x["perceived"] == "real")
        tot = sum(1 for x in d["probes"] if x["perceived"] in ("real", "hypothetical"))
        if tot:
            out[(d["shock_id"], d["rep"])] = real / tot
    return out


def build_records(runs: list, Mmap: dict, metric: str) -> list:
    recs = []
    for r in runs:
        y = r["metrics"].get(metric)
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        key = (r["shock_id"], r["rep"])
        if key not in Mmap:
            continue
        recs.append({
            "fam": r["matched_pair_id"],
            "T": 1.0 if r["framing"] == "narrative" else 0.0,
            "R": 1.0 if r["realism"] == "real" else 0.0,
            "M": Mmap[key],
            "Y": float(y),
        })
    return recs


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def mediation_point(recs: list) -> dict:
    """Total, direct, indirect effects of framing from one set of records."""
    T = np.array([r["T"] for r in recs])
    R = np.array([r["R"] for r in recs])
    M = np.array([r["M"] for r in recs])
    Y = np.array([r["Y"] for r in recs])
    one = np.ones(len(recs))

    a1 = _ols(np.column_stack([one, T, R]), M)[1]          # framing -> M
    te = _ols(np.column_stack([one, T, R]), Y)[1]          # total effect
    bo = _ols(np.column_stack([one, T, M, R]), Y)          # outcome w/ mediator
    nde, b2 = bo[1], bo[2]
    nie = a1 * b2
    return {"TE": te, "NDE": nde, "NIE": nie, "a1": a1, "b2": b2}


def mediation_bootstrap(recs: list, families: list, iters: int,
                        rng: np.random.Generator) -> dict:
    """Family-cluster bootstrap (mirrors analyze_2x2.cluster_bootstrap):
    resample families with replacement, then runs within each drawn family."""
    byfam: dict = defaultdict(list)
    for r in recs:
        byfam[r["fam"]].append(r)
    fams = [f for f in families if byfam[f]]
    nf = len(fams)
    TE, NDE, NIE = [], [], []
    for _ in range(iters):
        drawn = rng.integers(0, nf, size=nf)
        boot = []
        for di in drawn:
            fr = byfam[fams[di]]
            idx = rng.integers(0, len(fr), size=len(fr))
            boot.extend(fr[i] for i in idx)
        try:
            res = mediation_point(boot)
        except np.linalg.LinAlgError:
            continue
        TE.append(res["TE"]); NDE.append(res["NDE"]); NIE.append(res["NIE"])
    return {"TE": np.array(TE), "NDE": np.array(NDE), "NIE": np.array(NIE)}


# ---------------------------------------------------------------------------
# Mediator descriptive: does M vary with framing? (the key caveat driver)
# ---------------------------------------------------------------------------

def m_by_cell(runs: list, Mmap: dict) -> dict:
    """{(framing, realism): mean M} and overall M std, over runs of one model."""
    cell: dict = defaultdict(list)
    allM = []
    for r in runs:
        key = (r["shock_id"], r["rep"])
        if key in Mmap:
            cell[(r["framing"], r["realism"])].append(Mmap[key])
            allM.append(Mmap[key])
    means = {k: float(np.mean(v)) for k, v in cell.items()}
    # framing spread in M = |mean over narrative - mean over numeric|
    narr = [Mmap[(r["shock_id"], r["rep"])] for r in runs
            if r["framing"] == "narrative" and (r["shock_id"], r["rep"]) in Mmap]
    num = [Mmap[(r["shock_id"], r["rep"])] for r in runs
           if r["framing"] == "numeric" and (r["shock_id"], r["rep"]) in Mmap]
    framing_spread = (float(np.mean(narr)) - float(np.mean(num))) if narr and num else float("nan")
    return {"means": means, "std": float(np.std(allM)) if allM else float("nan"),
            "framing_spread": framing_spread}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _vs_zero(lo: float, hi: float) -> str:
    return "excludes" if (lo > 0 or hi < 0) else "  spans "


def analyse_model(model: str, runs: list, Mmap: dict, iters: int) -> dict:
    families = complete_families(index_runs(runs, "spread_widening"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    md = m_by_cell(runs, Mmap)
    out = {"model": model, "families": families, "m": md, "metrics": {}}
    for metric, _label in METRICS:
        recs = build_records(runs, Mmap, metric)
        if not recs:
            continue
        pt = mediation_point(recs)
        dist = mediation_bootstrap(recs, families, iters, rng)
        res = {"point": pt}
        for eff in ("TE", "NDE", "NIE"):
            lo, hi = ci(dist[eff]) if len(dist[eff]) else (float("nan"), float("nan"))
            res[eff] = {"est": pt[eff], "lo": lo, "hi": hi}
        out["metrics"][metric] = res
    return out


def render(results: list) -> str:
    L: list = []
    P = L.append
    P("=" * 84)
    P("  CAUSAL MEDIATION — natural direct / indirect effects of FRAMING")
    P("  Treatment: framing (narrative=1, numeric=0).  Mediator: perceived realism M.")
    P("  Family-cluster bootstrap (resample families), 1000 iters, 95% CI.")
    P("=" * 84)

    for res in results:
        model = res["model"]
        md = res["m"]
        P("\n" + "=" * 84)
        P(f"  MODEL: {model}   ({len(res['families'])} families)")
        P("=" * 84)
        P("  Mediator M (fraction perceiving 'real'), by cell — does framing move M?")
        for rl in REALISMS:
            row = "   ".join(
                f"{fr}:{md['means'].get((fr, rl), float('nan')):.2f}" for fr in FRAMINGS)
            P(f"    {rl:<10} {row}")
        P(f"    framing spread in M (narrative - numeric): {md['framing_spread']:+.3f}"
          f"   |  overall M std: {md['std']:.3f}")
        if abs(md["framing_spread"]) < 0.05 or md["std"] < 0.05:
            P("    ** CAVEAT: M barely varies for this model. a1~0 forces NIE~0 by")
            P("       construction — uninformative, NOT evidence of 'no mediation'.")

        for metric, label in METRICS:
            r = res["metrics"].get(metric)
            if not r:
                continue
            P("\n  " + "-" * 80)
            P(f"  {label}   [{metric}]")
            P("  " + "-" * 80)
            P(f"    {'effect':<26}{'estimate':>12}   {'95% CI':>24}   vs zero")
            for eff, name in (("TE", "Total effect (framing)"),
                              ("NDE", "Natural direct  (NDE)"),
                              ("NIE", "Natural indirect (NIE)")):
                e = r[eff]
                tag = _vs_zero(e["lo"], e["hi"]) if eff == "NIE" else ""
                P(f"    {name:<26}{e['est']:>12.5f}   [{e['lo']:>10.5f},{e['hi']:>10.5f}]"
                  f"   {tag}")
            nie = r["NIE"]
            excl = (nie["lo"] > 0 or nie["hi"] < 0)
            if excl:
                P("    READING: NIE excludes zero — framing's effect on this metric runs")
                P("             THROUGH perceived realism (real mediation).")
            else:
                P("    READING: NIE spans zero — no evidence framing acts through perceived")
                P("             realism; any framing effect runs through another channel.")
    P("\n" + "=" * 84)
    P("  CAVEATS (carry from analyze_2x2 + the manipulation check)")
    P("  * n=4 families: wide CIs; linear mediation assumes no framing x M interaction.")
    P("  * Mediator was probed ignoring order-book state, so M varies at the")
    P("    (framing x realism x family) level, not per tick — M is collinear with the")
    P("    design, which widens the NIE interval honestly.")
    P("  * A model whose M does not move with framing yields NIE~0 mechanically;")
    P("    read that as 'mediator undefined', not 'mediation ruled out'.")
    P("=" * 84)
    return "\n".join(L)


def render_markdown(results: list) -> str:
    M: list = []
    P = M.append
    P(SECTION_HEADER)
    P("")
    P("Extends the behavioral 2x2 with a mediator. **Step 1** probed each agent's "
      "*perceived* realism M (`probe_perceived_realism.py`, 9,600 calls); **Step 2** "
      "(`analyze_mediation.py`) decomposes framing's effect on behavior into a part "
      "running *through* M (natural indirect effect, NIE) and a part that does not "
      "(natural direct effect, NDE), family-cluster bootstrap, 1000 iters.")
    P("")
    P("**Step 1 — manipulation check (fraction perceiving 'real'):**")
    P("")
    P("| model | framing | P(real \\| real) | P(real \\| fictional) | separation |")
    P("|---|---|---|---|---|")
    for res in results:
        md = res["m"]["means"]
        for fr in FRAMINGS:
            rr = md.get((fr, "real"), float("nan"))
            ff = md.get((fr, "fictional"), float("nan"))
            P(f"| {res['model']} | {fr} | {rr:.0%} | {ff:.0%} | {rr - ff:+.0%} |")
    P("")
    P("Only **llama3.1:8b + narrative** shows a clean separation; **mistral:7b** "
      "answers “real” to ~94–100% of everything (no realism discrimination), so "
      "its mediator is near-constant and its NIE is ~0 by construction, not by finding.")
    P("")
    P("**Step 2 — natural indirect effect (NIE) of framing through M, per metric:**")
    P("")
    P("| model | metric | Total | NIE (indirect) | NDE (direct) | NIE vs 0 |")
    P("|---|---|---|---|---|---|")
    for res in results:
        for metric, _label in METRICS:
            r = res["metrics"].get(metric)
            if not r:
                continue
            nie = r["NIE"]
            tag = "**excludes**" if (nie["lo"] > 0 or nie["hi"] < 0) else "spans"
            P(f"| {res['model']} | {metric} | {r['TE']['est']:.4f} "
              f"| {nie['est']:.4f} [{nie['lo']:.4f}, {nie['hi']:.4f}] "
              f"| {r['NDE']['est']:.4f} | {tag} |")
    P("")
    any_excl = any(
        (res["metrics"][m]["NIE"]["lo"] > 0 or res["metrics"][m]["NIE"]["hi"] < 0)
        for res in results for m, _ in METRICS if m in res["metrics"])
    if any_excl:
        P("**Result:** at least one NIE excludes zero — evidence that part of "
          "framing's behavioral effect runs through perceived realism.")
    else:
        P("**Result:** every NIE spans zero. There is **no evidence** that framing "
          "affects behavior *through whether agents believe the event is real*. "
          "Combined with the flat behavioral interaction and the flat reasoning-text "
          "leakage, the belief channel carries no measurable causal weight here — "
          "and for mistral the question is moot because the manipulation never "
          "registered perceptually.")
    P("")
    P("*Caveats:* n=4 families (wide CIs); linear mediation, no framing×M "
      "interaction; M probed without order-book state so it is collinear with the "
      "design; n=2 models is descriptive. This is associational mediation — Step 3 "
      "(interventional M manipulation) would be needed to confirm.")
    P("")
    return "\n".join(M)


def append_to_markdown(md_path: str, section: str) -> str:
    block = section.rstrip() + "\n"
    content = open(md_path, encoding="utf-8").read() if os.path.exists(md_path) else ""
    if SECTION_HEADER in content:
        pat = re.compile(re.escape(SECTION_HEADER) + r".*?(?=\n## |\Z)", re.S)
        content = pat.sub(block.rstrip() + "\n", content)
        action = "replaced existing section"
    else:
        content = content.rstrip() + "\n\n---\n\n" + block if content else block
        action = "appended new section"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return action


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Mediation analysis (Step 2).")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--model", default=None, help="single model (default: all present)")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--md", default="RESULTS_2x2.md")
    ap.add_argument("--append", action="store_true",
                    help="append the Steps 1-2 section to RESULTS_2x2.md")
    args = ap.parse_args(argv)

    runs = load_runs(args.results, args.model)
    by_model = split_by_model(runs)
    models = [args.model] if args.model else sorted(by_model)

    results = []
    for model in models:
        mruns = by_model.get(model, [])
        Mmap = load_mediator_M(model)
        if not mruns or not Mmap:
            print(f"  [skip {model}] runs={len(mruns)} mediator_files={len(Mmap)}")
            continue
        results.append(analyse_model(model, mruns, Mmap, args.iters))

    print(render(results))

    if args.append:
        action = append_to_markdown(args.md, render_markdown(results))
        print(f"\n  RESULTS_2x2.md: {action} ({SECTION_HEADER!r}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
