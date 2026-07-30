"""
analyze_2x2.py — main effects and interaction for the framing x realism design.

Estimands
---------
With cell means M[framing][realism] (each averaged over event families):

  main effect of framing  = mean_realism( M[narrative][.] - M[numeric][.] )
  main effect of realism  = mean_framing( M[.][real]      - M[.][fictional] )
  interaction             = ( M[nar][real] - M[num][real] )
                          - ( M[nar][fict] - M[num][fict] )

The interaction is the headline: it asks whether the framing effect is *larger
for real shocks than fictional ones*, which is the signature of agents drawing
on recall of real events rather than reasoning about shock structure.

  interaction > 0  → framing matters more for real shocks (recall signature)
  interaction ≈ 0  → framing effect is realism-independent (structural response)
  interaction < 0  → framing matters more for fictional shocks

Why the bootstrap resamples FAMILIES, not runs
----------------------------------------------
Each event family (lithium, semiconductor, energy, credit) appears in all four
cells, so family is crossed with both factors, and runs sharing a family share
a shock text and a seed. The primary CI is therefore a **cluster bootstrap**:
resample families with replacement, then resample runs within each drawn
family-cell. Two reasons:

  1. It targets the question you actually want answered — would this
     interaction replicate on *new* event families? — rather than "would it
     replicate on new runs of these same four shocks?"
  2. It respects the clustering rather than assuming runs are independent.

The naive run-level bootstrap (resampling runs within each cell, pooled across
families) is reported alongside as a diagnostic, NOT as a fallback. Note it is
not simply "too narrow": because the design is fully crossed, additive family
effects cancel within the interaction contrast, and breaking the pairing can
inflate the naive interval as easily as shrink it. Treat a large gap between
the two as a signal that between-family heterogeneity is driving the result,
and read the per-family interaction values printed below.

Validation
----------
The cluster bootstrap was checked by simulation against a known ground truth
(true interaction = 0, correlated runs within family). Empirical coverage of
the nominal 95% interval:

    4 families, tau=0.6  ->  95.0%
    4 families, tau=0.0  ->  96.7%
   10 families, tau=0.6  ->  96.7%

and on data with a large injected interaction the CI excluded zero as expected.
Coverage is therefore approximately nominal even at four clusters *under
Gaussian family effects*. That is a real check, but not a blank cheque — see
the caveat printed at the end for what four families still costs you.

Cross-model replication
-----------------------
Results are filed per model (results_2x2/{model}/...). --compare-models runs
the interaction analysis separately for each model and reports them side by
side. The seeds are model-independent, so a given family+rep faces an identical
market on every model and a difference between models is attributable to the
model.

Two models is a replication check, not a sample of models. The combined
estimate averages over models and is reported as **descriptive**: with n=2 you
cannot resample models, so the interval carries family-level uncertainty only
and says nothing about whether the effect would hold on a third model. That is
the same standard as the four-families caveat below — the constraint is stated
rather than papered over.

Neither outcome is the "right" one. An interaction that appears on one model
and not the other is a finding about model-specific behaviour and is
reportable as such; this script does not treat replication as success and
non-replication as failure.

Run
---
  python3 analyze_2x2.py
  python3 analyze_2x2.py --iters 1000 --results results_2x2
  python3 analyze_2x2.py --model llama3.1:8b       # one model only
  python3 analyze_2x2.py --compare-models          # side-by-side + combined
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import numpy as np

DEFAULT_RESULTS = "results_2x2"
DEFAULT_ITERS = 1000
BOOTSTRAP_SEED = 20260719

FRAMINGS = ("narrative", "numeric")
REALISMS = ("real", "fictional")

# (metric key, display label, direction note)
METRICS = [
    ("spread_widening",      "Spread widening (post - pre)",        "higher = more illiquid"),
    ("kyle_lambda_delta",    "Kyle's lambda change (post - pre)",   "higher = more price impact"),
    ("archetype_divergence", "Archetype divergence (sd of returns)", "higher = more dispersion"),
    ("kyle_lambda_post",     "Kyle's lambda, post-shock level",     "secondary"),
    ("spread_post",          "Spread, post-shock level",            "secondary"),
]
PRIMARY = {"spread_widening", "kyle_lambda_delta", "archetype_divergence"}

# Below this many clusters, a percentile cluster bootstrap is not trustworthy.
MIN_CLUSTERS_FOR_INFERENCE = 10

# Number of models below which the model dimension is descriptive only. You
# cannot resample two clusters; see report_model_caveats.
MIN_MODELS_FOR_INFERENCE = 5

UNKNOWN_MODEL = "unknown"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _model_from_path(path: str, results_dir: str) -> str:
    """Recover the model from the directory layout as a fallback.

    Runs written by the current runner carry a "model" field; this only covers
    results produced before that field existed. The directory name is a slug
    ('llama3.1_8b'), not the true tag, so it is used only when the field is
    absent — and it is reported verbatim rather than guessed back into a tag.
    """
    rel = os.path.relpath(path, results_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) >= 3 else UNKNOWN_MODEL


def load_runs(results_dir: str, model: Optional[str] = None) -> list:
    """Load runs from either layout.

    Current: results_dir/{model}/{framing}__{realism}/{event}__repNN.json
    Legacy:  results_dir/{framing}__{realism}/{event}__repNN.json

    `model` filters on the run's recorded model tag (exact match), falling back
    to the directory slug for legacy runs so `--model llama3.1_8b` also works.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, "**", "*.json"),
                             recursive=True))
    runs = []
    for p in paths:
        if os.path.basename(p) == "manifest.json":
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                r = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if "metrics" not in r or "matched_pair_id" not in r:
            continue
        r.setdefault("model", _model_from_path(p, results_dir))
        if model is not None and r["model"] != model:
            continue
        runs.append(r)
    return runs


def models_present(runs: list) -> list:
    return sorted({r.get("model", UNKNOWN_MODEL) for r in runs})


def split_by_model(runs: list) -> dict:
    out: dict = defaultdict(list)
    for r in runs:
        out[r.get("model", UNKNOWN_MODEL)].append(r)
    return dict(out)


def index_runs(runs: list, metric: str) -> dict:
    """{family: {framing: {realism: [values]}}} for one metric, NaNs dropped."""
    idx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        v = r["metrics"].get(metric)
        if v is None:
            continue
        v = float(v)
        if np.isnan(v):
            continue
        idx[r["matched_pair_id"]][r["framing"]][r["realism"]].append(v)
    return idx


def complete_families(idx: dict) -> list:
    """Families with at least one run in all four cells."""
    out = []
    for fam, by_fr in idx.items():
        if all(by_fr.get(fr, {}).get(rl) for fr in FRAMINGS for rl in REALISMS):
            out.append(fam)
    return sorted(out)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def family_cell_means(idx: dict, families: list) -> dict:
    """{family: {(framing, realism): mean}}"""
    return {
        fam: {
            (fr, rl): float(np.mean(idx[fam][fr][rl]))
            for fr in FRAMINGS for rl in REALISMS
        }
        for fam in families
    }


def family_interaction(cell: dict) -> float:
    """Interaction contrast within a single family."""
    return ((cell[("narrative", "real")] - cell[("numeric", "real")])
            - (cell[("narrative", "fictional")] - cell[("numeric", "fictional")]))


def family_main_framing(cell: dict) -> float:
    return 0.5 * ((cell[("narrative", "real")] - cell[("numeric", "real")])
                  + (cell[("narrative", "fictional")] - cell[("numeric", "fictional")]))


def family_main_realism(cell: dict) -> float:
    return 0.5 * ((cell[("narrative", "real")] - cell[("narrative", "fictional")])
                  + (cell[("numeric", "real")] - cell[("numeric", "fictional")]))


def point_estimates(idx: dict, families: list) -> dict:
    fcm = family_cell_means(idx, families)
    cells = {
        (fr, rl): float(np.mean([fcm[f][(fr, rl)] for f in families]))
        for fr in FRAMINGS for rl in REALISMS
    }
    per_family = {f: family_interaction(fcm[f]) for f in families}
    return {
        "cells": cells,
        "main_framing": float(np.mean([family_main_framing(fcm[f]) for f in families])),
        "main_realism": float(np.mean([family_main_realism(fcm[f]) for f in families])),
        "interaction": float(np.mean(list(per_family.values()))),
        "per_family_interaction": per_family,
        "n_runs": {
            (fr, rl): sum(len(idx[f][fr][rl]) for f in families)
            for fr in FRAMINGS for rl in REALISMS
        },
    }


# ---------------------------------------------------------------------------
# Bootstraps
# ---------------------------------------------------------------------------

def resample_family_cells(idx: dict, fam: str,
                          rng: np.random.Generator) -> dict:
    """One family's four cell means, with runs resampled within each cell."""
    cell = {}
    for fr in FRAMINGS:
        for rl in REALISMS:
            vals = idx[fam][fr][rl]
            pick = rng.integers(0, len(vals), size=len(vals))
            cell[(fr, rl)] = float(np.mean([vals[i] for i in pick]))
    return cell


def cluster_bootstrap(idx: dict, families: list, iters: int,
                      rng: np.random.Generator) -> dict:
    """Resample families with replacement; resample runs within each draw.

    Returns bootstrap distributions for all three contrasts.
    """
    n_fam = len(families)
    inter, m_fr, m_rl = [], [], []

    for _ in range(iters):
        drawn = rng.integers(0, n_fam, size=n_fam)
        inter_b, fr_b, rl_b = [], [], []

        for di in drawn:
            cell = resample_family_cells(idx, families[di], rng)
            inter_b.append(family_interaction(cell))
            fr_b.append(family_main_framing(cell))
            rl_b.append(family_main_realism(cell))

        inter.append(float(np.mean(inter_b)))
        m_fr.append(float(np.mean(fr_b)))
        m_rl.append(float(np.mean(rl_b)))

    return {"interaction": np.array(inter),
            "main_framing": np.array(m_fr),
            "main_realism": np.array(m_rl)}


def cross_model_bootstrap(idx_by_model: dict, models: list, families: list,
                          iters: int, rng: np.random.Generator) -> tuple:
    """Bootstrap the interaction jointly across models, paired on family.

    Each iteration draws ONE set of families and evaluates every model on that
    same draw. Pairing matters: the models ran on identical seeds and identical
    shock texts, so the family draw is a shared source of variance. Resampling
    families independently per model would discard that pairing and inflate the
    difference interval — the same mistake as an unpaired t-test on paired data.

    Returns ({model: interaction distribution}, combined distribution). The
    combined distribution averages models within each iteration; because models
    are averaged rather than resampled, it carries family-level uncertainty
    only. See report_model_caveats.
    """
    n_fam = len(families)
    per_model: dict = {m: [] for m in models}
    combined = []

    for _ in range(iters):
        drawn = rng.integers(0, n_fam, size=n_fam)
        vals: dict = {m: [] for m in models}
        for di in drawn:
            fam = families[di]
            for m in models:
                vals[m].append(
                    family_interaction(resample_family_cells(idx_by_model[m], fam, rng))
                )
        means = {m: float(np.mean(vals[m])) for m in models}
        for m in models:
            per_model[m].append(means[m])
        combined.append(float(np.mean(list(means.values()))))

    return ({m: np.array(v) for m, v in per_model.items()}, np.array(combined))


def naive_bootstrap(idx: dict, families: list, iters: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Run-level bootstrap ignoring family clustering (anticonservative)."""
    pooled = {
        (fr, rl): [v for f in families for v in idx[f][fr][rl]]
        for fr in FRAMINGS for rl in REALISMS
    }
    out = []
    for _ in range(iters):
        m = {}
        for key, vals in pooled.items():
            pick = rng.integers(0, len(vals), size=len(vals))
            m[key] = float(np.mean([vals[i] for i in pick]))
        out.append(
            (m[("narrative", "real")] - m[("numeric", "real")])
            - (m[("narrative", "fictional")] - m[("numeric", "fictional")])
        )
    return np.array(out)


def ci(dist: np.ndarray, alpha: float = 0.05) -> tuple:
    lo = float(np.percentile(dist, 100 * alpha / 2))
    hi = float(np.percentile(dist, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _f(x: float, w: int = 9) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return f"{'n/a':>{w}}"
    return f"{x:>{w}.5f}"


def report_metric(metric: str, label: str, note: str, runs: list,
                  iters: int, rng: np.random.Generator) -> Optional[dict]:
    idx = index_runs(runs, metric)
    families = complete_families(idx)

    print("\n" + "=" * 84)
    print(f"  {label}   [{metric}]")
    print(f"  {note}")
    print("=" * 84)

    if len(families) < 2:
        print(f"  Not enough complete families ({len(families)}) — skipping.")
        return None

    est = point_estimates(idx, families)
    cells, n = est["cells"], est["n_runs"]

    print(f"\n  Cell means (each averaged over {len(families)} families)")
    print(f"    {'':<12} {'real':>12} {'fictional':>12}      n(real)/n(fict)")
    for fr in FRAMINGS:
        print(f"    {fr:<12} {_f(cells[(fr,'real')], 12)} {_f(cells[(fr,'fictional')], 12)}"
              f"      {n[(fr,'real')]}/{n[(fr,'fictional')]}")

    boot = cluster_bootstrap(idx, families, iters, rng)
    naive = naive_bootstrap(idx, families, iters, rng)

    print(f"\n  Effects  (cluster bootstrap over {len(families)} families, "
          f"{iters} iterations)")
    for key, name in (("main_framing", "MAIN framing (narrative - numeric)"),
                      ("main_realism", "MAIN realism (real - fictional)"),
                      ("interaction", "INTERACTION")):
        lo, hi = ci(boot[key])
        print(f"    {name:<38} {_f(est[key])}   95% CI [{lo:>9.5f}, {hi:>9.5f}]")

    lo, hi = ci(boot["interaction"])
    nlo, nhi = ci(naive)
    print(f"\n    naive run-level CI (breaks family pairing; diagnostic only): "
          f"[{nlo:.5f}, {nhi:.5f}]")

    print(f"\n  Per-family interaction (n={len(families)}):")
    for fam, v in sorted(est["per_family_interaction"].items()):
        print(f"    {fam:<34} {_f(v)}")
    spread = max(est["per_family_interaction"].values()) - \
             min(est["per_family_interaction"].values())
    print(f"    {'range across families':<34} {_f(spread)}")

    excludes_zero = (lo > 0) or (hi < 0)
    direction = "positive" if est["interaction"] > 0 else "negative"

    print(f"\n  VERDICT for {metric}:")
    if excludes_zero:
        print(f"    The 95% CI EXCLUDES zero  ->  interaction supported ({direction}).")
        if direction == "positive":
            print("    Framing matters MORE for real shocks than fictional ones —")
            print("    consistent with agents drawing on recall of real events.")
        else:
            print("    Framing matters MORE for fictional shocks than real ones —")
            print("    the opposite of the recall hypothesis; inspect before claiming.")
    else:
        print("    The 95% CI SPANS zero  ->  no evidence of an interaction.")
        print("    The framing effect is realism-independent in this data: these")
        print("    results are consistent with agents responding to shock structure")
        print("    rather than recalling specific real events.")
        print("    (Absence of evidence, not evidence of absence — see caveat below.)")

    return {
        "metric": metric, "estimate": est["interaction"],
        "ci_low": lo, "ci_high": hi,
        "naive_ci_low": nlo, "naive_ci_high": nhi,
        "excludes_zero": bool(excludes_zero),
        "n_families": len(families),
        "main_framing": est["main_framing"],
        "main_realism": est["main_realism"],
    }


def report_balance(runs: list) -> None:
    counts: dict = defaultdict(int)
    fams: dict = defaultdict(set)
    for r in runs:
        counts[(r["framing"], r["realism"])] += 1
        fams[(r["framing"], r["realism"])].add(r["matched_pair_id"])

    print("\n  Runs per cell")
    sizes = []
    for fr in FRAMINGS:
        for rl in REALISMS:
            c = counts[(fr, rl)]
            sizes.append(c)
            print(f"    {fr:<12} x {rl:<10} : {c:>4} runs across "
                  f"{len(fams[(fr, rl)])} families")
    if len(set(sizes)) > 1:
        print("    ** WARNING: cells are unbalanced. The family-weighted estimator")
        print("       tolerates this, but check for systematic run failures.")

    # Degeneracy guard: a backend that was failing produces all-HOLD runs,
    # which analyse as a clean null. That must never be read as a finding.
    forced = [r["metrics"]["forced_random_rate"] for r in runs
              if r["metrics"].get("forced_random_rate") is not None
              and not np.isnan(r["metrics"]["forced_random_rate"])]
    if forced:
        mean_forced = float(np.mean(forced))
        print(f"\n  Mean coin-flip fallback rate: {mean_forced:.1%}")
        if mean_forced > 0.20:
            print("    ** STOP: a fifth or more of all 'trades' are coin flips")
            print("       injected after 3 consecutive holds. They are random by")
            print("       construction, they enter the order book and every")
            print("       metric as real orders, and they dilute any true effect")
            print("       toward zero. Treat a null from this data as broken,")
            print("       not as evidence of no interaction.")
        elif mean_forced > 0.05:
            print("    ** NOTE: some orders in this data are random fallbacks.")

    holds = [r["metrics"]["hold_rate"] for r in runs
             if r["metrics"].get("hold_rate") is not None
             and not np.isnan(r["metrics"]["hold_rate"])]
    if holds:
        mean_hold = float(np.mean(holds))
        print(f"\n  Mean agent hold rate: {mean_hold:.1%}")
        if mean_hold > 0.5:
            print("    ** STOP: over half of all agent decisions produced no order.")
            print("       This usually means the LLM backend was failing, not that")
            print("       agents chose to sit out. An all-HOLD sweep analyses as a")
            print("       clean 'no interaction' result and is indistinguishable")
            print("       from a genuine null. Check a run's `reasoning` fields and")
            print("       the parse-failure log before interpreting ANYTHING below.")
        elif mean_hold > 0.25:
            print("    ** NOTE: elevated hold rate; some decisions may be failing.")
    else:
        print("\n  Mean agent hold rate: not recorded (runs predate hold-rate logging)")


def report_caveats(n_families: int, summary: list) -> None:
    print("\n" + "=" * 84)
    print("  HOW TO READ THIS")
    print("=" * 84)

    if n_families < MIN_CLUSTERS_FOR_INFERENCE:
        print(f"""
  ** THE BINDING CONSTRAINT IS {n_families} EVENT FAMILIES, NOT THE RUN COUNT. **

  The interaction is estimated within a family and averaged across families, so
  the effective sample size for the headline result is n = {n_families} — not the
  total number of runs. Simulation says the cluster bootstrap's coverage is
  about right even at four clusters, so the interval is not broken. What four
  families does cost you is stability and generalisation:

    • The point estimate is the mean of {n_families} numbers. One aberrant family
      moves it substantially. Check the per-family spread printed above.
    • Coverage was verified under Gaussian family effects. If the true family
      effects are skewed or heavy-tailed — one commodity behaving unlike the
      rest — a percentile interval from {n_families} clusters is not protected.
    • "Generalises to supply chain shocks" is a claim about the population of
      event families. Four is thin support for that claim regardless of what
      the interval says.

  Adding repetitions per shock will NOT help. Reps shrink within-family noise
  only; the interaction's variance is dominated by between-family spread. The
  high-value change is more event FAMILIES — add matched real/fictional pairs
  in generate_fictional_shocks.py (aim for 20+) and re-run. That is a change to
  the generator, not the runner or this script.

  Either way, report the per-family interaction values alongside the interval
  so readers can see the heterogeneity directly.""")
    else:
        print(f"\n  {n_families} families — cluster bootstrap is on reasonable footing.")

    print("""
  Other caveats worth stating in the write-up:

  • The numeric framing carries realism only in an entity token; every number
    is identical within a matched pair. The realism manipulation is therefore
    weaker in the numeric cells. A non-zero interaction may partly reflect that
    asymmetry in manipulation strength rather than a genuine moderating effect.

  • These are percentile intervals. Simulation put their coverage near nominal
    for this design, but BCa is the better choice if the bootstrap
    distribution turns out skewed — worth checking before final submission.

  • No multiplicity adjustment is applied across metrics. Three primary metrics
    are tested; if you intend to claim any single one, say which was
    pre-specified.""")

    if summary:
        print("\n" + "-" * 84)
        print("  ONE-LINE SUMMARY (primary metrics)")
        print("-" * 84)
        for s in summary:
            if s["metric"] not in PRIMARY:
                continue
            verdict = "INTERACTION" if s["excludes_zero"] else "no interaction"
            print(f"    {s['metric']:<24} {s['estimate']:>10.5f}  "
                  f"[{s['ci_low']:>9.5f}, {s['ci_high']:>9.5f}]   {verdict}")


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def report_model_health(by_model: dict) -> None:
    """Per-model run counts and hold rates.

    A model that holds far more often than another is not necessarily more
    cautious — small local models fail to emit parseable JSON at very different
    rates, and a failed parse is recorded as a hold. If the hold rates differ
    materially, an apparent "model-specific effect" may be a parse-reliability
    artefact, so it is checked before any comparison is read.
    """
    print("\n  Per-model coverage and decision health")
    print(f"    {'model':<20} {'runs':>5} {'fam':>4} {'hold':>7} "
          f"{'coinflip':>9} {'backend fail':>13}")

    def _mean_metric(runs: list, key: str) -> float:
        vals = [r["metrics"][key] for r in runs
                if r["metrics"].get(key) is not None
                and not np.isnan(r["metrics"][key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _show(v: float) -> str:
        return f"{v:.1%}" if not np.isnan(v) else "n/a"

    holds, forced, failed = {}, {}, {}
    legacy = 0
    for model in sorted(by_model):
        runs = by_model[model]
        fams = {r["matched_pair_id"] for r in runs}
        holds[model] = _mean_metric(runs, "hold_rate")
        forced[model] = _mean_metric(runs, "forced_random_rate")
        failed[model] = _mean_metric(runs, "api_failure_rate")
        legacy += sum(1 for r in runs if "forced_random_rate" not in r["metrics"])
        print(f"    {model:<20} {len(runs):>5} {len(fams):>4} "
              f"{_show(holds[model]):>7} {_show(forced[model]):>9} "
              f"{_show(failed[model]):>13}")

    # Coin-flip fallbacks are random orders that enter the book and the metrics
    # looking exactly like real trades. They dilute any true effect toward zero,
    # so a high rate makes a null result uninterpretable rather than negative.
    for model, fr in forced.items():
        if np.isnan(fr):
            continue
        if fr > 0.20:
            print(f"\n    ** STOP ({model}): {fr:.1%} of decisions are coin-flip")
            print("       fallbacks — random orders, not decisions. They enter the")
            print("       metrics as real trades and dilute any real effect toward")
            print("       zero. A null from this data is uninterpretable.")
        elif fr > 0.05:
            print(f"\n    ** NOTE ({model}): {fr:.1%} coin-flip fallbacks — some")
            print("       trades in this data are random.")

    for model, af in failed.items():
        if not np.isnan(af) and af > 0.05:
            print(f"\n    ** WARNING ({model}): {af:.1%} backend failures. These are")
            print("       recorded as holds, so they understate participation and")
            print("       can be mistaken for cautious agents.")

    def _gap_warn(d: dict, label: str, why: list) -> None:
        valid = {m: v for m, v in d.items() if not np.isnan(v)}
        if len(valid) < 2:
            return
        gap = max(valid.values()) - min(valid.values())
        if gap > 0.15:
            hi = max(valid, key=valid.get)
            print(f"\n    ** WARNING: {label} differ by {gap:.1%} across models "
                  f"({hi} highest).")
            for line in why:
                print(f"       {line}")

    _gap_warn(holds, "hold rates", [
        "A parse failure is logged as a hold, so the models may differ in",
        "JSON reliability rather than in trading behaviour. Check",
        "parse_failures.jsonl (it records the model per failure) before",
        "attributing any cross-model difference to the model's economics.",
    ])
    _gap_warn(forced, "coin-flip rates", [
        "The model with more fallbacks has more random orders in its data,",
        "which biases ITS interaction estimate toward zero specifically.",
        "That asymmetry can manufacture an apparent model-specific effect.",
    ])

    if legacy:
        print(f"\n    Note: {legacy} run(s) predate coin-flip/backend telemetry;")
        print("    their rates are excluded from the means above.")
    if any(np.isnan(v) for v in holds.values()):
        print("\n    Note: some runs predate hold-rate logging entirely.")


def compare_metric_across_models(metric: str, label: str, by_model: dict,
                                 models: list, iters: int,
                                 rng: np.random.Generator) -> Optional[dict]:
    """Interaction analysis per model, plus paired difference and combined."""
    idx_by_model = {m: index_runs(by_model[m], metric) for m in models}
    fam_by_model = {m: complete_families(idx_by_model[m]) for m in models}

    print("\n" + "=" * 84)
    print(f"  {label}   [{metric}]")
    print("=" * 84)

    usable = [m for m in models if len(fam_by_model[m]) >= 2]
    if len(usable) < 2:
        print(f"  Fewer than 2 models have >=2 complete families "
              f"({ {m: len(fam_by_model[m]) for m in models} }) — skipping.")
        return None

    # Per-model marginal estimates, each on that model's own complete families.
    per_model = {}
    print(f"\n  Interaction, estimated separately per model "
          f"(cluster bootstrap, {iters} iters)")
    print(f"    {'model':<22} {'fam':>4} {'estimate':>11}   "
          f"{'95% CI':^24}   {'vs zero':>9}")
    for m in usable:
        idx, fams = idx_by_model[m], fam_by_model[m]
        est = point_estimates(idx, fams)
        boot = cluster_bootstrap(idx, fams, iters, rng)
        lo, hi = ci(boot["interaction"])
        excl = (lo > 0) or (hi < 0)
        per_model[m] = {
            "estimate": est["interaction"], "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(excl), "n_families": len(fams),
            "per_family_interaction": est["per_family_interaction"],
        }
        print(f"    {m:<22} {len(fams):>4} {est['interaction']:>11.5f}   "
              f"[{lo:>10.5f}, {hi:>10.5f}]   "
              f"{'excludes' if excl else 'spans':>9}")

    # --- CI overlap (the crude check) -------------------------------------
    # The pairwise contrast is reported for the first two models only; the
    # combined estimate below still averages all of them. With more than two
    # models, say so rather than letting the omission pass silently.
    if len(usable) > 2:
        print(f"\n    Note: {len(usable)} models present. The pairwise contrast below "
              f"covers\n    {usable[0]} vs {usable[1]} only; the combined estimate "
              f"averages all {len(usable)}.")
    a, b = usable[0], usable[1]
    pa, pb = per_model[a], per_model[b]
    overlap = not (pa["ci_high"] < pb["ci_low"] or pb["ci_high"] < pa["ci_low"])

    print(f"\n  Marginal CI overlap ({a} vs {b}): "
          f"{'OVERLAP' if overlap else 'NO OVERLAP'}")
    if overlap:
        print("    Consistent with replication. Note this is a weak test: two")
        print("    intervals can overlap while their difference is still")
        print("    distinguishable from zero, so read the paired contrast below")
        print("    rather than stopping here.")
    else:
        print("    The intervals are disjoint — the models are clearly")
        print("    inconsistent on this metric. Flag this; do not average it away.")

    # --- Paired difference (the sharper test) ------------------------------
    shared = sorted(set(fam_by_model[a]) & set(fam_by_model[b]))
    diff = comb = None
    diff_ci = comb_ci = (float("nan"), float("nan"))
    diff_excludes = False

    if len(shared) < 2:
        print(f"\n  Only {len(shared)} family is complete in both models — the")
        print("    paired difference and combined estimate are not computed.")
    else:
        if len(shared) < max(len(fam_by_model[a]), len(fam_by_model[b])):
            print(f"\n    Note: {len(shared)} of "
                  f"{max(len(fam_by_model[a]), len(fam_by_model[b]))} families are"
                  f" complete in both models; the paired contrast uses those "
                  f"{len(shared)}.")

        dists, comb_dist = cross_model_bootstrap(
            idx_by_model, usable, shared, iters, rng)
        diff_dist = dists[a] - dists[b]

        fcm_a = family_cell_means(idx_by_model[a], shared)
        fcm_b = family_cell_means(idx_by_model[b], shared)
        int_a = float(np.mean([family_interaction(fcm_a[f]) for f in shared]))
        int_b = float(np.mean([family_interaction(fcm_b[f]) for f in shared]))
        diff = int_a - int_b
        comb = float(np.mean([int_a, int_b]))

        diff_ci = ci(diff_dist)
        comb_ci = ci(comb_dist)
        diff_excludes = (diff_ci[0] > 0) or (diff_ci[1] < 0)

        print(f"\n  Paired difference ({a} - {b}), {len(shared)} shared families")
        print(f"    {'difference':<22} {diff:>11.5f}   "
              f"[{diff_ci[0]:>10.5f}, {diff_ci[1]:>10.5f}]   "
              f"{'excludes' if diff_excludes else 'spans':>9} zero")
        print("    Families are drawn once per iteration and both models are")
        print("    evaluated on that draw, so the shared family variance cancels.")

        print(f"\n  Combined ({len(usable)} models averaged)  [DESCRIPTIVE — see below]")
        print(f"    {'combined':<22} {comb:>11.5f}   "
              f"[{comb_ci[0]:>10.5f}, {comb_ci[1]:>10.5f}]")

    # --- Classification ----------------------------------------------------
    both_excl = pa["excludes_zero"] and pb["excludes_zero"]
    same_sign = np.sign(pa["estimate"]) == np.sign(pb["estimate"])
    neither = not pa["excludes_zero"] and not pb["excludes_zero"]

    if diff_excludes:
        status = "model-specific"
    elif both_excl and same_sign:
        status = "replicates"
    elif neither:
        status = "null-both"
    elif both_excl and not same_sign:
        status = "opposite-signs"
    else:
        status = "inconclusive"

    print(f"\n  READING for {metric}:")
    if status == "model-specific":
        bigger = a if diff > 0 else b
        print("    The models DIFFER: the paired difference excludes zero. The")
        print(f"    interaction is larger on {bigger}. This is a model-specific")
        print("    effect. It is a legitimate result — report it as one, with")
        print("    both per-model estimates, rather than pooling it into an")
        print("    average that describes neither model.")
    elif status == "replicates":
        print("    Both models show an interaction in the same direction and the")
        print("    paired difference is not distinguishable from zero — the")
        print("    effect REPLICATES across these two models.")
    elif status == "null-both":
        print("    Neither model shows an interaction and they are not")
        print("    distinguishable from each other. The null replicates. That is")
        print("    a result about consistency, not evidence that the effect is")
        print("    absent — see the power caveats.")
    elif status == "opposite-signs":
        print("    Both CIs exclude zero but in OPPOSITE directions. Do not")
        print("    average these. Inspect the per-family values and the shock")
        print("    texts before writing anything up.")
    else:
        print("    One model's CI excludes zero and the other's does not, but the")
        print("    paired difference spans zero. This is NOT evidence that the")
        print("    models disagree — a difference in significance is not a")
        print("    significant difference. The honest reading is that the data")
        print("    cannot separate replication from a model-specific effect.")

    return {
        "metric": metric,
        "per_model": {m: {k: v for k, v in d.items()
                          if k != "per_family_interaction"}
                      for m, d in per_model.items()},
        "models": usable,
        "ci_overlap": bool(overlap),
        "n_shared_families": len(shared),
        "difference": diff,
        "difference_ci": list(diff_ci),
        "difference_excludes_zero": bool(diff_excludes),
        "combined": comb,
        "combined_ci": list(comb_ci),
        "status": status,
    }


def report_model_caveats(models: list, results: list) -> None:
    print("\n" + "=" * 84)
    print("  HOW TO READ THE CROSS-MODEL COMPARISON")
    print("=" * 84)

    n = len(models)
    if n < MIN_MODELS_FOR_INFERENCE:
        print(f"""
  ** n = {n} MODELS IS DESCRIPTIVE, NOT INFERENTIAL. **

  The combined estimate averages the per-model interactions. It does NOT treat
  models as a sample: you cannot resample {n} clusters with replacement and get
  a meaningful interval, so the model dimension contributes no uncertainty to
  it. Concretely, the combined CI answers "what would this average do on new
  event families?" and NOT "what would it do on a new model?"

  This is the same constraint as the four-families caveat, one level up:

    • 4 families  -> weak support for "generalises to supply chain shocks"
    • {n} models    -> weak support for "generalises to language models"

  Both models here are small local models served through Ollama. Agreement
  between them is evidence about *replication* — same design, second model —
  and is much weaker evidence about generalisation across model families,
  which would need models that differ in training data and scale.

  So: report the per-model estimates as the primary result and the combined
  number as a summary statistic. Do not present the combined CI as though the
  model dimension had been accounted for in it.""")
    else:
        print(f"\n  {n} models — a model-level cluster bootstrap becomes defensible.")

    print("""
  On the two ways this can come out:

  • REPLICATION and MODEL-SPECIFICITY are both findings. An interaction that
    holds on one model and not another says something real about how that model
    processes shock framing. The failure mode to avoid is running a second
    model, disliking the answer, and reporting only the first.

  • Before calling a difference model-specific, rule out the mundane causes:
    different hold/parse-failure rates (printed above), a model that was never
    pulled, and different sensitivity to prompt formatting. These produce
    cross-model differences that have nothing to do with recall of real events.

  • The per-model intervals are subject to every caveat in the single-model
    analysis — four families, percentile intervals, no multiplicity adjustment.
    Comparing two such intervals does not repair any of them.""")

    if results:
        print("\n" + "-" * 84)
        print("  PLAIN-LANGUAGE SUMMARY (primary metrics)")
        print("-" * 84)
        wording = {
            "replicates":     "replicates on both models",
            "model-specific": "MODEL-SPECIFIC — models differ",
            "null-both":      "no interaction on either model",
            "opposite-signs": "OPPOSITE directions — inspect",
            "inconclusive":   "cannot separate the two readings",
        }
        primary = [r for r in results if r["metric"] in PRIMARY]
        for r in primary:
            per_m = "  ".join(
                f"{m}={r['per_model'][m]['estimate']:.5f}" for m in r["models"]
            )
            print(f"\n    {r['metric']}: {wording[r['status']]}")
            print(f"      {per_m}")
            if r["difference"] is not None:
                print(f"      difference {r['difference']:>.5f} "
                      f"[{r['difference_ci'][0]:.5f}, {r['difference_ci'][1]:.5f}]"
                      f"   combined {r['combined']:.5f} (descriptive)")

        counts = defaultdict(int)
        for r in primary:
            counts[r["status"]] += 1

        print("\n  Overall, across the primary metrics:")
        if not primary:
            print("    No primary metric had enough data to compare.")
        elif counts["replicates"] == len(primary):
            print("    The interaction replicates across both models on every")
            print("    primary metric.")
        elif counts["null-both"] == len(primary):
            print("    Neither model shows an interaction on any primary metric,")
            print("    and the models agree with each other.")
        elif counts["model-specific"]:
            print(f"    {counts['model-specific']} of {len(primary)} primary metrics show a "
                  f"cross-model difference that")
            print("    excludes zero. On those the effect is model-specific and")
            print("    should be reported that way — do not lead with the")
            print("    combined number for them.")
        else:
            print("    The primary metrics are mixed and no cross-model difference")
            print("    is distinguishable from zero. The data support neither a")
            print("    clean replication claim nor a clean model-specific claim.")
        print("\n    Both replication and model-specificity are reportable outcomes.")
        print("    Whichever came out, state it and show the per-model numbers.")


def compare_models(runs: list, iters: int = DEFAULT_ITERS,
                   seed: int = BOOTSTRAP_SEED,
                   metrics: Optional[list] = None) -> list:
    """Run the interaction analysis per model and compare them side by side."""
    by_model = split_by_model(runs)
    models = sorted(by_model)

    print("\n" + "=" * 84)
    print(f"  CROSS-MODEL COMPARISON — {len(models)} model(s): {', '.join(models)}")
    print("=" * 84)

    if len(models) < 2:
        print(f"\n  Only one model present ({models[0] if models else 'none'}).")
        print("  Run the second sweep first:")
        print("    python -m experiments.design_2x2 --all-models --reps 5")
        return []

    if UNKNOWN_MODEL in models:
        print(f"\n  ** Some runs have no recorded model (grouped as "
              f"{UNKNOWN_MODEL!r}).")
        print("     These predate per-model results. Treat that group with care —")
        print("     it may mix models.")

    report_model_health(by_model)

    rng = np.random.default_rng(seed)
    results = []
    for metric, label, _note in (metrics or METRICS):
        res = compare_metric_across_models(
            metric, label, by_model, models, iters, rng)
        if res:
            results.append(res)

    report_model_caveats(models, results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Analyse the 2x2 framing x realism design.")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help=f"Results directory (default {DEFAULT_RESULTS}).")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                    help=f"Bootstrap iterations (default {DEFAULT_ITERS}).")
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED,
                    help="Bootstrap RNG seed (for reproducibility).")
    ap.add_argument("--out", default=None,
                    help="Optional JSON path for the summary.")
    ap.add_argument("--model", default=None,
                    help="Analyse only this model tag (e.g. llama3.1:8b). "
                         "Default: pool every model found — which is only "
                         "meaningful if you have checked they agree, so use "
                         "--compare-models first.")
    ap.add_argument("--compare-models", action="store_true",
                    help="Run the interaction analysis per model and compare "
                         "them side by side (replication check).")
    args = ap.parse_args(argv)

    runs = load_runs(args.results, model=args.model)
    if not runs:
        where = f"{args.results!r}"
        if args.model:
            where += f" for model {args.model!r}"
        print(f"No runs found in {where}.")
        print("Run: python -m experiments.design_2x2 --all-models --reps 5")
        return 1

    found = models_present(runs)

    print("=" * 84)
    print(f"  2x2 ANALYSIS — {len(runs)} runs from {args.results}/")
    print(f"  models: {', '.join(found)}"
          + (f"   (filtered to {args.model})" if args.model else ""))
    print("=" * 84)

    if args.compare_models:
        results = compare_models(runs, iters=args.iters, seed=args.seed)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump({"mode": "compare_models", "n_runs": len(runs),
                           "models": found, "iters": args.iters,
                           "seed": args.seed, "results": results}, fh, indent=2)
            print(f"\n  Summary → {args.out}")
        print("=" * 84)
        return 0

    if len(found) > 1:
        print("\n  ** These runs span more than one model and are being POOLED.")
        print("     Pooling assumes the models behave alike — which is the very")
        print("     thing to be tested. Run --compare-models before reporting")
        print("     anything below, or restrict with --model.")

    report_balance(runs)

    rng = np.random.default_rng(args.seed)
    summary = []
    n_fam = 0
    for metric, label, note in METRICS:
        res = report_metric(metric, label, note, runs, args.iters, rng)
        if res:
            summary.append(res)
            n_fam = max(n_fam, res["n_families"])

    report_caveats(n_fam, summary)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"n_runs": len(runs), "n_families": n_fam,
                       "models": found, "model_filter": args.model,
                       "iters": args.iters, "seed": args.seed,
                       "results": summary}, fh, indent=2)
        print(f"\n  Summary → {args.out}")

    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
