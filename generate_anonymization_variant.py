#!/usr/bin/env python3
"""
Causal-mediation Step 3 — interventional test: manipulate the mediator M directly.

Step 2 found the natural indirect effect of framing through perceived realism M
spans zero (associational null). Step 3 upgrades that to an interventional test:
force M downward on the REAL-NARRATIVE shocks with an explicit instruction, then
check whether behavior moves. If M shifts but behavior does not, that is
interventional confirmation the belief channel carries no causal weight.

Design (higher-quality than a bare single-arm test):

  * TWO instruction arms, appended to the real-narrative shock text:
      anon    — asserts the scenario is synthetic / not a real event  (moves M down)
      placebo — same "for this evaluation" framing, NO reality claim   (control)
    anon vs placebo isolates the *belief* manipulation from the mere
    "you are being evaluated" framing, which a single anon arm confounds.

  * SAME SEEDS as the existing real-narrative runs. make_seed(family, rep) depends
    only on (family, rep), so a variant with the same family+rep faces the
    identical market path — the comparison is PAIRED, cancelling market noise.
    The original real-narrative cell in results_2x2/ is reused as the baseline;
    it is not re-run.

  * INTERVENTION MANIPULATION CHECK: every new run is probed (perceived realism)
    exactly as in Step 1, so we can verify the instruction actually moved M
    before reading any behavioral result. Expect it to land for llama+narrative
    and likely NOT for mistral (which called ~everything real in Step 1) — that
    is reported, not forced.

  * Behavioral comparison uses the SAME family-cluster bootstrap as analyze_2x2,
    on the PAIRED per-(family,rep) differences.

Outputs:
  shocks_step3.json                                    (the variant stimuli)
  results_2x2_step3/{model}/{arm}/{event}__repNN.json  (runs + attached M)

Usage:
  python3 generate_anonymization_variant.py --generate        # write variants, show them
  python3 generate_anonymization_variant.py --smoke           # 1 shock/1 rep/llama, both arms: does M move?
  python3 generate_anonymization_variant.py --run --all-models --reps 5
  python3 generate_anonymization_variant.py --report          # manip check + behavioral CIs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from experiments.design_2x2 import (
    run_one, load_shocks, make_seed, model_slug, SHOCK_TICK,
)
from analyze_2x2 import load_runs, ci, BOOTSTRAP_SEED, DEFAULT_ITERS
import probe_perceived_realism as probe

N_TICKS = 20
ARMS = ("anon", "placebo")
STEP3_DIR = "results_2x2_step3"
STEP1_MEDIATOR = "results_2x2_mediator"
VARIANTS_FILE = "shocks_step3.json"
PRIMARY = [
    ("spread_widening", "Spread widening"),
    ("kyle_lambda_delta", "Kyle's lambda change"),
    ("archetype_divergence", "Archetype divergence"),
]
ALL_MODELS = ["llama3.1:8b", "mistral:7b"]

# The two instructions. Same "for this evaluation" frame so the ONLY systematic
# difference is the assertion that the scenario is not real — that assertion is
# the mediator manipulation; the shared frame is what the placebo controls for.
ANON_INSTRUCTION = (
    "Note for this evaluation: treat the scenario described above as fully "
    "anonymized, synthetic data generated for testing purposes. It does not "
    "describe any real event that has actually occurred in the world."
)
PLACEBO_INSTRUCTION = (
    "Note for this evaluation: read the scenario described above carefully and "
    "weigh it thoroughly on its own terms as you form your assessment for this "
    "exercise."
)
INSTRUCTION = {"anon": ANON_INSTRUCTION, "placebo": PLACEBO_INSTRUCTION}


def real_narrative_shocks() -> list:
    """The 4 real x narrative shocks (one per family) — the intervention targets."""
    out = [s for s in load_shocks()
           if s.get("realism") == "real" and s.get("framing") == "narrative"]
    return sorted(out, key=lambda s: s["matched_pair_id"])


def variant_shock(base: dict, arm: str) -> dict:
    """A real-narrative shock with an instruction appended to its scenario text.

    True realism/framing labels are unchanged — we are moving *perceived* realism,
    not relabelling the design. matched_pair_id is preserved so the seed (and thus
    the market path) matches the original real-narrative run for this family+rep.
    """
    v = dict(base)
    v["text"] = base["text"] + "\n\n" + INSTRUCTION[arm]
    v["shock_id"] = f"{base['shock_id']}__{arm}"
    v["arm"] = arm
    return v


def write_variants() -> list:
    variants = []
    for base in real_narrative_shocks():
        for arm in ARMS:
            variants.append(variant_shock(base, arm))
    with open(VARIANTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(variants, fh, indent=2, ensure_ascii=False)
    return variants


def step3_path(model: str, arm: str, event_key: str, rep: int) -> str:
    return os.path.join(STEP3_DIR, model_slug(model), arm,
                        f"{event_key}__rep{rep:02d}.json")


# ---------------------------------------------------------------------------
# Run + probe
# ---------------------------------------------------------------------------

def _probe_M(model: str, shock_text: str, pbackend) -> tuple:
    """Probe perceived realism of one scenario text: (M, tally). scope=all (60)."""
    probes = []
    for _tick in range(N_TICKS):
        for arch in probe.LLM_ARCHETYPES:
            perc, _raw, _dt = probe._one_probe(pbackend, arch, shock_text)
            probes.append(perc)
    tally = Counter(probes)
    real = tally.get("real", 0)
    tot = real + tally.get("hypothetical", 0)
    return (real / tot if tot else None), dict(tally)


def run_variants(models: list, reps: int, arms=ARMS, resume: bool = True,
                 limit: Optional[int] = None) -> dict:
    shocks = real_narrative_shocks()
    total = len(models) * len(arms) * len(shocks) * reps
    done = 0
    executed = 0
    print(f"  Step 3 runs: {len(models)} models x {len(arms)} arms x "
          f"{len(shocks)} shocks x {reps} reps = {total}")
    for model in models:
        pbackend = probe._make_backend(model)
        m_cache: dict = {}   # (arm, event_key) -> (M, tally): M depends on text only
        for arm in arms:
            for base in shocks:
                v = variant_shock(base, arm)
                for rep in range(reps):
                    path = step3_path(model, arm, base["event_key"], rep)
                    done += 1
                    if resume and os.path.exists(path):
                        continue
                    result = run_one(v, rep, N_TICKS, SHOCK_TICK, model=model)
                    key = (arm, base["event_key"])
                    if key not in m_cache:               # probe once per unique text
                        m_cache[key] = _probe_M(model, v["text"], pbackend)
                    M, tally = m_cache[key]
                    result["arm"] = arm
                    result["perceived_M"] = M
                    result["probe_tally"] = tally
                    result["baseline_shock_id"] = base["shock_id"]
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump(result, fh, indent=2)
                    executed += 1
                    print(f"\r    {done}/{total} ({model} {arm} "
                          f"{base['event_key']} r{rep}) executed={executed}",
                          end="", flush=True)
                    if limit and executed >= limit:
                        print(f"\n  --limit {limit} reached.")
                        return {"executed": executed}
    print(f"\n  Step 3 done: {executed} new runs -> {STEP3_DIR}/")
    return {"executed": executed}


# ---------------------------------------------------------------------------
# Analysis: intervention manipulation check + paired behavioral comparison
# ---------------------------------------------------------------------------

def _load_step3(model: str, arm: str) -> list:
    out = []
    for p in glob.glob(os.path.join(STEP3_DIR, model_slug(model), arm, "*.json")):
        try:
            out.append(json.load(open(p, encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _baseline_M(model: str) -> dict:
    """{event_key: mean M} for the ORIGINAL real-narrative runs, from Step 1."""
    byev: dict = defaultdict(list)
    for p in glob.glob(os.path.join(STEP1_MEDIATOR, model_slug(model), "*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("framing") == "narrative" and d.get("realism") == "real":
            real = sum(1 for x in d["probes"] if x["perceived"] == "real")
            tot = sum(1 for x in d["probes"] if x["perceived"] in ("real", "hypothetical"))
            if tot:
                byev[d["event_key"]].append(real / tot)
    return {k: float(np.mean(v)) for k, v in byev.items()}


def _paired_family_bootstrap(pairs: dict, iters: int, rng) -> tuple:
    """pairs: {family: [per-rep diff]}. Resample families, then reps within family."""
    fams = [f for f in pairs if pairs[f]]
    nf = len(fams)
    if nf == 0:
        return float("nan"), (float("nan"), float("nan"))
    point = float(np.mean([np.mean(pairs[f]) for f in fams]))
    dist = []
    for _ in range(iters):
        drawn = rng.integers(0, nf, size=nf)
        fam_means = []
        for di in drawn:
            vals = pairs[fams[di]]
            idx = rng.integers(0, len(vals), size=len(vals))
            fam_means.append(np.mean([vals[i] for i in idx]))
        dist.append(float(np.mean(fam_means)))
    return point, ci(np.array(dist))


def _diff_pairs(a_runs: list, b_runs: list, metric: str) -> dict:
    """{family: [a-b per (family,rep)]} joined on (event_key, rep)."""
    b_idx = {(r["event_key"], r["rep"]): r for r in b_runs}
    pairs: dict = defaultdict(list)
    for ra in a_runs:
        rb = b_idx.get((ra["event_key"], ra["rep"]))
        if rb is None:
            continue
        ya = ra["metrics"].get(metric)
        yb = rb["metrics"].get(metric)
        if ya is None or yb is None:
            continue
        if np.isnan(ya) or np.isnan(yb):
            continue
        pairs[ra["matched_pair_id"]].append(float(ya) - float(yb))
    return pairs


def report(models: list, iters: int = DEFAULT_ITERS) -> None:
    print("=" * 84)
    print("  STEP 3 — INTERVENTIONAL MEDIATION TEST (real-narrative)")
    print("=" * 84)

    for model in models:
        anon = _load_step3(model, "anon")
        placebo = _load_step3(model, "placebo")
        if not anon:
            print(f"\n  [{model}] no step-3 runs yet.")
            continue
        base_orig = load_runs("results_2x2", model)
        orig = [r for r in base_orig
                if r["framing"] == "narrative" and r["realism"] == "real"]
        base_M = _baseline_M(model)

        print("\n" + "=" * 84)
        print(f"  MODEL: {model}")
        print("=" * 84)

        # ---- (1) intervention manipulation check ----
        def _M(runs):
            v = [r["perceived_M"] for r in runs if r.get("perceived_M") is not None]
            return float(np.mean(v)) if v else float("nan")
        M_anon, M_plac = _M(anon), _M(placebo)
        M_orig = float(np.mean(list(base_M.values()))) if base_M else float("nan")
        print("  Intervention manipulation check — perceived realism M (P='real'):")
        print(f"    original real-narrative : {M_orig:.1%}   (from Step 1)")
        print(f"    + placebo instruction   : {M_plac:.1%}")
        print(f"    + anon instruction      : {M_anon:.1%}")
        shift = M_anon - M_orig
        print(f"    anon shift vs original  : {shift:+.1%}")
        landed = (not np.isnan(shift)) and shift <= -0.15
        if landed:
            print("    -> MANIPULATION LANDED: anon lowered perceived realism. Behavioral")
            print("       comparison below is interpretable.")
        else:
            print("    -> MANIPULATION DID NOT LAND (M barely moved). The behavioral")
            print("       comparison for this model is UNINFORMATIVE — you cannot test")
            print("       the effect of moving M if M did not move.")

        # ---- (2) paired behavioral comparison ----
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        print("\n  Paired behavioral effect (family-cluster bootstrap, "
              f"{iters} iters, same seeds):")
        for metric, label in PRIMARY:
            ao = _diff_pairs(anon, orig, metric)       # anon - original (belief+frame)
            ap = _diff_pairs(anon, placebo, metric)    # anon - placebo  (belief only)
            po, ci_ao = _paired_family_bootstrap(ao, iters, rng)
            pp, ci_ap = _paired_family_bootstrap(ap, iters, rng)
            tag_ao = "excludes" if (ci_ao[0] > 0 or ci_ao[1] < 0) else " spans "
            tag_ap = "excludes" if (ci_ap[0] > 0 or ci_ap[1] < 0) else " spans "
            print(f"\n    {label} [{metric}]")
            print(f"      anon - original : {po:+.5f}  [{ci_ao[0]:+.5f}, {ci_ao[1]:+.5f}]  {tag_ao}")
            print(f"      anon - placebo  : {pp:+.5f}  [{ci_ap[0]:+.5f}, {ci_ap[1]:+.5f}]  {tag_ap}  <- isolates belief")

        # ---- reading ----
        print("\n  READING:")
        if not landed:
            print(f"    {model}: mediator not manipulable — inconclusive (as Step 1 foretold).")
        else:
            print("    If 'anon - placebo' spans zero on every metric, behavior did NOT")
            print("    move when belief was forced down while holding the eval-frame fixed:")
            print("    interventional CONFIRMATION that perceived realism carries no causal")
            print("    weight. If it excludes zero, that is a real mediated effect.")
    print("\n" + "=" * 84)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 3 interventional mediation.")
    ap.add_argument("--generate", action="store_true", help="write shocks_step3.json and show")
    ap.add_argument("--run", action="store_true", help="execute the variant runs + probes")
    ap.add_argument("--report", action="store_true", help="manip check + behavioral CIs")
    ap.add_argument("--smoke", action="store_true",
                    help="1 shock x 1 rep on llama, both arms — does M move?")
    ap.add_argument("--model", default=None)
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    models = ALL_MODELS if args.all_models else [args.model or "llama3.1:8b"]

    if args.generate or args.smoke or args.run:
        variants = write_variants()
        print(f"  wrote {len(variants)} variants -> {VARIANTS_FILE}")
        if args.generate and not (args.smoke or args.run):
            for arm in ARMS:
                print(f"\n  --- {arm} instruction ---\n  {INSTRUCTION[arm]}")
            print(f"\n  {len(real_narrative_shocks())} real-narrative shocks targeted; "
                  f"seeds pair to existing results_2x2 runs.")
            return 0

    if args.smoke:
        base = real_narrative_shocks()[:1]
        print("\n  SMOKE: 1 shock x 1 rep on llama3.1:8b, both arms "
              "(validates pipeline + whether anon moves M).")
        # temporarily restrict shocks by monkeypatching not needed; run first shock/rep
        pbackend = probe._make_backend("llama3.1:8b")
        base_M = _baseline_M("llama3.1:8b")
        for arm in ARMS:
            v = variant_shock(base[0], arm)
            result = run_one(v, 0, N_TICKS, SHOCK_TICK, model="llama3.1:8b")
            M, tally = _probe_M("llama3.1:8b", v["text"], pbackend)
            result.update({"arm": arm, "perceived_M": M, "probe_tally": tally,
                           "baseline_shock_id": base[0]["shock_id"]})
            path = step3_path("llama3.1:8b", arm, base[0]["event_key"], 0)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(result, open(path, "w"), indent=2)
            print(f"    {arm}: perceived_M={M:.1%}  tally={tally}")
        bm = base_M.get(base[0]["event_key"], float("nan"))
        print(f"    original M for {base[0]['event_key']} (Step 1): {bm:.1%}")
        print("    -> if anon M is well below original/placebo, the manipulation lands.")
        return 0

    if args.run:
        run_variants(models, args.reps, resume=not args.no_resume)
        print("\n  Next: python3 generate_anonymization_variant.py --report")
        return 0

    if args.report:
        report(models, iters=args.iters)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
