#!/usr/bin/env python3
"""
Formal power analysis for the 2x2 design (rigor-closing Step 3).

The family count is read from the results, not assumed: the design grew from 4
to 10 families, and every n-dependent number below (SE, MDE, n-for-half-MDE,
the section header) follows from that count.

No new LLM calls. Uses the existing cluster-bootstrap machinery to turn "n is
small" into concrete minimum-detectable-effect (MDE) numbers.

Method. The bootstrap distribution of an estimate is its sampling distribution;
its standard deviation is the standard error (SE). For a two-sided test at
alpha=0.05 with 80% power,

    MDE = (z_{0.975} + z_{0.80}) * SE = (1.960 + 0.842) * SE = 2.802 * SE.

An estimator that averages a contrast over families has SE proportional to
1/sqrt(n_families) once between-family variance dominates (the script prints the
variance split so you can check it still does). So to detect an effect of
size e instead of the current MDE,

    n_needed = n_current * (MDE_current / e)^2 ,   e.g. half the MDE -> 4x families.

Covers both the framing x realism INTERACTION (analyze_2x2.cluster_bootstrap) and
the natural INDIRECT effect of framing (analyze_mediation.mediation_bootstrap).

Usage:
    python3 power_analysis.py                 # print
    python3 power_analysis.py --append        # also append section to RESULTS_2x2.md
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Optional

import numpy as np

from analyze_2x2 import (
    load_runs, split_by_model, index_runs, complete_families, cluster_bootstrap,
    point_estimates, family_interaction, family_cell_means,
    DEFAULT_RESULTS, DEFAULT_ITERS, BOOTSTRAP_SEED,
)
from analyze_mediation import (
    load_mediator_M, build_records, mediation_bootstrap, METRICS,
)

Z = 1.959964 + 0.841621          # 80% power, two-sided alpha=0.05

# The rendered header carries the family count, so it changes when n does. Match
# on the stem instead: keying the replace on the full literal would have missed
# the existing "(n = 4 families)" section once the design grew to 10 and appended
# a second, contradictory power section below the stale one.
SECTION_STEM = "## Statistical Power"
SECTION_RE = re.compile(r"^## Statistical Power.*$", re.M)


def section_header(n_families: int) -> str:
    return f"{SECTION_STEM} (n = {n_families} families)"


def _se(dist: np.ndarray) -> float:
    return float(np.std(dist, ddof=1)) if len(dist) > 1 else float("nan")


def _mde(se: float) -> float:
    return Z * se


def _n_for_effect(n_cur: int, mde_cur: float, target: float) -> float:
    """Families needed to make MDE == target (family-mean SE ~ 1/sqrt(n))."""
    if target <= 0 or np.isnan(mde_cur):
        return float("nan")
    return n_cur * (mde_cur / target) ** 2


def variance_split(idx: dict, families: list) -> tuple:
    """(between-family var, within-family mean var) of the interaction contrast."""
    fcm = family_cell_means(idx, families)
    per_fam = np.array([family_interaction(fcm[f]) for f in families])
    between = float(np.var(per_fam, ddof=1)) if len(per_fam) > 1 else float("nan")
    # crude within: variance of run values pooled within each family-cell, averaged
    within = []
    for f in families:
        for cell in idx[f].values():
            for vals in cell.values():
                if len(vals) > 1:
                    within.append(np.var(vals, ddof=1))
    within_mean = float(np.mean(within)) if within else float("nan")
    return between, within_mean


def analyse(results_dir: str, iters: int) -> dict:
    runs = load_runs(results_dir)
    by_model = split_by_model(runs)
    out: dict = {}
    for model in sorted(by_model):
        mruns = by_model[model]
        Mmap = load_mediator_M(model)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        m_out = {"n_families": None, "metrics": {}}
        for metric, label in METRICS:
            idx = index_runs(mruns, metric)
            fams = complete_families(idx)
            m_out["n_families"] = len(fams)
            pt = point_estimates(idx, fams)
            inter_dist = cluster_bootstrap(idx, fams, iters, rng)["interaction"]
            se_i = _se(inter_dist)
            mde_i = _mde(se_i)
            btw, win = variance_split(idx, fams)

            entry = {
                "label": label,
                "interaction": {
                    "est": pt["interaction"], "se": se_i, "mde": mde_i,
                    "n_half": _n_for_effect(len(fams), mde_i, mde_i / 2),
                    "detectable": abs(pt["interaction"]) >= mde_i,
                    "var_between": btw, "var_within": win,
                },
            }
            if Mmap:
                recs = build_records(mruns, Mmap, metric)
                nie_dist = mediation_bootstrap(recs, fams, iters, rng)["NIE"]
                se_n = _se(nie_dist)
                mde_n = _mde(se_n)
                # NIE point via full-sample fit
                from analyze_mediation import mediation_point
                nie_pt = mediation_point(recs)["NIE"]
                entry["nie"] = {
                    "est": nie_pt, "se": se_n, "mde": mde_n,
                    "n_half": _n_for_effect(len(fams), mde_n, mde_n / 2),
                    "detectable": abs(nie_pt) >= mde_n,
                }
            m_out["metrics"][metric] = entry
        out[model] = m_out
    return out


def render(res: dict) -> str:
    L: list = []
    P = L.append
    P("=" * 92)
    P("  STATISTICAL POWER — minimum detectable effect (MDE) at 80% power, "
      "two-sided alpha=0.05")
    P(f"  MDE = {Z:.3f} x SE(bootstrap).  Family-mean SE ~ 1/sqrt(n_families).")
    P("=" * 92)
    for model, mo in res.items():
        P(f"\n  MODEL: {model}   (n = {mo['n_families']} families)")
        P("  " + "-" * 88)
        P(f"    {'metric':<22}{'estimate':>11}{'SE':>10}{'MDE(80%)':>12}"
          f"{'|est|>=MDE?':>12}{'n for MDE/2':>13}")
        for metric, e in mo["metrics"].items():
            i = e["interaction"]
            P(f"    INTERACTION {metric:<10}{i['est']:>11.5f}{i['se']:>10.5f}"
              f"{i['mde']:>12.5f}{('yes' if i['detectable'] else 'NO'):>12}"
              f"{i['n_half']:>13.0f}")
            if "nie" in e:
                n = e["nie"]
                P(f"    NIE (indirect)      {n['est']:>11.5f}{n['se']:>10.5f}"
                  f"{n['mde']:>12.5f}{('yes' if n['detectable'] else 'NO'):>12}"
                  f"{n['n_half']:>13.0f}")
        # variance split (once per model, from first metric)
        fe = next(iter(mo["metrics"].values()))
        fi = fe["interaction"]
        btw, win = fi["var_between"], fi["var_within"]
        # This line used to assert "between dominates" unconditionally, which is
        # the assumption the whole 1/sqrt(n) extrapolation — and so the "n for
        # MDE/2" column — rests on. It is not always true: report the ratio and
        # say plainly when it fails, because when within dominates, adding
        # families buys less than the projection promises.
        ratio = btw / win if win and not np.isnan(win) and win > 0 else float("nan")
        verdict = ("between dominates, so 1/sqrt(n) scaling holds" if ratio >= 1
                   else "WITHIN dominates -> 1/sqrt(n_families) OVERSTATES the "
                        "gain from more families; treat 'n for MDE/2' as a "
                        "lower bound")
        P(f"    [variance split, {fe['label']}: between-family="
          f"{btw:.2e}  within-family≈{win:.2e}  (ratio {ratio:.2f}) "
          f"-> {verdict}]")
    # Stated conditionally on purpose. This block used to assert the all-nulls
    # reading as a constant, which is only a property of the n=4 results — at a
    # larger n it would have kept printing "every estimate is smaller than its
    # MDE" over a table showing otherwise.
    detectable = [(model, metric, kind)
                  for model, mo in res.items()
                  for metric, e in mo["metrics"].items()
                  for kind in ("interaction", "nie")
                  if kind in e and e[kind]["detectable"]]
    P("\n" + "=" * 92)
    if detectable:
        P("  READING: at least one estimate now EXCEEDS its own MDE —")
        for model, metric, kind in detectable:
            P(f"    {kind:<12} {metric:<12} on {model}")
        P("  those are powered detections at the observed size. Every other row is")
        P("  still an 'underpowered null', not a 'confirmed zero'.")
    else:
        P("  READING: every observed estimate is SMALLER than its own MDE — i.e. the")
        P("  design could not have detected effects of the observed size even if real.")
        P("  The nulls are therefore 'underpowered null', not 'confirmed zero'. The")
        P("  'n for MDE/2' column is the concrete cost of halving that blind spot.")
    P("=" * 92)
    return "\n".join(L)


def render_markdown(res: dict) -> str:
    M: list = []
    P = M.append
    n_fam = max((mo["n_families"] or 0) for mo in res.values()) if res else 0
    P(section_header(n_fam))
    P("")
    P(f"Minimum detectable effect (MDE) at 80% power, two-sided α=0.05: "
      f"`MDE = {Z:.2f} × SE`, SE from the family cluster bootstrap. This makes the "
      f"\"n={n_fam} is small\" caveat concrete (`power_analysis.py`, no new LLM calls).")
    P("")
    for model, mo in res.items():
        P(f"**{model}** (n = {mo['n_families']} families):")
        P("")
        P("| effect | estimate | SE | MDE (80%) | observed detectable? | n for ½·MDE |")
        P("|---|---|---|---|---|---|")
        for metric, e in mo["metrics"].items():
            i = e["interaction"]
            P(f"| interaction · {metric} | {i['est']:.5f} | {i['se']:.5f} "
              f"| {i['mde']:.5f} | {'yes' if i['detectable'] else '**no**'} "
              f"| {i['n_half']:.0f} |")
            if "nie" in e:
                n = e["nie"]
                P(f"| NIE · {metric} | {n['est']:.5f} | {n['se']:.5f} "
                  f"| {n['mde']:.5f} | {'yes' if n['detectable'] else '**no**'} "
                  f"| {n['n_half']:.0f} |")
        P("")
    n_half = max(
        (e["interaction"]["n_half"] for mo in res.values()
         for e in mo["metrics"].values()
         if not np.isnan(e["interaction"]["n_half"])),
        default=float("nan"),
    )
    detectable = [
        f"{model} · {metric}"
        for model, mo in res.items()
        for metric, e in mo["metrics"].items()
        if e["interaction"]["detectable"]
    ]
    if detectable:
        P(f"**Reading:** the interaction estimate now exceeds its own MDE for "
          f"{', '.join(f'`{d}`' for d in detectable)} — for those the design is "
          f"powered to detect an effect of the observed size. Everything else "
          f"remains an **underpowered null**: consistent with no effect, but not "
          f"proof of zero. Halving the detectable floor would need roughly "
          f"**4× the event families** (the `n for ½·MDE` column, up to "
          f"~{n_half:.0f}).")
    else:
        P(f"**Reading:** across the board the observed estimates are **smaller than "
          f"their own MDE**, so this design could not have detected effects of the "
          f"observed magnitude even if they were real. The interaction, mediation, and "
          f"interventional nulls are all **underpowered nulls** — consistent with no "
          f"effect, but not proof of zero. Detecting an effect half the current MDE "
          f"would need roughly **4× the event families** (the `n for ½·MDE` column); "
          f"e.g. moving from {n_fam} to ~{n_half:.0f} families.")
    P("")
    return "\n".join(M)


def append_md(md_path: str, section: str) -> str:
    block = section.rstrip() + "\n"
    content = open(md_path, encoding="utf-8").read() if os.path.exists(md_path) else ""
    m = SECTION_RE.search(content)
    if m:
        content = (content[:m.start()]
                   + re.sub(r"^.*?(?=\n## |\Z)", block.rstrip() + "\n",
                            content[m.start():], count=1, flags=re.S))
        action = "replaced"
    else:
        content = (content.rstrip() + "\n\n---\n\n" + block) if content else block
        action = "appended"
    open(md_path, "w", encoding="utf-8").write(content)
    return action


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Power / MDE analysis (n=4 families).")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--md", default="RESULTS_2x2.md")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args(argv)
    res = analyse(args.results, args.iters)
    print(render(res))
    if args.append:
        print(f"\n  RESULTS_2x2.md: {append_md(args.md, render_markdown(res))} "
              f"({SECTION_HEADER!r}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
