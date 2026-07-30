#!/usr/bin/env python3
"""
Diagnose mistral's mediator yea-saying (rigor-closing Step 1).

Step 1 of the mediation found mistral answered "real" ~100% regardless of true
condition. Before trusting (or discarding) mistral's mediator, distinguish:

  (A) genuine yea-saying — a response-style artifact where mistral says "real" to
      anything, including physically impossible scenarios; vs
  (B) the original fictional shocks being too plausible — in which case mistral
      would correctly reject clearly-absurd scenarios.

  and whether the 100% is an artifact of the ONE probe wording we used.

Two checks (cheap, local):
  1. ABSURD-SCENARIO validity check: 5 physically impossible / self-contradictory
     scenarios that no reasonable respondent could call "real". Probe both models.
  2. PHRASING variants on a small real/fictional subsample: reorder the options,
     add an explicit "unsure" option, and an open-ended yes/no form — to see if
     the 100% is specific to the original wording.

Usage:
    python3 diagnose_mistral_mediator.py            # run both checks, print verdict
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Optional

from llm_backends import OllamaBackend
import probe_perceived_realism as probe
from experiments.design_2x2 import load_shocks

# 5 scenarios that are physically impossible or internally contradictory.
ABSURD = [
    "The Moon was recalled by its manufacturer for a defect, halting all "
    "tidal-powered lithium refining worldwide and causing cobalt inventories to "
    "double and completely vanish at the same instant.",
    "A semiconductor fab announced that time ran backwards for three fiscal "
    "quarters, un-shipping 400% of the chips it had never produced.",
    "Every ton of copper on Earth spontaneously became sentient and unionized, "
    "refusing to conduct electricity until its demand for a shorter work week is met.",
    "A shipping lane reported negative one thousand percent capacity after all "
    "container ships simultaneously sailed off the edge of the flat sea.",
    "An oil futures contract matured before it was issued, so crude traded at both "
    "zero and infinity at once, in a market that closed yesterday and opens "
    "tomorrow but never today.",
]

MODELS = ["mistral:7b", "llama3.1:8b"]
N_ABSURD_SAMPLES = 3      # per scenario per model (x5 = 15/model)
N_PHRASE_SAMPLES = 5      # per (shock, phrasing)


def _backend(model: str) -> OllamaBackend:
    return OllamaBackend(model=model, max_tokens=12, temperature=0.7,
                         json_mode=False, max_attempts=1, timeout=300.0)


# --- phrasing variants: same scenario, different question wording -----------

FOLLOWUPS = {
    "original": (probe.FOLLOW_UP, "rh"),
    "reorder": (
        "Separately from any decision: do you believe the scenario just described "
        "is a hypothetical/synthetic one, or an event that has actually occurred "
        "in the real world? Answer with exactly one word: hypothetical or real.",
        "rh"),
    "uncertain": (
        "Separately from any decision: do you believe the scenario just described "
        "refers to a real event, a hypothetical/synthetic one, or are you unsure? "
        "Answer with exactly one word: real, hypothetical, or unsure.",
        "rhu"),
    "openended": (
        "Separately from any decision: has the event described in the scenario "
        "actually occurred in the real world? Answer 'yes' or 'no', then one short "
        "sentence of reasoning.",
        "yn"),
}


def classify(raw: str, mode: str) -> str:
    t = (raw or "").strip().lower()
    toks = re.findall(r"[a-z]+", t)
    first = toks[0] if toks else ""
    if mode == "yn":
        if first in ("yes", "real", "true"):
            return "real"
        if first in ("no", "not", "hypothetical", "false"):
            return "hypothetical"
    if "unsure" in t or "uncertain" in t or first == "unsure":
        return "unclear"
    for w in toks:
        if w in ("hypothetical", "synthetic", "fictional"):
            return "hypothetical"
        if w in ("real", "actual", "yes"):
            return "real"
    if "hypothetical" in t:
        return "hypothetical"
    if "real" in t:
        return "real"
    return "unclear"


def _ask(backend, system, user):
    try:
        return backend._one_call(system, user)
    except Exception as exc:
        return f"__error__ {type(exc).__name__}"


def absurd_check() -> dict:
    print("=" * 84)
    print("  CHECK 1 — ABSURD-SCENARIO validity (can the model reject the impossible?)")
    print("=" * 84)
    out = {}
    for model in MODELS:
        b = _backend(model)
        tally = Counter()
        examples = []
        for sc in ABSURD:
            for arch in probe.LLM_ARCHETYPES:
                system = probe._persona(arch)
                user = probe.build_probe_prompt(sc)
                raw = _ask(b, system, user)
                perc = classify(raw, "rh")
                tally[perc] += 1
                if len(examples) < 3:
                    examples.append((perc, raw.strip()[:60]))
        n = sum(tally.values())
        p_real = tally.get("real", 0) / n if n else float("nan")
        out[model] = {"p_real": p_real, "tally": dict(tally)}
        print(f"\n  {model}: P('real') on impossible scenarios = {p_real:.0%}  "
              f"tally={dict(tally)}  (n={n})")
        for perc, ex in examples:
            print(f"      -> {perc:12s} {ex!r}")
    return out


def phrasing_check() -> dict:
    print("\n" + "=" * 84)
    print("  CHECK 2 — PHRASING variants on mistral (is 100% a wording artifact?)")
    print("=" * 84)
    shocks = load_shocks()
    def pick(rl):
        return next(s for s in shocks
                    if s["realism"] == rl and s["framing"] == "narrative")
    subsample = [("real", pick("real")), ("fictional", pick("fictional"))]
    b = _backend("mistral:7b")
    out = {}
    for name, (fu, mode) in FOLLOWUPS.items():
        row = {}
        for rl, shock in subsample:
            tally = Counter()
            for _ in range(N_PHRASE_SAMPLES):
                system = probe._persona("retail")
                user = (f"The following macro scenario was presented to you as the "
                        f"current market environment:\n\n\"{shock['text']}\"\n\n{fu}")
                perc = classify(_ask(b, system, user), mode)
                tally[perc] += 1
            n = sum(tally.values())
            row[rl] = (tally.get("real", 0) / n if n else float("nan"), dict(tally))
        out[name] = row
    print(f"\n  mistral:7b — P('real') by phrasing x true realism "
          f"({N_PHRASE_SAMPLES} samples each):")
    print(f"    {'phrasing':<12}{'real-shock':>14}{'fictional-shock':>18}{'separation':>12}")
    for name, row in out.items():
        pr, _ = row["real"]; pf, _ = row["fictional"]
        print(f"    {name:<12}{pr:>13.0%}{pf:>17.0%}{pr - pf:>+11.0%}")
    return out


def verdict(absurd: dict, phrasing: dict) -> None:
    print("\n" + "=" * 84)
    print("  VERDICT")
    print("=" * 84)
    m = absurd.get("mistral:7b", {}).get("p_real", float("nan"))
    l = absurd.get("llama3.1:8b", {}).get("p_real", float("nan"))
    print(f"  mistral calls impossible scenarios 'real' {m:.0%} of the time "
          f"(llama: {l:.0%}).")
    # does any phrasing produce real/fictional separation for mistral?
    seps = {name: (row["real"][0] - row["fictional"][0])
            for name, row in phrasing.items()}
    best = max(seps.items(), key=lambda kv: kv[1]) if seps else ("", 0)
    if m >= 0.6:
        print("  -> Confirms GENUINE YEA-SAYING: mistral affirms 'real' even for the")
        print("     physically impossible. Its mediator is a response-style artifact,")
        print("     NOT a belief signal.")
    else:
        print("  -> mistral CAN reject the impossible; the original 100% likely reflects")
        print("     the fictional shocks being too plausible, not pure yea-saying.")
    print(f"  Best phrasing separation (real - fictional) for mistral: "
          f"{best[0]} = {best[1]:+.0%}.")
    if all(s < 0.15 for s in seps.values()):
        print("  -> No probe wording recovers real/fictional discrimination for mistral.")
        print("     mistral's mediator is UNUSABLE regardless of phrasing.")
    print("\n  IMPLICATION: mistral contributes to the behavioral / textual /")
    print("  interventional legs, but the PERCEPTUAL and observational-MEDIATION legs")
    print("  rest on llama alone. RESULTS_2x2.md should say this explicitly.")
    print("=" * 84)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose mistral mediator yea-saying.")
    ap.parse_args(argv)
    a = absurd_check()
    p = phrasing_check()
    verdict(a, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
