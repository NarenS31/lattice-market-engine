#!/usr/bin/env python3
"""
Reasoning-text leakage analysis for the 2x2 framing x realism sweep.

Read-only. Scans every LLM agent's `reasoning` field in results_2x2/*.json for
signs of REAL-WORLD RECALL that go beyond what the shock prompt actually
provided — i.e. textual evidence that a model is drawing on memorized knowledge
of a real event rather than reasoning from the stimulus it was handed.

This is the companion to the behavioral 2x2 result (RESULTS_2x2.md): the
behavioral interaction is a null that replicates across both models. The
question here is whether the *text* tells a different story — do agents show
real-world association in narrative+real that the trading metrics don't capture?

Method
------
1. Per REAL event, a leakage lexicon of SPECIFIC real-world markers a model
   could only use from outside knowledge: real firms, real places, named
   historical crises / years, and explicit "this is like the real X" phrasings.
   FICTIONAL events have an EMPTY lexicon by construction — there is no real
   referent to recall, so nothing there can count as leakage.

2. For each decision record, a lexicon term counts as leakage ONLY if it
   appears in the reasoning AND does NOT appear in the shock prompt that record
   was actually given (`shock_text`). The same term is therefore in-prompt (not
   leakage) in the narrative cell and out-of-prompt (leakage) in the numeric
   cell, where the model saw only a bare entity token.

   Two tiers are scored separately:
     Tier 1 (primary) — specific memorized-knowledge markers (TSMC, Chile, SVB,
       2008, "in reality", ...). This is the genuine recall signal.
     Tier 2 (sensitivity) — generic inferable synonyms of an in-prompt entity
       (crude_futures -> "oil", chip -> "semiconductor"). These are derivable
       from the prompt by general knowledge, NOT proof of event recall, and are
       reported apart so they cannot masquerade as Tier-1 leakage.

3. Leakage rate per cell (framing x realism x model), then the test the design
   cares about: is narrative+real elevated relative to the other three cells?

Excluded from scoring: the market-maker (templated, never calls an LLM), and
coin-flip / backend-failure records (their reasoning is a fixed fallback string,
not a decision). Counts of exclusions are reported.

Usage:
    python3 analyze_reasoning_leakage.py            # analyse + append to RESULTS_2x2.md
    python3 analyze_reasoning_leakage.py --no-append # print only
    python3 analyze_reasoning_leakage.py --results results_2x2 --md RESULTS_2x2.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Optional

LLM_ARCHETYPES = ("retail", "institutional", "quant")  # market_maker excluded
FRAMINGS = ("narrative", "numeric")
REALISMS = ("real", "fictional")
CELLS = [(fr, rl) for rl in REALISMS for fr in FRAMINGS]

SECTION_HEADER = "## Reasoning-Text Leakage Analysis (Step 1)"

# ---------------------------------------------------------------------------
# Leakage lexicon — Tier 1: specific real-world recall markers, per REAL event.
#
# These are things a model can only say by drawing on memorized real-world
# knowledge. Terms that happen to appear in a given record's prompt are removed
# at scoring time, so e.g. "tsmc"/"taiwan" (present in the narrative prompt)
# only count as leakage in the numeric cell, where the model saw "3nm_wafer".
# ---------------------------------------------------------------------------
LEXICON_T1: dict[str, list[str]] = {
    "lithium_shortage": [
        # real places / producers of the real lithium supply chain
        "chile", "argentina", "bolivia", "lithium triangle", "atacama",
        "albemarle", "sqm", "ganfeng", "tianqi",
        # real downstream names / associated commodities not in the prompt
        "tesla", "panasonic", "byd", "cobalt", "nickel", "gigafactory",
    ],
    "semiconductor_shortage": [
        "tsmc", "taiwan", "samsung", "intel", "nvidia", "amd", "qualcomm",
        "apple", "asml", "globalfoundries", "arizona", "south korea",
        "netherlands", "chips act",
        # the real 2020-21 chip shortage context
        "pandemic", "covid",
    ],
    "energy_shock": [
        "hormuz", "opec", "saudi", "iran", "russia", "ukraine", "venezuela",
        "persian gulf", "wti", "brent", "barrel", "embargo",
    ],
    "credit_crunch": [
        "svb", "silicon valley bank", "lehman", "bear stearns", "credit suisse",
        "first republic", "signature bank", "washington mutual", "subprime",
        "global financial crisis", "great recession", "mortgage crisis",
    ],
}

# Cross-cutting explicit real-world-comparison phrasings. Applied to REAL events
# only (fictional lexicon is empty by construction). Deliberately strong: bare
# "historically"/"in the past" are excluded because they fire on strategy talk
# ("historically aligns with our momentum signal") rather than event recall.
RECALL_PHRASES: list[str] = [
    "in reality", "real world", "real-world", "actually happened",
    "reminds me of", "similar to when", "as happened in", "the actual event",
    "back in 20", "during the 20", "the 2008", "the 2020", "the 2021",
    "the 2022", "the 2023", "in 2008", "1973", "financial crisis",
    "dot-com", "dotcom",
]

# Named year tokens on their own are a strong recall marker in this domain.
YEAR_RE = re.compile(r"(?<![0-9])(19[7-9]\d|20[0-2]\d)(?![0-9])")

# ---------------------------------------------------------------------------
# Tier 2: generic inferable synonyms of an IN-PROMPT entity, per REAL event.
# Reported separately — NOT counted as Tier-1 leakage.
# ---------------------------------------------------------------------------
LEXICON_T2: dict[str, list[str]] = {
    "energy_shock": ["oil", "petroleum", "petrol"],
    "semiconductor_shortage": ["semiconductor", "semiconductors", "microchip"],
    "lithium_shortage": ["electric vehicle", "electric vehicles"],
    "credit_crunch": [],
}


def _norm(s: str) -> str:
    """Lowercase, underscores -> spaces, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"_+", " ", (s or "").lower())).strip()


def _contains(text: str, term: str) -> bool:
    """Whole-token/phrase containment on already-normalised text.

    Alphanumeric boundaries so 'ev' does not match 'seven' and 'svb' does not
    match a substring of a longer token.
    """
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def score_record(reasoning: str, prompt: str, event_key: str, realism: str) -> dict:
    """Return {'t1': [terms], 't2': [terms]} for one decision record.

    A term counts only if present in the reasoning and absent from the prompt.
    Fictional records have an empty Tier-1 lexicon by construction.
    """
    r = _norm(reasoning)
    p = _norm(prompt)

    t1_terms: list[str] = []
    t2_terms: list[str] = []
    if realism == "real":
        candidates = LEXICON_T1.get(event_key, []) + RECALL_PHRASES
        t1_terms = [t for t in candidates if _contains(r, t) and not _contains(p, t)]
        # standalone real-world year references not present in the prompt
        for y in set(YEAR_RE.findall(r)):
            if y not in p:
                t1_terms.append(f"year:{y}")
        t2_terms = [t for t in LEXICON_T2.get(event_key, [])
                    if _contains(r, t) and not _contains(p, t)]
    # realism == 'fictional': lexicon empty by construction -> no leakage.
    return {"t1": t1_terms, "t2": t2_terms}


def _iter_records(results_dir: str):
    """Yield (model, framing, realism, event_key, prompt, reasoning) per LLM
    decision record, skipping the market maker and non-substantive fallbacks.
    Also yields a status so the caller can tally exclusions."""
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "*.json"))):
        if os.path.basename(path) == "manifest.json":
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        model = d.get("model", "?")
        framing = d.get("framing", "?")
        realism = d.get("realism", "?")
        event_key = d.get("event_key", "?")
        prompt = d.get("shock_text", "")
        for tick in d.get("decisions", []):
            for a in tick:
                if a.get("archetype") == "market_maker":
                    continue
                if a.get("archetype") not in LLM_ARCHETYPES:
                    continue
                reasoning = (a.get("reasoning") or "").strip()
                if a.get("forced_random") or a.get("api_failed"):
                    yield (model, framing, realism, event_key, prompt, reasoning,
                           "excluded_fallback")
                    continue
                if not reasoning:
                    yield (model, framing, realism, event_key, prompt, reasoning,
                           "excluded_empty")
                    continue
                yield (model, framing, realism, event_key, prompt, reasoning, "scored")


def analyse(results_dir: str) -> dict:
    models: set = set()
    # per (model, framing, realism): counts
    scored = defaultdict(int)
    excluded = defaultdict(int)
    t1_instances = defaultdict(int)
    t2_instances = defaultdict(int)
    t1_examples: list = []
    t2_examples: list = []
    t1_term_counter = defaultdict(int)

    for (model, fr, rl, ek, prompt, reasoning, status) in _iter_records(results_dir):
        models.add(model)
        key = (model, fr, rl)
        if status.startswith("excluded"):
            excluded[key] += 1
            continue
        scored[key] += 1
        res = score_record(reasoning, prompt, ek, rl)
        if res["t1"]:
            t1_instances[key] += 1
            for t in res["t1"]:
                t1_term_counter[t] += 1
            if len(t1_examples) < 25:
                t1_examples.append((model, fr, rl, ek, res["t1"], reasoning[:150]))
        if res["t2"]:
            t2_instances[key] += 1
            if len(t2_examples) < 25:
                t2_examples.append((model, fr, rl, ek, res["t2"], reasoning[:150]))

    return {
        "models": sorted(models),
        "scored": dict(scored),
        "excluded": dict(excluded),
        "t1_instances": dict(t1_instances),
        "t2_instances": dict(t2_instances),
        "t1_examples": t1_examples,
        "t2_examples": t2_examples,
        "t1_term_counter": dict(t1_term_counter),
    }


def _rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def _cell_rates(res: dict, model: Optional[str], tier: str) -> dict:
    """{(framing, realism): (instances, scored, rate)} for one model or combined."""
    inst = res["t1_instances"] if tier == "t1" else res["t2_instances"]
    out = {}
    for (fr, rl) in CELLS:
        if model is None:
            i = sum(v for (m, f, r), v in inst.items() if (f, r) == (fr, rl))
            n = sum(v for (m, f, r), v in res["scored"].items() if (f, r) == (fr, rl))
        else:
            i = inst.get((model, fr, rl), 0)
            n = res["scored"].get((model, fr, rl), 0)
        out[(fr, rl)] = (i, n, _rate(i, n))
    return out


def _elevation_verdict(rates: dict) -> tuple:
    """Is narrative+real elevated vs the other three cells? Returns (bool, text)."""
    nr_i, nr_n, nr_rate = rates[("narrative", "real")]
    others = [rates[c] for c in CELLS if c != ("narrative", "real")]
    other_rate_mean = sum(r for _, _, r in others) / len(others)
    other_max = max((r for _, _, r in others), default=0.0)
    # Elevated requires a non-trivial absolute rate AND clear separation.
    elevated = (nr_rate >= 0.02) and (nr_i >= 5) and (nr_rate >= 2 * max(other_max, 1e-9))
    if elevated:
        txt = (f"narrative+real ({nr_rate:.2%}) is ELEVATED vs the other three "
               f"cells (max {other_max:.2%}).")
    else:
        txt = (f"narrative+real ({nr_rate:.2%}) is NOT elevated vs the other "
               f"three cells (mean {other_rate_mean:.2%}, max {other_max:.2%}) "
               f"— the pattern is flat.")
    return elevated, txt


def _fmt_cell_table(res: dict, model: Optional[str], tier: str) -> list:
    rates = _cell_rates(res, model, tier)
    lines = []
    for rl in REALISMS:
        for fr in FRAMINGS:
            i, n, rt = rates[(fr, rl)]
            lines.append(f"    {fr:>9} x {rl:<9}  {i:>4} / {n:<5} = {rt:>7.3%}")
    return lines


def render_report(res: dict) -> str:
    L: list = []
    P = L.append
    models = res["models"]
    total_scored = sum(res["scored"].values())
    total_excl = sum(res["excluded"].values())
    total_t1 = sum(res["t1_instances"].values())
    total_t2 = sum(res["t2_instances"].values())

    P("=" * 84)
    P("  REASONING-TEXT LEAKAGE ANALYSIS — 2x2 framing x realism sweep")
    P("=" * 84)
    P(f"  models              : {', '.join(models)}")
    P(f"  LLM decision records: {total_scored:,} scored  "
      f"({total_excl:,} excluded: coin-flip/backend-fail/empty)")
    P(f"  Tier-1 instances    : {total_t1}  (specific real-world recall markers)")
    P(f"  Tier-2 instances    : {total_t2}  (generic inferable synonyms — sensitivity only)")
    P("")

    # ---- Tier 1 per model + combined ----
    P("-" * 84)
    P("  TIER 1 — specific real-world recall (the primary leakage signal)")
    P("  leakage instances / scored records = rate, per cell")
    P("-" * 84)
    for model in models:
        P(f"  [{model}]")
        for line in _fmt_cell_table(res, model, "t1"):
            P(line)
        elevated, txt = _elevation_verdict(_cell_rates(res, model, "t1"))
        P(f"    -> {txt}")
        P("")
    P("  [combined — both models pooled]")
    for line in _fmt_cell_table(res, None, "t1"):
        P(line)
    combined_elevated, combined_txt = _elevation_verdict(_cell_rates(res, None, "t1"))
    P(f"    -> {combined_txt}")
    P("")

    if res["t1_term_counter"]:
        P("  Tier-1 terms that fired (term: count):")
        for t, c in sorted(res["t1_term_counter"].items(), key=lambda x: -x[1]):
            P(f"    {t}: {c}")
        P("")
        P("  Tier-1 examples:")
        for (m, fr, rl, ek, terms, txt) in res["t1_examples"][:12]:
            P(f"    [{m} {fr}/{rl} {ek}] {terms} :: {txt}")
        P("")
    else:
        P("  Tier-1: NO specific real-world recall markers fired in any record,")
        P("          any cell, either model.")
        P("")

    # ---- Tier 2 sensitivity ----
    P("-" * 84)
    P("  TIER 2 — generic inferable synonyms (sensitivity; NOT event recall)")
    P("-" * 84)
    P("  [combined — both models pooled]")
    for line in _fmt_cell_table(res, None, "t2"):
        P(line)
    P("")
    if res["t2_examples"]:
        P("  Tier-2 examples (synonyms derivable from the prompt entity):")
        for (m, fr, rl, ek, terms, txt) in res["t2_examples"][:8]:
            P(f"    [{m} {fr}/{rl} {ek}] {terms} :: {txt}")
        P("")

    # ---- Relation to the behavioral null ----
    P("=" * 84)
    P("  READING — leakage vs the established behavioral null")
    P("=" * 84)
    nr_i, nr_n, nr_rate = _cell_rates(res, None, "t1")[("narrative", "real")]
    if combined_elevated:
        P("  Behavioral interaction: NULL (replicates across both models).")
        P("  Textual leakage:        ELEVATED in narrative+real.")
        P("")
        P("  => A real, reportable dissociation: agents show textual signs of")
        P("     drawing on real-world association in the narrative+real cell")
        P("     WITHOUT it moving the trading metrics. Recall surfaces in the")
        P("     words but not in the behavior.")
    else:
        P("  Behavioral interaction: NULL (replicates across both models).")
        P("  Textual leakage:        FLAT / absent across all four cells.")
        P("")
        P("  => Independent confirmation. The behavioral null is not a")
        P("     measurement artifact: there is no hidden real-world-recall")
        P("     signal in the reasoning text that the trading metrics failed")
        P("     to capture. Real and fictional shocks are processed the same")
        P("     way in the text as in the behavior. The one caveat that")
        P("     remains is manipulation strength, not leakage (see numeric-")
        P("     framing limitation in this document).")
    P("")
    P("  Scope: this detects lexical recall (named entities, places, dated")
    P("  events, explicit real-world comparisons). It cannot detect recall")
    P("  that is purely latent — a model influenced by a real event without")
    P("  naming anything. Absence of lexical leakage is evidence, not proof,")
    P("  of no recall.")
    P("=" * 84)
    return "\n".join(L)


def render_markdown(res: dict) -> str:
    models = res["models"]
    total_scored = sum(res["scored"].values())
    total_t1 = sum(res["t1_instances"].values())
    total_t2 = sum(res["t2_instances"].values())
    comb = _cell_rates(res, None, "t1")
    comb_elevated, _ = _elevation_verdict(comb)

    M: list = []
    P = M.append
    P(SECTION_HEADER)
    P("")
    P("Read-only textual companion to the behavioral result above "
      "(`analyze_reasoning_leakage.py`, no new runs). It scans every LLM "
      "agent's `reasoning` field for **real-world recall beyond the shock "
      "prompt** — named firms/places, dated crises, or explicit \"like the "
      "real X\" comparisons. Fictional shocks have an empty lexicon by "
      "construction (no real referent to recall). A term counts only if it is "
      "in the reasoning and NOT in the prompt that record was given, so the "
      "same term is in-prompt (narrative) yet out-of-prompt (numeric).")
    P("")
    P(f"- **Records scored:** {total_scored:,} LLM decisions across "
      f"{len(models)} models (market-maker, coin-flip, and backend-failure "
      f"records excluded).")
    P(f"- **Tier 1 — specific recall markers:** {total_t1} instances.")
    P(f"- **Tier 2 — generic inferable synonyms** (crude→\"oil\", "
      f"chip→\"semiconductor\"; reported apart, not counted as recall): "
      f"{total_t2} instances.")
    P("")
    P("**Tier-1 leakage rate per cell (combined, both models):**")
    P("")
    P("| framing × realism | instances / records | rate |")
    P("|---|---|---|")
    for rl in REALISMS:
        for fr in FRAMINGS:
            i, n, rt = comb[(fr, rl)]
            P(f"| {fr} × {rl} | {i} / {n} | {rt:.3%} |")
    P("")
    if comb_elevated:
        P("**Pattern:** narrative+real is **elevated** relative to the other "
          "three cells.")
        P("")
        P("**Relation to the behavioral null:** the trading-metric interaction "
          "is null and replicates, yet the *text* shows elevated real-world "
          "association in narrative+real. This is a genuine dissociation — "
          "recall appears in the words without changing behavior — and is "
          "reportable on its own.")
    else:
        P("**Pattern:** narrative+real is **not** elevated — leakage is flat "
          "(effectively zero) across all four cells, both models.")
        P("")
        P("**Relation to the behavioral null:** this is **independent "
          "confirmation** that the replicated behavioral null is real, not a "
          "measurement artifact. There is no hidden lexical real-world-recall "
          "signal that the trading metrics missed; real and fictional shocks "
          "are processed alike in text as in behavior. The only near-misses "
          "are Tier-2 synonyms (\"oil\" for `crude_futures`, \"semiconductor\" "
          "for `chip`) that are trivially inferable from the prompt and are "
          "not evidence of event recall.")
    P("")
    P("**Scope/limitation:** this detects *lexical* recall only. A model "
      "influenced by a real event without naming any specific entity would "
      "not be flagged, so absence of lexical leakage is evidence — not proof "
      "— of no recall.")
    P("")
    return "\n".join(M)


def append_to_markdown(md_path: str, section: str) -> str:
    """Replace an existing leakage section or append a new one. Returns action."""
    new_block = section.rstrip() + "\n"
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as fh:
            content = fh.read()
    else:
        content = ""
    if SECTION_HEADER in content:
        # Replace from the section header up to the next top-level "## " or EOF.
        pat = re.compile(
            re.escape(SECTION_HEADER) + r".*?(?=\n## (?!Reasoning-Text)|\Z)",
            re.S,
        )
        content = pat.sub(new_block.rstrip() + "\n", content)
        action = "replaced existing section"
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content = content.rstrip() + "\n\n---\n\n" + new_block
        action = "appended new section"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return action


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Reasoning-text leakage analysis.")
    ap.add_argument("--results", default="results_2x2", help="results dir")
    ap.add_argument("--md", default="RESULTS_2x2.md", help="markdown to append to")
    ap.add_argument("--no-append", action="store_true",
                    help="print only; do not touch the markdown file")
    args = ap.parse_args(argv)

    res = analyse(args.results)
    print(render_report(res))

    if not args.no_append:
        action = append_to_markdown(args.md, render_markdown(res))
        print(f"\n  RESULTS_2x2.md: {action} ({SECTION_HEADER!r}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
