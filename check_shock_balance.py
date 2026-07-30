"""
check_shock_balance.py — confound check for the realism x framing shock design.

The realism factor is only interpretable if real and fictional shocks are
otherwise equivalent. If the fictional texts turn out to be (say) systematically
longer or harder to read, then any behavioural difference between conditions
could be a readability effect wearing a realism costume.

This script prints, side by side for every matched pair:

  • word / sentence / syllable counts
  • Flesch Reading Ease
  • the multiset of numeric values appearing in the text

and then aggregates real vs. fictional so drift is visible at a glance.

Flesch Reading Ease = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/word)
Higher = easier. Implemented here rather than pulled from `textstat` so the
check has no dependencies; the syllable counter is the standard vowel-group
heuristic and will be a syllable or two off on coined words — which is itself
worth watching, and is why raw syllable counts are printed too.

Run
---
  python3 check_shock_balance.py
  python3 check_shock_balance.py --shocks shocks.json --tolerance 5
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from typing import Optional

DEFAULT_SHOCKS = "shocks.json"

# Flesch deltas below this are treated as noise rather than confound.
DEFAULT_TOLERANCE = 5.0
# Word-count deltas above this fraction are flagged.
WORD_TOLERANCE_FRAC = 0.10

_VOWELS = "aeiouy"
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
# Numbers with optional sign, decimals, and trailing %. Handles en/em dashes
# in ranges by treating them as separators.
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

def count_syllables(word: str) -> int:
    """Vowel-group heuristic with a silent-final-e correction."""
    w = word.lower().strip("'’-")
    if not w:
        return 0
    groups = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            groups += 1
        prev_vowel = is_vowel
    if w.endswith("e") and not w.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(1, groups)


def words(text: str) -> list:
    return _WORD_RE.findall(text)


def sentence_count(text: str) -> int:
    n = len([s for s in _SENT_RE.split(text) if s.strip()])
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    ws = words(text)
    if not ws:
        return float("nan")
    n_words = len(ws)
    n_sents = sentence_count(text)
    n_syll = sum(count_syllables(w) for w in ws)
    return round(
        206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syll / n_words), 1
    )


def numeric_values(text: str) -> list:
    """Every numeric literal in the text, in order of appearance."""
    return _NUM_RE.findall(text)


def metrics(text: str) -> dict:
    ws = words(text)
    n_syll = sum(count_syllables(w) for w in ws)
    return {
        "words": len(ws),
        "sentences": sentence_count(text),
        "syllables": n_syll,
        "syll_per_word": round(n_syll / len(ws), 2) if ws else float("nan"),
        "flesch": flesch_reading_ease(text),
        "numbers": numeric_values(text),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_nums(nums: list) -> str:
    return ", ".join(nums) if nums else "—"


def report_pairs(shocks: list, framing: str) -> list:
    """Print a real-vs-fictional table for one framing. Returns pair deltas."""
    print("\n" + "=" * 92)
    print(f"  FRAMING: {framing.upper()}")
    print("=" * 92)

    by_pair: dict = {}
    for s in shocks:
        if s["framing"] != framing:
            continue
        by_pair.setdefault(s["matched_pair_id"], {})[s["realism"]] = s

    deltas = []
    for pair_id, members in sorted(by_pair.items()):
        real, fict = members.get("real"), members.get("fictional")
        if real is None or fict is None:
            print(f"\n  {pair_id}: INCOMPLETE PAIR — skipping")
            continue

        mr, mf = metrics(real["text"]), metrics(fict["text"])
        print(f"\n  {pair_id}")
        print(f"    {'':<12} {'words':>6} {'sents':>6} {'syl/wd':>7} {'flesch':>7}   numbers")
        print(f"    {'real':<12} {mr['words']:>6} {mr['sentences']:>6} "
              f"{mr['syll_per_word']:>7} {mr['flesch']:>7}   {_fmt_nums(mr['numbers'])}")
        print(f"    {'fictional':<12} {mf['words']:>6} {mf['sentences']:>6} "
              f"{mf['syll_per_word']:>7} {mf['flesch']:>7}   {_fmt_nums(mf['numbers'])}")

        d_words = mf["words"] - mr["words"]
        d_flesch = round(mf["flesch"] - mr["flesch"], 1)
        nums_match = Counter(mr["numbers"]) == Counter(mf["numbers"])
        print(f"    {'Δ (fict-real)':<12} {d_words:>+6} {'':>6} "
              f"{'':>7} {d_flesch:>+7}   numbers match: "
              f"{'YES' if nums_match else 'NO  <-- MAGNITUDES DIVERGE'}")

        deltas.append({
            "pair_id": pair_id, "framing": framing,
            "d_words": d_words, "d_flesch": d_flesch,
            "numbers_match": nums_match,
            "real_words": mr["words"], "fict_words": mf["words"],
            "real_flesch": mr["flesch"], "fict_flesch": mf["flesch"],
        })
    return deltas


def report_aggregate(deltas: list, tolerance: float) -> bool:
    """Print the aggregate verdict. Returns True if the design looks balanced."""
    print("\n" + "=" * 92)
    print("  AGGREGATE — is realism confounded with readability?")
    print("=" * 92)

    ok = True
    for framing in sorted({d["framing"] for d in deltas}):
        rows = [d for d in deltas if d["framing"] == framing]
        if not rows:
            continue
        rw = statistics.mean(d["real_words"] for d in rows)
        fw = statistics.mean(d["fict_words"] for d in rows)
        rf = statistics.mean(d["real_flesch"] for d in rows)
        ff = statistics.mean(d["fict_flesch"] for d in rows)
        max_abs_flesch = max(abs(d["d_flesch"]) for d in rows)
        word_frac = abs(fw - rw) / rw if rw else 0.0
        all_nums = all(d["numbers_match"] for d in rows)

        print(f"\n  {framing}:")
        print(f"    mean words   real {rw:6.1f}   fictional {fw:6.1f}   "
              f"Δ {fw - rw:+6.1f} ({word_frac * 100:.1f}%)")
        print(f"    mean flesch  real {rf:6.1f}   fictional {ff:6.1f}   "
              f"Δ {ff - rf:+6.1f}")
        print(f"    largest per-pair |Δ flesch|: {max_abs_flesch:.1f} "
              f"(tolerance {tolerance:.1f})")
        print(f"    numeric magnitudes match in every pair: "
              f"{'YES' if all_nums else 'NO'}")

        if not all_nums:
            ok = False
            print("    ** FAIL: magnitudes differ within a pair — fix before running.")
        if max_abs_flesch > tolerance:
            ok = False
            print(f"    ** FLAG: readability gap exceeds tolerance. Realism may be")
            print(f"       confounded with reading difficulty in this framing.")
        if word_frac > WORD_TOLERANCE_FRAC:
            ok = False
            print(f"    ** FLAG: mean length differs by >"
                  f"{WORD_TOLERANCE_FRAC * 100:.0f}%.")

    print("\n" + "-" * 92)
    if ok:
        print("  VERDICT: real and fictional shocks are matched within tolerance on")
        print("           length, readability, and numeric magnitude.")
    else:
        print("  VERDICT: at least one imbalance flagged above — see ** lines.")
    print("-" * 92)
    return ok


def report_caveats(catalogue: dict) -> None:
    caveat = catalogue.get("design", {}).get("caveat")
    if not caveat:
        return
    print("\n  DESIGN CAVEAT (from shocks.json)")
    for line in _wrap(caveat, 86):
        print(f"    {line}")
    print(
        "\n    Practical consequence: a realism x framing interaction may reflect\n"
        "    differing manipulation strength across framings rather than a true\n"
        "    moderating effect. Readability matching cannot fix this — it is a\n"
        "    property of what the numeric framing can carry."
    )


def _wrap(text: str, width: int) -> list:
    out, line = [], []
    for w in text.split():
        if len(" ".join(line + [w])) > width:
            out.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        out.append(" ".join(line))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Check real/fictional shock balance.")
    ap.add_argument("--shocks", default=DEFAULT_SHOCKS,
                    help=f"Catalogue path (default {DEFAULT_SHOCKS}).")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help=f"Max acceptable |Δ Flesch| per pair (default {DEFAULT_TOLERANCE}).")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any imbalance is flagged.")
    args = ap.parse_args(argv)

    try:
        with open(args.shocks, encoding="utf-8") as fh:
            catalogue = json.load(fh)
    except FileNotFoundError:
        print(f"No catalogue at {args.shocks!r}. Run: python3 generate_fictional_shocks.py")
        return 1

    shocks = catalogue["shocks"]
    print("=" * 92)
    print(f"  SHOCK BALANCE CHECK — {len(shocks)} shocks from {args.shocks}")
    print("=" * 92)
    for realism in ("real", "fictional"):
        for framing in ("narrative", "numeric"):
            c = sum(1 for s in shocks
                    if s["realism"] == realism and s["framing"] == framing)
            print(f"  {realism:<10} x {framing:<10} : {c}")

    deltas = []
    for framing in ("narrative", "numeric"):
        deltas.extend(report_pairs(shocks, framing))

    print("\n  Note: Flesch on the numeric framing is reported for completeness but")
    print("  is not meaningful — those strings are templated key:value vectors, not")
    print("  prose. The numeric cells are identical by construction except for the")
    print("  entity token, so structural matching there is exact.")

    ok = report_aggregate(deltas, args.tolerance)
    report_caveats(catalogue)

    return 0 if (ok or not args.strict) else 2


if __name__ == "__main__":
    sys.exit(main())
