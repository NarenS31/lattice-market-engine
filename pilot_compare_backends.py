"""
Pilot: Anthropic vs. local Ollama decision quality.

Runs the same N decision prompts through both backends and reports:

  • JSON parse success rate  (first attempt, and final after retries)
  • response time            (mean / median / p95)
  • the decisions themselves, side by side, for manual inspection

The prompts are built with the real `Agent._build_user_prompt`, cycling
through the Retail / Institutional / Quant archetypes over a deterministic
synthetic tick sequence — so what the models see here is the same shape they
see in a live run. No order book is touched.

Run
---
  python3 pilot_compare_backends.py
  python3 pilot_compare_backends.py --n 20 --ollama-model llama3.1:8b
  python3 pilot_compare_backends.py --only ollama --json-mode

Exit code is 0 even if a backend is unavailable — the report says so.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from typing import Optional

import llm_backends
from llm_backends import AnthropicBackend, OllamaBackend
from agents import RetailAgent, InstitutionalAgent, QuantAgent

DEFAULT_N = 20
DEFAULT_LOG = "pilot_parse_failures.jsonl"


# ---------------------------------------------------------------------------
# Synthetic prompt construction
# ---------------------------------------------------------------------------

_MACROS = [
    {},
    {"fed_rate": 5.75, "vix": 31.0, "sentiment": -0.2},
    {"fed_rate": 5.25, "vix": 44.0, "sentiment": -0.7},
    {
        "multi_sentence_narrative": (
            "A major lithium supplier has halted exports following a regulatory "
            "review, tightening battery-cell supply across the sector. "
            "Downstream manufacturers are signalling production cuts next "
            "quarter, and analysts expect margin compression to spread to "
            "adjacent industrials."
        )
    },
]


def _make_snapshot(rng: random.Random, tick: int, price: float) -> dict:
    """A market snapshot in the shape `Agent._build_user_prompt` expects."""
    spread = round(rng.uniform(0.04, 0.35), 2)
    best_bid = round(price - spread / 2, 2)
    best_ask = round(price + spread / 2, 2)
    history = []
    p = price
    for _ in range(6):
        p = round(p + rng.uniform(-0.45, 0.45), 2)
        history.append({"clearing_price": p, "volume": round(rng.uniform(2, 60), 1)})
    return {
        "mid_price": round((best_bid + best_ask) / 2, 2),
        "last_clearing_price": history[-1]["clearing_price"],
        "order_book": {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bids": [{"price": best_bid, "qty": 10.0}],
            "asks": [{"price": best_ask, "qty": 10.0}],
        },
        "tick_history": history,
    }


def build_prompts(n: int, seed: int = 7) -> list[dict]:
    """Return n {archetype, system, user} prompt records, deterministic in seed."""
    rng = random.Random(seed)
    agents = [
        ("retail", RetailAgent(cash=10_000.0, name="Retail")),
        ("institutional", InstitutionalAgent(cash=100_000.0, name="Inst")),
        ("quant", QuantAgent(cash=50_000.0, name="Quant")),
    ]
    records = []
    price = 100.0
    for i in range(n):
        arch, agent = agents[i % len(agents)]
        price = round(price + rng.uniform(-0.8, 0.8), 2)
        agent.position = round(rng.uniform(-20, 40), 1)
        agent._wealth_history.clear()
        for _ in range(rng.randint(0, 5)):
            agent._wealth_history.append(agent.cash + rng.uniform(-2000, 2000))
        snap = _make_snapshot(rng, i, price)
        macro = _MACROS[i % len(_MACROS)]
        records.append(
            {
                "idx": i,
                "archetype": arch,
                "system": agent._SYSTEM_PROMPT,
                "user": agent._build_user_prompt(snap, macro, i),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Running one backend over the prompt set
# ---------------------------------------------------------------------------

def _attempts_used(before: dict, after: dict, name: str) -> int:
    b = before.get(name, {}).get("attempts", 0)
    a = after.get(name, {}).get("attempts", 0)
    return a - b


def run_backend(backend, prompts: list[dict], label: str) -> dict:
    """Run every prompt through `backend`, timing and scoring each call."""
    print(f"\n  {label:<26} ", end="", flush=True)
    rows = []
    for rec in prompts:
        before = llm_backends.stats()
        t0 = time.perf_counter()
        decision = backend.get_decision(rec["system"], rec["user"])
        elapsed = time.perf_counter() - t0
        after = llm_backends.stats()

        used = _attempts_used(before, after, backend.name)
        rows.append(
            {
                "idx": rec["idx"],
                "archetype": rec["archetype"],
                "decision": decision,
                "seconds": elapsed,
                "attempts": max(1, used),
                "first_try": decision is not None and used <= 1,
            }
        )
        print("." if decision is not None else "x", end="", flush=True)
    print()

    times = [r["seconds"] for r in rows]
    ok = [r for r in rows if r["decision"] is not None]
    return {
        "label": label,
        "backend": backend.name,
        "model": backend.model,
        "rows": rows,
        "n": len(rows),
        "final_ok": len(ok),
        "first_try_ok": sum(1 for r in rows if r["first_try"]),
        "total_attempts": sum(r["attempts"] for r in rows),
        "mean_s": statistics.mean(times) if times else float("nan"),
        "median_s": statistics.median(times) if times else float("nan"),
        "p95_s": (
            sorted(times)[max(0, int(round(0.95 * len(times))) - 1)]
            if times else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:5.1f}%" if den else "  n/a"


def _fmt_decision(d: Optional[dict]) -> str:
    if d is None:
        return "PARSE FAILED / unavailable"
    action = str(d.get("action", "?")).lower()
    qty = d.get("quantity")
    lp = d.get("limit_price")
    qty_s = f"{float(qty):g}" if isinstance(qty, (int, float)) else str(qty)
    lp_s = f"{float(lp):.2f}" if isinstance(lp, (int, float)) else "—"
    return f"{action:<4} qty={qty_s:<6} @ {lp_s:<8}"


def _reasoning(d: Optional[dict], width: int = 96) -> str:
    if d is None:
        return ""
    r = str(d.get("reasoning", "")).strip().replace("\n", " ")
    return r[:width] + ("…" if len(r) > width else "")


def print_report(results: list[dict], log_path: str) -> None:
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(
        f"  {'Backend':<12} {'Model':<28} {'1st-try':>8} {'final':>8} "
        f"{'mean':>8} {'med':>8} {'p95':>8}"
    )
    print("  " + "-" * 74)
    for r in results:
        print(
            f"  {r['backend']:<12} {r['model'][:28]:<28} "
            f"{_pct(r['first_try_ok'], r['n']):>8} "
            f"{_pct(r['final_ok'], r['n']):>8} "
            f"{r['mean_s']:>7.2f}s {r['median_s']:>7.2f}s {r['p95_s']:>7.2f}s"
        )
    print()
    for r in results:
        extra = r["total_attempts"] - r["n"]
        print(
            f"  {r['backend']}: {r['total_attempts']} calls for {r['n']} decisions "
            f"({extra} retr{'y' if extra == 1 else 'ies'})."
        )
    print(f"\n  Parse failures logged to: {log_path}")

    # Side-by-side decisions
    print("\n" + "=" * 78)
    print("  DECISIONS SIDE BY SIDE")
    print("=" * 78)
    n = max(r["n"] for r in results)
    for i in range(n):
        arch = next(
            (row["archetype"] for r in results for row in r["rows"] if row["idx"] == i),
            "?",
        )
        print(f"\n  [{i:02d}] {arch}")
        for r in results:
            row = next((x for x in r["rows"] if x["idx"] == i), None)
            if row is None:
                continue
            print(
                f"      {r['backend']:<10} {_fmt_decision(row['decision'])}"
                f"  ({row['seconds']:.2f}s, {row['attempts']} call"
                f"{'' if row['attempts'] == 1 else 's'})"
            )
            reason = _reasoning(row["decision"])
            if reason:
                print(f"      {'':<10} └─ {reason}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"Number of prompts (default {DEFAULT_N}).")
    ap.add_argument("--seed", type=int, default=7,
                    help="Seed for synthetic market states (default 7).")
    ap.add_argument("--anthropic-model", default=None,
                    help="Override the Anthropic model name.")
    ap.add_argument("--ollama-model", default=llm_backends.DEFAULT_OLLAMA_MODEL,
                    help=f"Ollama model (default {llm_backends.DEFAULT_OLLAMA_MODEL}).")
    ap.add_argument("--ollama-host", default=llm_backends.DEFAULT_OLLAMA_HOST,
                    help="Ollama server URL.")
    ap.add_argument("--retries", type=int, default=3,
                    help="Max attempts per prompt for Ollama (default 3).")
    ap.add_argument("--json-mode", action="store_true",
                    help="Use Ollama's native format=json constraint.")
    ap.add_argument("--only", choices=["anthropic", "ollama"], default=None,
                    help="Run just one backend.")
    ap.add_argument("--log", default=DEFAULT_LOG,
                    help=f"Parse-failure JSONL path (default {DEFAULT_LOG}).")
    ap.add_argument("--dump-json", default=None,
                    help="Also write the full result set to this JSON path.")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("  Backend pilot — Anthropic vs. Ollama")
    print(f"  {args.n} identical prompts per backend, seed={args.seed}")
    print("=" * 78)

    prompts = build_prompts(args.n, seed=args.seed)
    llm_backends.reset_stats()
    results: list[dict] = []

    if args.only != "ollama":
        anth = AnthropicBackend(model=args.anthropic_model, log_path=args.log)
        err = anth.client_error
        if err:
            print(f"\n  anthropic UNAVAILABLE — {err}")
            print("  Set ANTHROPIC_API_KEY (or run `ant auth login`) to include it.")
        else:
            results.append(run_backend(anth, prompts, f"anthropic ({anth.model})"))

    if args.only != "anthropic":
        oll = OllamaBackend(
            model=args.ollama_model,
            host=args.ollama_host,
            max_attempts=args.retries,
            json_mode=args.json_mode,
            log_path=args.log,
        )
        results.append(run_backend(oll, prompts, f"ollama ({oll.model})"))

    if not results:
        print("\n  No backend produced results.")
        return 0

    print_report(results, args.log)

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "config": {
                        "n": args.n, "seed": args.seed,
                        "ollama_model": args.ollama_model,
                        "json_mode": args.json_mode, "retries": args.retries,
                    },
                    "results": results,
                },
                fh, indent=2, default=str,
            )
        print(f"\n  Full results → {args.dump_json}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
