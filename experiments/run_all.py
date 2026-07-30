"""
experiments/run_all.py — master runner for all three research experiments.

Usage
-----
  # Run all three experiments (default: 10 runs, 20 ticks each)
  python -m experiments.run_all

  # Run a single experiment
  python -m experiments.run_all --exp 1
  python -m experiments.run_all --exp 2
  python -m experiments.run_all --exp 3

  # Adjust Monte Carlo size
  python -m experiments.run_all --exp 1 --runs 20 --ticks 25

  # Quick smoke-test (1 run, 5 ticks — just checks plumbing)
  python -m experiments.run_all --runs 1 --ticks 5

Cost notes
----------
  Each run × tick × LLM agent = 1 Anthropic API call.
  Prompt caching is enabled on every call (system prompts are cached).

  Default (10 runs, 20 ticks, 3 LLM agents):
    Exp 1: 10 × 20 × 3 × 2 conditions = 1 200 calls  (~$0.50 with Sonnet + caching)
    Exp 2: 10 × 25 × ~2 LLM × 4 conditions = 2 000 calls  (~$0.80)
    Exp 3:  5 × 20 × 3 × 2 models          =   600 calls  (~$0.30)
  Total: ~3 800 cached API calls  ≈  $1–2
"""
from __future__ import annotations

import argparse
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.behavioral    import run_experiment1  # noqa: E402
from experiments.kyle_asymmetry import run_experiment2  # noqa: E402
from experiments.model_scale    import run_experiment3  # noqa: E402
from experiments.design_2x2     import run_experiment4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM market research experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Which experiment to run (1-4). Omit to run 1-3; exp 4 (the 2x2) must be requested explicitly — it is ~4,800 LLM calls.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Monte Carlo runs per condition (default: 10).",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=20,
        help="Ticks per simulation run (default: 20).",
    )
    args = parser.parse_args()

    run_all = args.exp is None
    n_runs  = args.runs
    n_ticks = args.ticks

    if run_all or args.exp == 1:
        run_experiment1(n_runs=n_runs, n_ticks=n_ticks)

    if run_all or args.exp == 2:
        # Experiment 2 uses 25 ticks by default so there are 15 post-shock ticks.
        # If the user passed --ticks explicitly, honour it; otherwise use 25.
        ticks2 = n_ticks if args.ticks != 20 else 25
        run_experiment2(n_runs=n_runs, n_ticks=ticks2)

    if run_all or args.exp == 3:
        # Experiment 3 defaults to 5 runs (two API models, costs double).
        runs3 = n_runs if args.runs != 10 else min(n_runs, 5)
        run_experiment3(n_runs=runs3, n_ticks=n_ticks)

    if args.exp == 4:
        # Experiment 4 (2x2 framing x realism) is deliberately excluded from
        # the run-all sweep: at default settings it is ~4,800 LLM calls, an
        # order of magnitude more than experiments 1-3 combined. Ask for it.
        # It is resumable — re-running skips completed runs.
        #
        # This runs the default model only. For the cross-model replication
        # check (both models, ~9,600 calls, local so no API cost) call the
        # runner directly:
        #     python -m experiments.design_2x2 --all-models --reps 5
        run_experiment4(reps=args.runs if args.runs != 10 else 5, n_ticks=n_ticks)


if __name__ == "__main__":
    main()
