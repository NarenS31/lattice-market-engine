"""
analysis.py — Monte-Carlo analysis of the lattice market engine.

Runs 100 simulations with different random seeds and aggregates:
  1. Return distribution (kurtosis, skewness, fat-tail test)
  2. Flash crash detection (>2% price drop in a single tick)
  3. Agent archetype P&L by shock regime
  4. Post-shock wealth correlation (contagion measure)

Saves all results to results.json and prints a clean summary report.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict

import numpy as np

from engine import OrderBook
from agents import Simulation, RetailAgent, InstitutionalAgent, QuantAgent
from factor_pulse import FactorPulse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_RUNS = 100
N_TICKS = 25
SHOCK_TICK = 10
POST_SHOCK_WINDOW = 5  # ticks post-shock to measure contagion

INITIAL_CASH = {"retail": 10_000.0, "inst": 100_000.0, "quant": 50_000.0}
# Agent list layout: indices 0-4 retail, 5-6 institutional, 7-8 quant
ARCHETYPE_IDX = {"retail": range(0, 5), "inst": range(5, 7), "quant": range(7, 9)}
N_AGENTS = 9


# ---------------------------------------------------------------------------
# Pure-NumPy moments (avoids scipy dependency)
# ---------------------------------------------------------------------------
def _kurtosis(arr: np.ndarray) -> float:
    sigma = np.std(arr)
    if sigma == 0:
        return float("nan")
    return float(np.mean(((arr - np.mean(arr)) / sigma) ** 4))


def _skewness(arr: np.ndarray) -> float:
    sigma = np.std(arr)
    if sigma == 0:
        return float("nan")
    return float(np.mean(((arr - np.mean(arr)) / sigma) ** 3))


# ---------------------------------------------------------------------------
# Statistical test helpers (no scipy — stdlib math + numpy only)
# ---------------------------------------------------------------------------

def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Exact normal CDF via stdlib erf."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _ks_pvalue(ks_stat: float, n: int) -> float:
    """
    Asymptotic p-value from the Kolmogorov distribution:
      P = 2 * sum_{k=1}^inf (-1)^{k+1} exp(-2 k^2 lambda^2),  lambda = D * sqrt(n)
    Series converges to machine precision in < 10 terms for typical inputs.
    """
    lam = ks_stat * math.sqrt(n)
    p = 0.0
    for k in range(1, 101):
        term = (-1) ** (k + 1) * math.exp(-2.0 * k * k * lam * lam)
        p += term
        if abs(term) < 1e-15:
            break
    return min(1.0, max(0.0, 2.0 * p))


def _jarque_bera(ret: np.ndarray) -> tuple[float, float]:
    """
    Jarque-Bera test statistic and exact p-value.
    JB = n/6 * (S^2 + K^2/4),  K = excess kurtosis = kurtosis - 3.
    Under H0: JB ~ chi^2(2), so p = exp(-JB/2)  [exact for 2 df].
    """
    n = len(ret)
    s = _skewness(ret)
    k_excess = _kurtosis(ret) - 3.0
    jb = (n / 6.0) * (s ** 2 + k_excess ** 2 / 4.0)
    pval = math.exp(-jb / 2.0)
    return float(jb), float(pval)


def _adf_stat(series: np.ndarray) -> float:
    """
    Basic Dickey-Fuller t-stat for gamma in:  Δy_t = alpha + gamma * y_{t-1} + e_t
    Critical values (with constant, no trend):  1%=-3.43  5%=-2.86  10%=-2.57
    """
    if len(series) < 4:
        return float("nan")
    y = series.astype(float)
    dy = np.diff(y)
    y_lag = y[:-1]
    X = np.column_stack([np.ones(len(dy)), y_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
        resid = dy - X @ beta
        s2 = float(np.dot(resid, resid)) / (len(dy) - 2)
        cov = s2 * np.linalg.inv(X.T @ X)
        se = math.sqrt(max(cov[1, 1], 0.0))
        return float(beta[1] / se) if se > 0 else float("nan")
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return float("nan")


def _bootstrap_kurtosis_ci(ret: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    """95% bootstrap CI for kurtosis via percentile method (1000 resamples, fixed seed)."""
    rng = np.random.default_rng(42)
    boot = np.array([
        _kurtosis(rng.choice(ret, size=len(ret), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _ks_test(ret: np.ndarray) -> tuple[float, float]:
    """KS statistic and asymptotic p-value comparing empirical returns to a fitted normal."""
    mu = float(np.mean(ret))
    sigma = float(np.std(ret))
    if sigma == 0:
        return float("nan"), float("nan")
    n = len(ret)
    sorted_ret = np.sort(ret)
    emp = np.arange(1, n + 1) / n
    theo = np.array([_normal_cdf(x, mu, sigma) for x in sorted_ret])
    ks = float(np.max(np.abs(emp - theo)))
    return ks, _ks_pvalue(ks, n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_regime(macro_log: list[dict]) -> str:
    """Identify which shock fired by inspecting macro state at SHOCK_TICK.

    Handles both the new supply-chain event format (event_name key) and the
    legacy numeric format (fed_rate / sentiment / vix keys) transparently.
    """
    if len(macro_log) <= SHOCK_TICK:
        return "baseline"
    ctx = macro_log[SHOCK_TICK]
    # Supply-chain event format: use the event name directly as the regime label.
    event = ctx.get("event_name")
    if event:
        return event
    # Legacy numeric macro format.
    if ctx.get("fed_rate", 5.25) > 5.5:
        return "rate_spike"
    if ctx.get("sentiment", 0.1) < -0.6:
        return "sentiment_crash"
    if ctx.get("vix", 18.0) > 40:
        return "liquidity_shock"
    return "baseline"


def tick_returns(clearing_prices: list[float | None]) -> list[float]:
    valid = [p for p in clearing_prices if p is not None]
    if len(valid) < 2:
        return []
    return [(valid[i] - valid[i - 1]) / valid[i - 1] for i in range(1, len(valid))]


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------
def run_simulation(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    agents = (
        [RetailAgent(cash=INITIAL_CASH["retail"], name=f"Retail-{i+1}") for i in range(5)]
        + [InstitutionalAgent(cash=INITIAL_CASH["inst"], name=f"Inst-{i+1}") for i in range(2)]
        + [QuantAgent(cash=INITIAL_CASH["quant"], name=f"Quant-{i+1}") for i in range(2)]
    )

    book = OrderBook(symbol="SIM")
    pulse = FactorPulse()
    sim = Simulation(book, agents)
    sim.seed_book(fair_value=100.0)
    summaries = sim.run(n_ticks=N_TICKS, factor_pulse=pulse)

    clearing_prices: list[float | None] = [s.clearing_price for s in summaries]
    regime = detect_regime(sim.macro_log)

    # P&L = final mark-to-market wealth minus initial cash
    initial_wealth = (
        [INITIAL_CASH["retail"]] * 5
        + [INITIAL_CASH["inst"]] * 2
        + [INITIAL_CASH["quant"]] * 2
    )
    final_wealth = sim.wealth_log[-1] if sim.wealth_log else initial_wealth
    pnl = [final_wealth[i] - initial_wealth[i] for i in range(N_AGENTS)]

    # Per-tick wealth deltas in the 5 ticks starting at SHOCK_TICK
    shock_deltas: dict[str, list[float]] = {"retail": [], "inst": [], "quant": []}
    if len(sim.wealth_log) > SHOCK_TICK + POST_SHOCK_WINDOW:
        for t in range(SHOCK_TICK, SHOCK_TICK + POST_SHOCK_WINDOW):
            prev = sim.wealth_log[t - 1]
            curr = sim.wealth_log[t]
            shock_deltas["retail"].append(
                float(np.mean([curr[i] - prev[i] for i in ARCHETYPE_IDX["retail"]]))
            )
            shock_deltas["inst"].append(
                float(np.mean([curr[i] - prev[i] for i in ARCHETYPE_IDX["inst"]]))
            )
            shock_deltas["quant"].append(
                float(np.mean([curr[i] - prev[i] for i in ARCHETYPE_IDX["quant"]]))
            )

    # Per-tick mean wealth by archetype (used by visualize.py for Fig 3 & 4)
    wealth_by_tick = [
        {
            "retail": float(np.mean([w[i] for i in ARCHETYPE_IDX["retail"]])),
            "inst":   float(np.mean([w[i] for i in ARCHETYPE_IDX["inst"]])),
            "quant":  float(np.mean([w[i] for i in ARCHETYPE_IDX["quant"]])),
        }
        for w in sim.wealth_log
    ]

    # Bucket per-tick microstructure metrics by pre/post shock.
    micro_pre: list[dict] = []
    micro_post: list[dict] = []
    for i, s in enumerate(summaries):
        entry = {
            "kyles_lambda": s.kyles_lambda,
            "amihud_illiquidity": s.amihud_illiquidity,
            "spread_adverse_selection": s.spread_adverse_selection,
            "spread_inventory_cost": s.spread_inventory_cost,
        }
        (micro_post if i >= SHOCK_TICK else micro_pre).append(entry)

    return {
        "seed": seed,
        "regime": regime,
        "clearing_prices": clearing_prices,
        "pnl": pnl,
        "shock_deltas": shock_deltas,
        "micro_pre": micro_pre,
        "micro_post": micro_post,
        "wealth_by_tick": wealth_by_tick,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Running {N_RUNS} simulations ({N_TICKS} ticks each)...")
    all_runs: list[dict] = []
    for seed in range(N_RUNS):
        all_runs.append(run_simulation(seed))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{N_RUNS} complete", flush=True)

    # -------------------------------------------------------------------
    # 1. Return distribution + statistical tests
    # -------------------------------------------------------------------
    all_returns: list[float] = []
    for run in all_runs:
        all_returns.extend(tick_returns(run["clearing_prices"]))

    ret = np.array(all_returns)
    kurt = _kurtosis(ret)
    skew = _skewness(ret)
    is_fat_tailed = kurt > 3

    # [A] Jarque-Bera
    jb_stat, jb_pval = _jarque_bera(ret)
    jb_reject = jb_pval < 0.05

    # [B] ADF — run per simulation, report fraction non-stationary (prices) / stationary (returns)
    _ADF_CV5 = -2.86
    adf_price_tstats: list[float] = []
    adf_ret_tstats:   list[float] = []
    for run in all_runs:
        prices_clean = np.array(
            [p for p in run["clearing_prices"] if p is not None], dtype=float
        )
        if len(prices_clean) >= 4:
            t = _adf_stat(prices_clean)
            if not math.isnan(t):
                adf_price_tstats.append(t)
        rets_run = tick_returns(run["clearing_prices"])
        if len(rets_run) >= 4:
            t = _adf_stat(np.array(rets_run, dtype=float))
            if not math.isnan(t):
                adf_ret_tstats.append(t)

    # Non-stationary price = fail to reject H0 (t > CV)
    price_nonstatio_n = sum(t > _ADF_CV5 for t in adf_price_tstats)
    # Stationary returns = reject H0 (t < CV)
    ret_statio_n      = sum(t < _ADF_CV5 for t in adf_ret_tstats)
    mean_adf_price    = float(np.mean(adf_price_tstats)) if adf_price_tstats else float("nan")
    mean_adf_ret      = float(np.mean(adf_ret_tstats))   if adf_ret_tstats   else float("nan")
    n_adf_p           = len(adf_price_tstats)
    n_adf_r           = len(adf_ret_tstats)

    # [C] Bootstrap CI for kurtosis
    print("   Bootstrap CI for kurtosis (1000 resamples)...", flush=True)
    ci_lo, ci_hi = _bootstrap_kurtosis_ci(ret)
    normal_kurt_in_ci = ci_lo <= 3.0 <= ci_hi

    # [D] KS test
    ks_stat, ks_pval = _ks_test(ret)
    ks_reject = ks_pval < 0.05

    print("\n" + "=" * 66)
    print("1. RETURN DISTRIBUTION ANALYSIS")
    print("=" * 66)
    print(f"   Observations : {len(all_returns)}")
    print(f"   Kurtosis     : {kurt:.4f}  (normal = 3.0)")
    print(f"   Skewness     : {skew:.4f}  (normal = 0.0)")
    print(
        f"   Fat-tailed   : {'YES — heavier tails than a normal distribution' if is_fat_tailed else 'NO — tails are normal or thinner'}"
    )

    print(f"\n   [A] Jarque-Bera Normality Test  (JB = n/6·(S² + K²/4), chi²(2) p-value)")
    print(f"       JB statistic : {jb_stat:.4f}")
    print(f"       p-value      : {jb_pval:.4e}")
    print(
        f"       Verdict      : "
        f"{'REJECT normality at 95% confidence — returns are non-normal' if jb_reject else 'FAIL TO REJECT normality at 95% confidence'}"
    )

    print(f"\n   [B] Dickey-Fuller Stationarity Test  (5% critical value = {_ADF_CV5})")
    print(
        f"       Prices  — mean t-stat : {mean_adf_price:7.3f}  |  "
        f"non-stationary in {price_nonstatio_n}/{n_adf_p} runs  "
        f"({price_nonstatio_n / n_adf_p * 100:.0f}%)"
        if n_adf_p else "       Prices  — insufficient data"
    )
    print(
        f"       Returns — mean t-stat : {mean_adf_ret:7.3f}  |  "
        f"stationary in {ret_statio_n}/{n_adf_r} runs  "
        f"({ret_statio_n / n_adf_r * 100:.0f}%)"
        if n_adf_r else "       Returns — insufficient data"
    )

    print(f"\n   [C] Bootstrap 95% CI for Kurtosis  (1000 resamples, percentile method)")
    print(f"       Kurtosis : {kurt:.4f}   95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(
        f"       Normal kurtosis (3.0) is "
        f"{'INSIDE  the CI — cannot rule out normal kurtosis' if normal_kurt_in_ci else 'OUTSIDE the CI — kurtosis is significantly above normal'}"
    )

    print(f"\n   [D] Kolmogorov-Smirnov Test vs Fitted Normal")
    print(f"       KS statistic : {ks_stat:.6f}")
    print(f"       p-value      : {ks_pval:.4e}")
    print(
        f"       Verdict      : "
        f"{'REJECT normality — empirical distribution differs significantly from normal' if ks_reject else 'FAIL TO REJECT — distribution is consistent with fitted normal'}"
    )

    # -------------------------------------------------------------------
    # 2. Flash crash detection
    # -------------------------------------------------------------------
    crash_run_count = 0
    crash_ticks: list[int] = []

    for run in all_runs:
        prices = run["clearing_prices"]
        run_crashed = False
        for i in range(1, len(prices)):
            p0, p1 = prices[i - 1], prices[i]
            if p0 is not None and p1 is not None and (p1 - p0) / p0 < -0.02:
                run_crashed = True
                crash_ticks.append(i)
        if run_crashed:
            crash_run_count += 1

    print("\n" + "=" * 62)
    print("2. FLASH CRASH DETECTION  (>2% single-tick price drop)")
    print("=" * 62)
    print(f"   Runs with ≥1 crash : {crash_run_count} / {N_RUNS}  ({crash_run_count}%)")
    print(f"   Total crash events : {len(crash_ticks)}")
    if crash_ticks:
        ta = np.array(crash_ticks)
        tick_freq: dict[int, int] = defaultdict(int)
        for t in crash_ticks:
            tick_freq[t] += 1
        peak_tick = max(tick_freq, key=tick_freq.__getitem__)
        print(f"   Mean crash tick    : {np.mean(ta):.1f}")
        print(f"   Median crash tick  : {np.median(ta):.1f}")
        print(f"   Peak crash tick    : tick {peak_tick} ({tick_freq[peak_tick]} events)")
    else:
        print("   No flash crash events detected.")

    # -------------------------------------------------------------------
    # 3. Agent archetype P&L by regime
    # -------------------------------------------------------------------
    regime_pnl: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"retail": [], "inst": [], "quant": []}
    )
    for run in all_runs:
        r = run["regime"]
        p = run["pnl"]
        regime_pnl[r]["retail"].append(float(np.mean([p[i] for i in ARCHETYPE_IDX["retail"]])))
        regime_pnl[r]["inst"].append(float(np.mean([p[i] for i in ARCHETYPE_IDX["inst"]])))
        regime_pnl[r]["quant"].append(float(np.mean([p[i] for i in ARCHETYPE_IDX["quant"]])))

    print("\n" + "=" * 62)
    print("3. AGENT ARCHETYPE P&L BY REGIME")
    print("=" * 62)

    pnl_summary: dict = {}
    for regime in sorted(regime_pnl):
        data = regime_pnl[regime]
        n = len(data["retail"])
        print(f"\n   [{regime.upper()}]  ({n} runs)")
        print(f"   {'Archetype':<12} {'Mean P&L':>12}  {'Std':>10}")
        print(f"   {'-' * 38}")
        avgs: dict[str, float] = {}
        arch_stats: dict = {}
        for arch in ("retail", "inst", "quant"):
            vals = np.array(data[arch])
            m, s = float(np.mean(vals)), float(np.std(vals))
            avgs[arch] = m
            arch_stats[arch] = {"mean_pnl": round(m, 4), "std_pnl": round(s, 4)}
            print(f"   {arch:<12} {m:>12.2f}  {s:>10.2f}")
        best = max(avgs, key=avgs.__getitem__)
        worst = min(avgs, key=avgs.__getitem__)
        print(f"   Best: {best.upper()}   Worst: {worst.upper()}")
        pnl_summary[regime] = {
            "n_runs": n,
            "archetypes": arch_stats,
            "best_archetype": best,
            "worst_archetype": worst,
        }

    # -------------------------------------------------------------------
    # 4. Contagion graph
    # -------------------------------------------------------------------
    correlations: list[float] = []
    for run in all_runs:
        inst_s = run["shock_deltas"]["inst"]
        quant_s = run["shock_deltas"]["quant"]
        if len(inst_s) < 2 or len(quant_s) < 2:
            continue
        c = np.corrcoef(inst_s, quant_s)[0, 1]
        if not np.isnan(c):
            correlations.append(float(c))

    mean_corr = float(np.mean(correlations)) if correlations else float("nan")
    high_contagion_runs = int(np.sum(np.array(correlations) > 0.5)) if correlations else 0
    overall_contagion = mean_corr > 0.5

    print("\n" + "=" * 62)
    print("4. CONTAGION GRAPH  (Institutional vs Quant post-shock)")
    print("=" * 62)
    print(f"   Valid runs          : {len(correlations)}")
    print(f"   Mean correlation    : {mean_corr:.4f}")
    if correlations:
        ca = np.array(correlations)
        print(f"   Min / Median / Max  : {np.min(ca):.4f} / {np.median(ca):.4f} / {np.max(ca):.4f}")
        print(f"   High-contagion runs : {high_contagion_runs} / {len(correlations)}  (corr > 0.5)")
    print(
        f"   Contagion verdict   : {'HIGH — agents sell together, amplifying the crash' if overall_contagion else 'LOW — agents behave independently post-shock'}"
    )

    # -------------------------------------------------------------------
    # 5. Market microstructure — pre vs post shock
    # -------------------------------------------------------------------
    _METRICS = [
        ("kyles_lambda",            "Kyle's Lambda"),
        ("amihud_illiquidity",      "Amihud Illiquidity"),
        ("spread_adverse_selection","Spread: Adverse Selection"),
        ("spread_inventory_cost",   "Spread: Inventory Cost"),
    ]

    pre_buckets:  dict[str, list[float]] = {k: [] for k, _ in _METRICS}
    post_buckets: dict[str, list[float]] = {k: [] for k, _ in _METRICS}

    for run in all_runs:
        for entry in run["micro_pre"]:
            for key, _ in _METRICS:
                if entry[key] is not None:
                    pre_buckets[key].append(entry[key])
        for entry in run["micro_post"]:
            for key, _ in _METRICS:
                if entry[key] is not None:
                    post_buckets[key].append(entry[key])

    def _fmt(v: float) -> str:
        return f"{v:.4e}" if abs(v) < 1e-3 else f"{v:.6f}"

    print("\n" + "=" * 70)
    print("5. MARKET MICROSTRUCTURE METRICS  (Pre-Shock vs Post-Shock)")
    print("=" * 70)
    print(f"   Shock fires at tick {SHOCK_TICK}  |  {N_RUNS} simulation runs\n")
    print(f"   {'Metric':<28}  {'Pre-Shock':>12}  {'Post-Shock':>12}  {'Change':>9}")
    print(f"   {'-' * 66}")

    micro_summary: dict = {}
    for key, label in _METRICS:
        pre_arr = np.array(pre_buckets[key])
        post_arr = np.array(post_buckets[key])
        pre_mean  = float(np.mean(pre_arr))  if len(pre_arr)  else float("nan")
        post_mean = float(np.mean(post_arr)) if len(post_arr) else float("nan")

        if not (np.isnan(pre_mean) or np.isnan(post_mean) or pre_mean == 0):
            chg_pct = (post_mean - pre_mean) / abs(pre_mean) * 100.0
            chg_str = f"{chg_pct:+.1f}%"
        else:
            chg_str = "N/A"

        pre_str  = _fmt(pre_mean)  if not np.isnan(pre_mean)  else "N/A"
        post_str = _fmt(post_mean) if not np.isnan(post_mean) else "N/A"
        print(f"   {label:<28}  {pre_str:>12}  {post_str:>12}  {chg_str:>9}")
        micro_summary[key] = {
            "pre_mean":  round(pre_mean,  8) if not np.isnan(pre_mean)  else None,
            "post_mean": round(post_mean, 8) if not np.isnan(post_mean) else None,
            "change_pct": round(chg_pct, 4) if chg_str != "N/A" else None,
        }

    # -------------------------------------------------------------------
    # Visualization data (consumed by visualize.py)
    # -------------------------------------------------------------------
    # Forward-fill None clearing prices so price paths are continuous.
    price_matrix = np.full((N_RUNS, N_TICKS), np.nan)
    for i, run in enumerate(all_runs):
        last_p: float | None = None
        for j, p in enumerate(run["clearing_prices"]):
            if p is not None:
                last_p = p
            if last_p is not None:
                price_matrix[i, j] = last_p

    price_series_sample = [
        [None if np.isnan(v) else round(float(v), 4) for v in price_matrix[i]]
        for i in range(min(10, N_RUNS))
    ]
    price_mean_series = [
        None if np.isnan(v) else round(float(v), 4)
        for v in np.nanmean(price_matrix, axis=0)
    ]

    # Per-archetype mean wealth per tick, averaged across all 100 runs.
    arch_wealth: dict[str, list] = {"retail": [], "inst": [], "quant": []}
    for t in range(N_TICKS):
        for arch in ("retail", "inst", "quant"):
            vals = [
                run["wealth_by_tick"][t][arch]
                for run in all_runs
                if t < len(run.get("wealth_by_tick", []))
            ]
            arch_wealth[arch].append(round(float(np.mean(vals)), 4) if vals else None)

    # 3×3 post-shock wealth-change correlation matrix (stacked across valid runs).
    stacked: dict[str, list[float]] = {"retail": [], "inst": [], "quant": []}
    for run in all_runs:
        sd = run["shock_deltas"]
        if all(len(sd.get(a, [])) == POST_SHOCK_WINDOW for a in ("retail", "inst", "quant")):
            for arch in ("retail", "inst", "quant"):
                stacked[arch].extend(sd[arch])

    if stacked["retail"]:
        corr3 = np.corrcoef([stacked["retail"], stacked["inst"], stacked["quant"]])
        post_shock_corr: list | None = [[round(v, 6) for v in row] for row in corr3.tolist()]
    else:
        post_shock_corr = None

    viz_data: dict = {
        "raw_returns":                 [round(r, 8) for r in all_returns],
        "price_series_sample":         price_series_sample,
        "price_mean_series":           price_mean_series,
        "wealth_by_archetype":         arch_wealth,
        "initial_wealth_by_archetype": {
            "retail": INITIAL_CASH["retail"],
            "inst":   INITIAL_CASH["inst"],
            "quant":  INITIAL_CASH["quant"],
        },
        "post_shock_corr_matrix":      post_shock_corr,
        "post_shock_corr_labels":      ["Retail", "Institutional", "Quant"],
    }

    # -------------------------------------------------------------------
    # Save results.json
    # -------------------------------------------------------------------
    results = {
        "meta": {"n_runs": N_RUNS, "n_ticks": N_TICKS, "shock_tick": SHOCK_TICK},
        "return_distribution": {
            "n_observations": len(all_returns),
            "kurtosis": round(kurt, 6),
            "skewness": round(skew, 6),
            "is_fat_tailed": bool(is_fat_tailed),
        },
        "flash_crash": {
            "runs_with_crash": crash_run_count,
            "crash_rate_pct": crash_run_count,
            "total_events": len(crash_ticks),
            "mean_tick": round(float(np.mean(crash_ticks)), 2) if crash_ticks else None,
            "median_tick": round(float(np.median(crash_ticks)), 2) if crash_ticks else None,
            "tick_distribution": {
                str(t): crash_ticks.count(t) for t in sorted(set(crash_ticks))
            },
        },
        "pnl_by_regime": pnl_summary,
        "contagion": {
            "n_valid_runs": len(correlations),
            "mean_correlation": round(mean_corr, 6) if not np.isnan(mean_corr) else None,
            "high_contagion_runs": high_contagion_runs,
            "overall_high_contagion": bool(overall_contagion),
            "distribution": {
                "min": round(float(np.min(correlations)), 6) if correlations else None,
                "median": round(float(np.median(correlations)), 6) if correlations else None,
                "max": round(float(np.max(correlations)), 6) if correlations else None,
            },
        },
        "microstructure_shock_comparison": micro_summary,
        "visualization_data": viz_data,
        "statistical_tests": {
            "jarque_bera": {
                "statistic": round(jb_stat, 6),
                "pvalue": round(jb_pval, 8),
                "reject_normality_95pct": bool(jb_reject),
                "formula": "n/6 * (S^2 + K_excess^2 / 4), p from chi^2(2)",
            },
            "dickey_fuller_stationarity": {
                "critical_value_5pct": _ADF_CV5,
                "model": "with constant, no trend",
                "prices": {
                    "mean_tstat": round(mean_adf_price, 4) if not math.isnan(mean_adf_price) else None,
                    "nonstationary_runs": price_nonstatio_n,
                    "total_runs": n_adf_p,
                    "nonstationary_frac": round(price_nonstatio_n / n_adf_p, 4) if n_adf_p else None,
                },
                "returns": {
                    "mean_tstat": round(mean_adf_ret, 4) if not math.isnan(mean_adf_ret) else None,
                    "stationary_runs": ret_statio_n,
                    "total_runs": n_adf_r,
                    "stationary_frac": round(ret_statio_n / n_adf_r, 4) if n_adf_r else None,
                },
            },
            "bootstrap_kurtosis_ci": {
                "n_resamples": 1000,
                "kurtosis_point_estimate": round(kurt, 6),
                "ci_lower_95": round(ci_lo, 6),
                "ci_upper_95": round(ci_hi, 6),
                "normal_kurtosis_in_ci": bool(normal_kurt_in_ci),
            },
            "ks_test_vs_normal": {
                "statistic": round(ks_stat, 8),
                "pvalue": round(ks_pval, 8),
                "reject_normality_95pct": bool(ks_reject),
                "method": "asymptotic Kolmogorov distribution",
            },
        },
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 62)
    print("Saved → results.json")
    print("=" * 62)


if __name__ == "__main__":
    main()
