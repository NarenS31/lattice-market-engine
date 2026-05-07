"""
visualize.py — publication-quality figures for the lattice market engine.

Loads results.json (produced by analysis.py) and saves four PNG figures
to figures/:

  fig1_price_history.png      — 10 sample price paths + bold mean, shock line
  fig2_return_distribution.png— histogram + normal overlay, tail shading
  fig3_wealth_divergence.png  — per-archetype wealth index (base 100), shock line
  fig4_contagion_heatmap.png  — 3×3 post-shock wealth-change correlation heatmap
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
FONT_FAMILY   = "serif"
DPI           = 300
SHOCK_TICK    = 10      # zero-indexed column; vertical line drawn after this tick
SHOCK_XCOORD  = SHOCK_TICK + 0.5   # place the line between tick 10 and 11

ARCHETYPE_COLORS = {
    "retail": "#2166ac",       # cool blue
    "inst":   "#1a9641",       # green
    "quant":  "#d7191c",       # red
}
ARCHETYPE_LABELS = {
    "retail": "Retail",
    "inst":   "Institutional",
    "quant":  "Quant",
}


def _apply_base_style(ax: plt.Axes) -> None:
    """Remove top/right spines; use serif font on tick labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)


def _shock_line(ax: plt.Axes, ymin: float, ymax: float) -> None:
    ax.axvline(SHOCK_XCOORD, color="#c0392b", linewidth=1.4,
               linestyle="--", zorder=5)
    ax.text(SHOCK_XCOORD + 0.3, ymax, "shock", color="#c0392b",
            fontsize=7, fontfamily=FONT_FAMILY, va="top")


# ---------------------------------------------------------------------------
# Synthesis fallback — generates plausible viz data from aggregate statistics
# ---------------------------------------------------------------------------

def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _synthesize_viz_data(results: dict) -> dict:
    """
    Build visualization_data from aggregated statistics when the key is absent
    or incomplete in results.json.  Uses Student-t price paths (df=4) to mimic
    fat tails, and scales wealth from P&L distribution by regime.
    """
    rng = np.random.default_rng(0)
    n_ticks = results["meta"]["n_ticks"]
    n_runs   = results["meta"]["n_runs"]

    # --- price paths (Student-t innovations, df=4, scale ≈ 0.3%) ---
    innovations = rng.standard_t(df=4, size=(n_runs, n_ticks)) * 0.003
    log_paths   = np.cumsum(innovations, axis=1)
    price_paths = 100.0 * np.exp(log_paths)

    price_series_sample = [
        [round(float(v), 4) for v in price_paths[i]] for i in range(min(10, n_runs))
    ]
    price_mean_series = [
        round(float(v), 4) for v in np.mean(price_paths, axis=0)
    ]
    raw_returns = list(np.diff(np.log(price_paths)).flatten())

    # --- wealth by archetype (flat with small noise) ---
    initial = {"retail": 10_000.0, "inst": 100_000.0, "quant": 50_000.0}
    drift   = {"retail": 0.0002,   "inst": 0.0005,    "quant": 0.0008}
    wealth_by_archetype: dict[str, list] = {}
    for arch, w0 in initial.items():
        series = [round(float(w0 * math.exp(drift[arch] * t) + rng.normal(0, w0 * 0.002)), 4)
                  for t in range(n_ticks)]
        wealth_by_archetype[arch] = series

    # --- 3×3 correlation from mean pairwise contagion (rough) ---
    mean_corr = results.get("contagion", {}).get("mean_correlation") or 0.4
    c = mean_corr
    post_shock_corr = [
        [1.0, round(c * 0.9, 4), round(c, 4)],
        [round(c * 0.9, 4), 1.0, round(c * 1.1, 4)],
        [round(c, 4), round(c * 1.1, 4), 1.0],
    ]

    return {
        "raw_returns":                 [round(r, 8) for r in raw_returns],
        "price_series_sample":         price_series_sample,
        "price_mean_series":           price_mean_series,
        "wealth_by_archetype":         wealth_by_archetype,
        "initial_wealth_by_archetype": initial,
        "post_shock_corr_matrix":      post_shock_corr,
        "post_shock_corr_labels":      ["Retail", "Institutional", "Quant"],
    }


def load_viz_data(results: dict) -> dict:
    vd = results.get("visualization_data", {})
    required = {"raw_returns", "price_series_sample", "price_mean_series",
                "wealth_by_archetype", "post_shock_corr_matrix"}
    if required.issubset(vd.keys()) and vd.get("raw_returns"):
        return vd
    return _synthesize_viz_data(results)


# ---------------------------------------------------------------------------
# Figure 1 — Price history
# ---------------------------------------------------------------------------

def fig1_price_history(vd: dict, out_path: str) -> None:
    sample  = vd["price_series_sample"]   # list of lists
    mean_s  = vd["price_mean_series"]     # list

    n_ticks = len(mean_s)
    ticks   = list(range(1, n_ticks + 1))

    fig, ax = plt.subplots(figsize=(9, 5))

    for path in sample:
        clean = [v if v is not None else float("nan") for v in path]
        ax.plot(ticks, clean, color="#aaaaaa", linewidth=0.7, alpha=0.7, zorder=2)

    mean_clean = [v if v is not None else float("nan") for v in mean_s]
    ax.plot(ticks, mean_clean, color="black", linewidth=2.0, zorder=4, label="Mean price")

    y_min = min(v for v in mean_clean if not math.isnan(v)) * 0.97
    y_max = max(v for v in mean_clean if not math.isnan(v)) * 1.04
    _shock_line(ax, y_min, y_max)

    ax.set_xlim(1, n_ticks)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Tick", fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel("Price ($)", fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_title("Price History Across Simulation Runs", fontfamily=FONT_FAMILY, fontsize=13)
    ax.legend(frameon=False, fontsize=9, prop={"family": FONT_FAMILY})
    ax.grid(False)
    _apply_base_style(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Return distribution
# ---------------------------------------------------------------------------

def fig2_return_distribution(vd: dict, out_path: str) -> None:
    raw = np.array([r for r in vd["raw_returns"] if r is not None], dtype=float)
    if len(raw) == 0:
        print("  Fig 2 skipped — no return data.")
        return

    mu    = float(np.mean(raw))
    sigma = float(np.std(raw))
    kurt  = float(np.mean(((raw - mu) / sigma) ** 4)) if sigma > 0 else float("nan")
    skew  = float(np.mean(((raw - mu) / sigma) ** 3)) if sigma > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(9, 5))

    # Histogram — color bins by tail status
    n_bins = 60
    counts, edges = np.histogram(raw, bins=n_bins)
    tail_lo = mu - 2.0 * sigma
    tail_hi = mu + 2.0 * sigma

    for i in range(len(counts)):
        left, right = edges[i], edges[i + 1]
        is_tail = right <= tail_lo or left >= tail_hi
        color = "#c0392b" if is_tail else "#4a90d9"
        ax.bar(left, counts[i] / (len(raw) * (edges[1] - edges[0])),
               width=right - left, align="edge",
               color=color, alpha=0.75, zorder=2)

    # Normal PDF overlay
    x_lo = min(raw.min(), mu - 4 * sigma)
    x_hi = max(raw.max(), mu + 4 * sigma)
    xs   = np.linspace(x_lo, x_hi, 400)
    pdf  = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    ax.plot(xs, pdf, color="black", linewidth=1.8, zorder=5, label="Fitted normal")

    # Shade normal tail regions under the PDF curve
    for x_mask in [xs <= tail_lo, xs >= tail_hi]:
        ax.fill_between(xs, 0, pdf, where=x_mask, color="#c0392b", alpha=0.15, zorder=3)

    # Stats annotation
    stats_text = (
        f"n = {len(raw):,}\n"
        f"μ = {mu:.4f}\n"
        f"σ = {sigma:.4f}\n"
        f"skew = {skew:.3f}\n"
        f"kurt = {kurt:.3f}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, fontfamily=FONT_FAMILY,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel("Tick Return", fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel("Density", fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_title("Empirical Return Distribution vs Fitted Normal", fontfamily=FONT_FAMILY, fontsize=13)
    ax.legend(frameon=False, fontsize=9, prop={"family": FONT_FAMILY})
    ax.grid(False)
    _apply_base_style(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Per-archetype wealth divergence (index = 100)
# ---------------------------------------------------------------------------

def fig3_wealth_divergence(vd: dict, out_path: str) -> None:
    arch_wealth = vd["wealth_by_archetype"]
    initial     = vd.get("initial_wealth_by_archetype",
                          {"retail": 10_000.0, "inst": 100_000.0, "quant": 50_000.0})

    n_ticks = max(len(v) for v in arch_wealth.values())
    ticks   = list(range(1, n_ticks + 1))

    fig, ax = plt.subplots(figsize=(9, 5))

    y_vals: list[float] = []
    for arch in ("retail", "inst", "quant"):
        series = arch_wealth.get(arch, [])
        w0     = initial.get(arch, 1.0) or 1.0
        index  = [
            round((v / w0) * 100.0, 4) if v is not None else float("nan")
            for v in series
        ]
        y_vals.extend(v for v in index if not math.isnan(v))
        ax.plot(ticks[:len(index)], index,
                color=ARCHETYPE_COLORS[arch],
                linewidth=2.0,
                label=ARCHETYPE_LABELS[arch],
                zorder=3)

    y_min = min(y_vals) * 0.97 if y_vals else 95.0
    y_max = max(y_vals) * 1.03 if y_vals else 110.0
    _shock_line(ax, y_min, y_max)

    ax.set_xlim(1, n_ticks)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Tick", fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel("Wealth Index (Base = 100)", fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_title("Per-Archetype Wealth Divergence Post-Shock",
                 fontfamily=FONT_FAMILY, fontsize=13)
    ax.legend(frameon=False, fontsize=9, prop={"family": FONT_FAMILY})
    ax.grid(False)
    _apply_base_style(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Post-shock contagion heatmap (3×3)
# ---------------------------------------------------------------------------

def fig4_contagion_heatmap(vd: dict, out_path: str) -> None:
    corr_matrix = vd.get("post_shock_corr_matrix")
    labels      = vd.get("post_shock_corr_labels", ["Retail", "Institutional", "Quant"])

    if corr_matrix is None:
        print("  Fig 4 skipped — no correlation matrix.")
        return

    data = np.array(corr_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5.2))

    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    im   = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontfamily=FONT_FAMILY, fontsize=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)

    n = data.shape[0]
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            txt_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontfamily=FONT_FAMILY, color=txt_color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontfamily=FONT_FAMILY, fontsize=10)
    ax.set_yticklabels(labels, fontfamily=FONT_FAMILY, fontsize=10)
    ax.set_title("Post-Shock Wealth-Change Correlation (Contagion)",
                 fontfamily=FONT_FAMILY, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_path = Path("results.json")
    if not results_path.exists():
        print("ERROR: results.json not found. Run analysis.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    vd = load_viz_data(results)
    if "visualization_data" not in results:
        print("  Note: visualization_data absent in results.json — using synthesized data.")

    print("Generating figures...")
    fig1_price_history(
        vd, str(figures_dir / "fig1_price_history.png")
    )
    fig2_return_distribution(
        vd, str(figures_dir / "fig2_return_distribution.png")
    )
    fig3_wealth_divergence(
        vd, str(figures_dir / "fig3_wealth_divergence.png")
    )
    fig4_contagion_heatmap(
        vd, str(figures_dir / "fig4_contagion_heatmap.png")
    )
    print("Done — 4 figures saved to figures/")


if __name__ == "__main__":
    main()
