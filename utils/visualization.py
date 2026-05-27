"""
Visualization utilities reproducing Figures 1–4 from the paper.

Figure 1 : RMSE and RSSI under different training-data proportions (two panels)
Figure 2 : (a) Temporal evolution of Kendall's τ with 95% CI;
           (b) Cumulative Mass Balance Error (MBE) over prediction horizon
Figure 3 : Box plots of Ranking Instability Index (RII = 1 − τ) across 20 runs
Figure 4 : Kendall's τ over 120-month horizon under six-strategy expansion
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "Plain LSTM":             "#2C7BB6",
    "Feature-augmented LSTM": "#ABD9E9",
    "Standard PINN":          "#FDAE61",
    "PhyLSTM (proposed)":     "#D7191C",
}
MODEL_MARKERS = {
    "Plain LSTM":             "o",
    "Feature-augmented LSTM": "s",
    "Standard PINN":          "^",
    "PhyLSTM (proposed)":     "D",
}
MODEL_KEYS = list(MODEL_COLORS.keys())

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(SAVE_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — RMSE and RSSI vs training-data proportion  (two panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig1(data_ratios: Sequence[float],
              rmse_dict:   dict,
              rssi_dict:   dict,
              rmse_ci:     dict | None = None,
              rssi_ci:     dict | None = None,
              filename: str = "fig1_rmse_rssi_vs_data.png"):
    """
    Fig. 1  Comparison of (a) RMSE and (b) RSSI under different
    training-data proportions (mean ± 95% CI).

    Parameters
    ----------
    data_ratios : sequence of floats, e.g. [0.05, 0.10, 0.20, 0.50, 1.00]
    rmse_dict   : {model_name: list of mean RMSE at each ratio}
    rssi_dict   : {model_name: list of mean RSSI at each ratio}
    rmse_ci     : {model_name: list of 95% CI half-widths for RMSE}
    rssi_ci     : {model_name: list of 95% CI half-widths for RSSI}
    """
    ratios_pct = [r * 100 for r in data_ratios]
    fig, axes  = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric_dict, ci_dict, ylabel, panel in zip(
            axes,
            [rmse_dict, rssi_dict],
            [rmse_ci, rssi_ci] if rmse_ci else [None, None],
            ["RMSE (mg L\u207b\u00b9)", "RSSI (Kendall\u2019s \u03c4)"],
            ["(a)", "(b)"]):

        for name in MODEL_KEYS:
            if name not in metric_dict:
                continue
            vals = np.array(metric_dict[name])
            c    = MODEL_COLORS[name]
            m    = MODEL_MARKERS[name]
            ax.plot(ratios_pct, vals, color=c, label=name,
                    marker=m, lw=2, ms=6)
            if ci_dict and name in ci_dict:
                ci = np.array(ci_dict[name])
                ax.fill_between(ratios_pct, vals - ci, vals + ci,
                                color=c, alpha=0.15)

        ax.set_xlabel("Training data proportion (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Fig. 1{panel}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — (a) Temporal Kendall's τ with 95% CI
#             (b) Cumulative MBE over 120 months
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig2(rssi_mean:          dict,
              rssi_ci:            dict,
              mbe_mean:           dict,
              collapse_threshold: float = 0.15,
              mbe_threshold:      float = 0.030,
              mbe_cross_month:    int   = 60,
              filename: str = "fig2_kendall_tau_and_mbe.png"):
    """
    Fig. 2  Temporal dynamics of ranking coherence and physical consistency.

    Panel (a): temporal evolution of Kendall's τ with 95% bootstrap CI
               (1,000 draws, 20 independent runs); dashed line at τ = 0.15.
    Panel (b): cumulative mass-balance error (MBE, mg L⁻¹); horizontal dashed
               line at MBE = 0.030 mg L⁻¹; vertical dashed line at t = 60
               (when Plain LSTM first exceeds the threshold).

    Parameters
    ----------
    rssi_mean        : {model_name: (T,) mean Kendall's τ series}
    rssi_ci          : {model_name: (T,) 95% CI half-width}
    mbe_mean         : {model_name: (T-1,) mean cumulative MBE series}
    collapse_threshold: horizontal threshold in panel (a) [default 0.15]
    mbe_threshold    : critical MBE threshold in panel (b) [default 0.030]
    mbe_cross_month  : month when Plain LSTM first exceeds mbe_threshold
                       [default 60]; drawn as a vertical dashed line
    """
    T_tau = next(iter(rssi_mean.values())).shape[0]
    T_mbe = next(iter(mbe_mean.values())).shape[0]
    time_tau = np.arange(1, T_tau + 1)
    time_mbe = np.arange(1, T_mbe + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel (a): Kendall's τ ────────────────────────────────────────────
    ax = axes[0]
    for name in MODEL_KEYS:
        if name not in rssi_mean:
            continue
        mu = rssi_mean[name]
        ci = rssi_ci[name]
        c  = MODEL_COLORS[name]
        ax.plot(time_tau, mu, color=c, label=name, lw=2,
                marker=MODEL_MARKERS[name], markevery=10, ms=5)
        ax.fill_between(time_tau, mu - ci, mu + ci, color=c, alpha=0.15)

    ax.axhline(collapse_threshold, ls="--", color="gray", lw=1.2,
               label=f"Instability threshold \u03c4\u00a0=\u00a0{collapse_threshold}")
    ax.set_xlabel("Prediction horizon (months)", fontsize=12)
    ax.set_ylabel("Kendall\u2019s \u03c4 (RSSI)", fontsize=12)
    ax.set_title("Fig. 2(a)  Temporal evolution of Kendall\u2019s \u03c4 "
                 "with 95% CI", fontsize=11)
    ax.set_xlim(1, T_tau)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel (b): Cumulative MBE ─────────────────────────────────────────
    ax = axes[1]
    for name in MODEL_KEYS:
        if name not in mbe_mean:
            continue
        ax.plot(time_mbe, mbe_mean[name], color=MODEL_COLORS[name],
                label=name, lw=2,
                marker=MODEL_MARKERS[name], markevery=10, ms=5)

    ax.axhline(mbe_threshold, ls="--", color="gray", lw=1.2,
               label=f"Critical threshold {mbe_threshold}\u00a0mg\u00a0L\u207b\u00b9")
    ax.axvline(mbe_cross_month, ls="--", color="gray", lw=1.0,
               label=f"t\u00a0=\u00a0{mbe_cross_month} (Plain LSTM exceeds threshold)")
    ax.set_xlabel("Prediction horizon (months)", fontsize=12)
    ax.set_ylabel("Cumulative MBE (mg\u00a0L\u207b\u00b9)", fontsize=12)
    ax.set_title("Fig. 2(b)  Cumulative mass-balance error (MBE)", fontsize=11)
    ax.set_xlim(1, T_mbe)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Box plots of Ranking Instability Index (RII = 1 − τ)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig3(rii_runs: dict,
              sig_pairs: list | None = None,
              filename: str = "fig3_rii_boxplots.png"):
    """
    Fig. 3  Box plots of Ranking Instability Index (RII = 1 − τ) across
    20 independent runs for four models.

    Parameters
    ----------
    rii_runs  : {model_name: (n_runs,) array of per-run mean RII values}
                RII = 1 − τ; caller must pass 1 − RSSI, NOT raw RSSI.
    sig_pairs : list of (label_a, label_b, p_value, bracket_style) tuples
                for significance brackets, e.g.
                [("Feature-augmented LSTM", "Plain LSTM", 0.05,  "dashed"),
                 ("PhyLSTM (proposed)",     "Plain LSTM", 0.001, "solid")]
                bracket_style: "dashed" or "solid"
    """
    # Order left-to-right: decreasing instability
    ordered = ["Plain LSTM", "Feature-augmented LSTM",
               "Standard PINN", "PhyLSTM (proposed)"]
    names  = [n for n in ordered if n in rii_runs]
    data   = [rii_runs[n] for n in names]
    colors = [MODEL_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    showfliers=True, widths=0.5,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker="o", ms=5, linestyle="none"))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Mean diamonds
    for i, d in enumerate(data):
        ax.scatter(i + 1, np.mean(d), marker="D", color="black",
                   zorder=5, s=40, label="\u25c6 Mean" if i == 0 else None)

    # Significance brackets
    if sig_pairs:
        y_top = max(np.max(d) for d in data) * 1.05
        for label_a, label_b, pval, style in sig_pairs:
            if label_a not in names or label_b not in names:
                continue
            xa = names.index(label_a) + 1
            xb = names.index(label_b) + 1
            y_top *= 1.08
            ls = "--" if style == "dashed" else "-"
            if pval < 0.001:
                p_str = "p < 0.001"
            elif pval < 0.05:
                p_str = "p < 0.05"
            else:
                p_str = f"p = {pval:.3f}"
            ax.plot([xa, xa, xb, xb], [y_top * 0.97, y_top, y_top, y_top * 0.97],
                    color="black", lw=1.2, ls=ls)
            ax.text((xa + xb) / 2, y_top * 1.01, p_str,
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Ranking Instability Index (RII\u00a0=\u00a01\u00a0\u2212\u00a0\u03c4)",
                  fontsize=12)
    ax.set_title("Fig. 3  RII box plots across 20 independent runs\n"
                 "(IQR\u00a0=\u00a0box, 1.5\u00d7IQR\u00a0=\u00a0whiskers, "
                 "\u25c6\u00a0=\u00a0mean, \u25e6\u00a0=\u00a0outlier)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Kendall's τ over 120 months, six-strategy expansion
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig4(rssi_mean:  dict,
              rssi_ci:    dict,
              threshold:  float = 0.70,
              filename: str = "fig4_six_strategy_kendall_tau.png"):
    """
    Fig. 4  Ranking stability under the expanded six-strategy decision scenario.

    Kendall's τ over 120 months with 95% CI bands; dashed threshold at τ = 0.70.

    Parameters
    ----------
    rssi_mean : {model_name: (T,) mean Kendall's τ time series}
    rssi_ci   : {model_name: (T,) 95% CI half-width}
    threshold : stability threshold (default 0.70 for six-strategy scenario)
    """
    T    = next(iter(rssi_mean.values())).shape[0]
    time = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for name in MODEL_KEYS:
        if name not in rssi_mean:
            continue
        mu = rssi_mean[name]
        ci = rssi_ci[name]
        c  = MODEL_COLORS[name]
        ax.plot(time, mu, color=c, label=name, lw=2,
                marker=MODEL_MARKERS[name], markevery=10, ms=5)
        ax.fill_between(time, mu - ci, mu + ci, color=c, alpha=0.15)

    ax.axhline(threshold, ls="--", color="gray", lw=1.2,
               label=f"Stability threshold \u03c4\u00a0=\u00a0{threshold}")
    ax.set_xlabel("Prediction horizon (months)", fontsize=12)
    ax.set_ylabel("Kendall\u2019s \u03c4", fontsize=12)
    ax.set_title("Fig. 4  Kendall\u2019s \u03c4 over 120 months \u2014 "
                 "six-strategy expansion (shaded: 95% CI)", fontsize=11)
    ax.set_xlim(1, T)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Table printer helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_table3(results: dict):
    """Print Table 3 / Table S3 to stdout."""
    header = (f"{'Model':<28} "
              f"{'10% RMSE':>12} {'10% RSSI':>10} "
              f"{'50% RMSE':>12} {'50% RSSI':>10} "
              f"{'100% RMSE':>12} {'100% RSSI':>10}")
    print("\n" + "=" * len(header))
    print("Table 3 / S3  Performance under different training data proportions")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, vals in results.items():
        row = f"{model_name:<28} "
        for k in ["10%_rmse", "10%_rssi", "50%_rmse", "50%_rssi",
                  "100%_rmse", "100%_rssi"]:
            if k in vals:
                mu, sd = vals[k]
                row += f"  {mu:.3f}±{sd:.3f}"
            else:
                row += f"  {'N/A':>12}"
        print(row)
    print("=" * len(header) + "\n")
