import os, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.framealpha": 0.9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "lines.linewidth": 1.8, "lines.markersize": 5.5,
})

NAVY = "#1B2E6E";
BLUE = "#4472C4";
LBLUE = "#5B9BD5"
RED = "#C00000";
GRAY = "#888888"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(FIGS, f"{name}.{ext}")
        fig.savefig(p)
        print(f"  saved → {p}")


# ── Figure 1: Fitness vs Deviation Rate ──────────────────────────────
def fig_fitness():
    df = pd.read_csv(os.path.join(RESULTS, "experiment1.csv"))
    rates = [0, 20, 40]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    fig.canvas.manager.set_window_title("Fig 1 — Fitness vs Deviation")

    ax.plot(rates, df["f"], "o-", color=NAVY, label=r"$f$  (global)")
    ax.plot(rates, df["f_order"], "s--", color=BLUE, label=r"$f_{\tau=\mathrm{order}}$")
    ax.plot(rates, df["f_item"], "^:", color=LBLUE, label=r"$f_{\tau=\mathrm{item}}$")

    for xv, yv in zip(rates, df["f"]):
        ax.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(0, 7),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color=NAVY, fontweight="bold")

    ax.set_xlabel("Deviation rate (%)")
    ax.set_ylabel("Fitness")
    ax.set_title("OC-TBR Fitness vs Deviation Rate")
    ax.set_xticks(rates)
    ax.set_xticklabels(["0 %\n(Log A)", "20 %\n(Log B)", "40 %\n(Log C)"])
    ax.set_ylim(0.70, 1.10)
    ax.legend(loc="lower left")

    fig.tight_layout()
    save(fig, "fitness_vs_deviation")
    return fig


# ── Figure 2: Fitness comparison OC-TBR vs Alignments ────────────────
def fig_comparison():
    df = pd.read_csv(os.path.join(RESULTS, "comparison_oc_alignments.csv"))
    rates = [0, 20, 40]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    fig.canvas.manager.set_window_title("Fig 2 — OC-TBR vs Alignments (fitness)")

    ax.plot(rates, df["octbr_fitness"], "o-", color=NAVY,
            label="OC-TBR (ours)")
    ax.plot(rates, df["oc_align_fitness"], "s--", color=RED,
            label="OC Alignments [2]")

    # annotate both lines
    for xv, yv in zip(rates, df["octbr_fitness"]):
        ax.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(6, 2),
                    textcoords="offset points",
                    fontsize=7, color=NAVY, fontweight="bold")
    for xv, yv in zip(rates, df["oc_align_fitness"]):
        ax.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(6, -10),
                    textcoords="offset points",
                    fontsize=7, color=RED, fontweight="bold")

    ax.set_xlabel("Deviation rate (%)")
    ax.set_ylabel("Fitness")
    ax.set_title("OC-TBR vs OC Alignments — Fitness")
    ax.set_xticks(rates)
    ax.set_xticklabels(["0 %\n(Log A)", "20 %\n(Log B)", "40 %\n(Log C)"])
    ax.set_ylim(0.50, 1.10)
    ax.legend(loc="lower left")

    for xv, y1, y2 in zip(rates,
                          df["octbr_fitness"],
                          df["oc_align_fitness"]):
        if abs(y1 - y2) > 0.01:
            mid = (y1 + y2) / 2
            ax.annotate(
                f"Δ={y1 - y2:.2f}",
                xy=(xv, mid),
                xytext=(18, 0), textcoords="offset points",
                fontsize=6.5, color=GRAY,
                arrowprops=dict(arrowstyle="-", color=GRAY,
                                lw=0.8, linestyle="dotted"),
            )

    fig.tight_layout()
    save(fig, "comparison_alignments")
    return fig


# ── Figure 3: Runtime — OC-TBR vs Alignments ─────────────────────────
def fig_runtime():
    df = pd.read_csv(os.path.join(RESULTS, "comparison_oc_alignments.csv"))
    tbr_s = df["octbr_time_s"].tolist()
    aln_s = df["oc_align_time_s"].tolist()
    labels = ["Log A\n(0 %)", "Log B\n(20 %)", "Log C\n(40 %)"]
    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    fig.canvas.manager.set_window_title("Fig 3 — Runtime comparison")

    b1 = ax.bar(x - w / 2, tbr_s, w, label="OC-TBR (ours)",
                color=NAVY, zorder=3)
    b2 = ax.bar(x + w / 2, aln_s, w, label="OC Alignments [2]",
                color=RED, alpha=0.85, zorder=3)

    ax.set_yscale("log")
    ax.set_ylabel("Runtime (s)  —  log scale")
    ax.set_title("Runtime: OC-TBR vs OC Alignments")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")

    for bar, val in zip(b1, [t * 1000 for t in tbr_s]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 2.8,
                f"{val:.1f} ms",
                ha="center", va="bottom", fontsize=7,
                color=NAVY, fontweight="bold")
    for bar, val in zip(b2, aln_s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 2.8,
                f"{val:.1f} s",
                ha="center", va="bottom", fontsize=7,
                color=RED, fontweight="bold")

    fig.tight_layout()
    save(fig, "runtime_comparison")
    return fig


# ── Figure 4: Scalability ─────────────────────────────────────────────
def fig_scalability():
    df = pd.read_csv(os.path.join(RESULTS, "scalability.csv"))
    events = df["n_events"].tolist()
    times_ms = (df["time_s"] * 1000).tolist()

    coeffs = np.polyfit(events, times_ms, 1)
    x_fit = np.linspace(min(events), max(events), 300)
    y_fit = np.polyval(coeffs, x_fit)

    yp = np.polyval(coeffs, events)
    ss_res = np.sum((np.array(times_ms) - yp) ** 2)
    ss_tot = np.sum((np.array(times_ms) - np.mean(times_ms)) ** 2)
    rv = 1.0 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    fig.canvas.manager.set_window_title("Fig 4 — Scalability")

    ax.plot(events, times_ms, "o", color=NAVY, zorder=4, label="Measured")
    ax.plot(x_fit, y_fit, "--", color=GRAY, linewidth=1.2, zorder=3,
            label=f"Linear fit  ($R^2 = {rv:.4f}$)")

    ax.set_xlabel(r"Number of events  $|\mathcal{L}|$")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("OC-TBR Runtime Scalability")
    ax.legend(loc="upper left")

    fig.tight_layout()
    save(fig, "scalability")
    return fig


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...\n")

    print("Figure 1: fitness_vs_deviation");
    f1 = fig_fitness()
    print("Figure 2: comparison_alignments");
    f2 = fig_comparison()
    print("Figure 3: runtime_comparison");
    f3 = fig_runtime()
    print("Figure 4: scalability");
    f4 = fig_scalability()

    print(f"\nAll saved to: {FIGS}/")

    plt.show()
