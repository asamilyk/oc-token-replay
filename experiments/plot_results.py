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
    df = df[df["log"].isin(["A", "B", "C"])].reset_index(drop=True)
    rates = [0, 20, 40]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    fig.canvas.manager.set_window_title("Fig 1 — Fitness vs Deviation")

    ax.plot(rates, df["f"], "o-", color=NAVY, label=r"$f$  (global)")
    ax.plot(rates, df["f_order"], "s--", color=BLUE, label=r"$f_{\tau=\mathrm{order}}$")
    ax.plot(rates, df["f_item"], "^:", color=LBLUE, label=r"$f_{\tau=\mathrm{item}}$")


    offsets = [7, 14, 7]
    for xv, yv, off in zip(rates, df["f"], offsets):
        ax.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(0, off),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color=NAVY, fontweight="bold")

    ax.set_xlabel("Deviation rate (%)")
    ax.set_ylabel("Fitness")
    ax.set_title("OC-TBR Fitness vs Deviation Rate")
    ax.set_xticks(rates)
    ax.set_xticklabels(["0 %\n(Log A)", "20 %\n(Log B)", "40 %\n(Log C)"])
    ax.set_ylim(0.70, 1.12)
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

    ax.plot(rates, df["octbr_fitness"], "o-", color=NAVY, label="OC-TBR (ours)")
    ax.plot(rates, df["oc_align_fitness"], "s--", color=RED, label="OC Alignments [2]")


    for xv, yv in zip(rates, df["octbr_fitness"]):
        ax.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(-18, 6),
                    textcoords="offset points",
                    fontsize=7, color=NAVY, fontweight="bold")


    align_offsets = [-12, -14, -12]
    for xv, yv, off in zip(rates, df["oc_align_fitness"], align_offsets):
        ax.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(-18, off),
                    textcoords="offset points",
                    fontsize=7, color=RED, fontweight="bold")

    ax.set_xlabel("Deviation rate (%)")
    ax.set_ylabel("Fitness")
    ax.set_title("OC-TBR vs OC Alignments — Fitness")
    ax.set_xticks(rates)
    ax.set_xticklabels(["0 %\n(Log A)", "20 %\n(Log B)", "40 %\n(Log C)"])
    ax.set_ylim(0.50, 1.12)
    ax.legend(loc="lower left")

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

    ax.legend(loc="lower right", fontsize=7.5)

    for bar, val in zip(b1, [t * 1000 for t in tbr_s]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.8,
                f"{val:.1f} ms",
                ha="center", va="bottom", fontsize=7,
                color="white", fontweight="bold")

    aln_labels_y = [v * 1.3 for v in aln_s]
    for bar, val, y in zip(b2, aln_s, aln_labels_y):
        ax.text(bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:.1f} s",
                ha="center", va="bottom", fontsize=7,
                color=RED, fontweight="bold")

    ax.set_ylim(bottom=ax.get_ylim()[0],
                top=max(aln_s) * 8)

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


# ── Figure 5: Auto vs Manual OC-PN ───────────────────────────────────
def fig_auto_vs_manual():
    objects = ["i1\n(Item)", "i2\n(Item)", "p1\n(Package)"]
    f_auto = [0.8333, 0.4500, 1.0000]
    f_manual = [0.8333, 0.8750, 1.0000]

    x = np.arange(len(objects))
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    fig.canvas.manager.set_window_title("Fig 5 — Auto vs Manual")

    b1 = ax.bar(x - width / 2, f_auto, width,
                label="Auto (pm4py)", color=NAVY, alpha=0.85)
    b2 = ax.bar(x + width / 2, f_manual, width,
                label="Manual OC-PN", color=BLUE, alpha=0.85)

    for bar, val in zip(b1, f_auto):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", fontsize=7.5,
                color=NAVY, fontweight="bold")
    for bar, val in zip(b2, f_manual):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", fontsize=7.5,
                color=BLUE, fontweight="bold")

    arrow_x = x[1] - width / 2 + 0.2
    ax.annotate("",
                xy=(arrow_x, f_manual[1]),
                xytext=(arrow_x, f_auto[1]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(arrow_x - 0.08,
            (f_auto[1] + f_manual[1]) / 2,
            f"Δ={f_manual[1] - f_auto[1]:.3f}",
            fontsize=7.5, color=RED, fontweight="bold",
            va="center", ha="right")

    ax.set_ylabel("Per-object fitness $f_o$")
    ax.set_title("Paper Example: Auto vs Manual OC-PN")
    ax.set_xticks(x)
    ax.set_xticklabels(objects)
    ax.set_ylim(0.3, 1.18)
    ax.set_xlim(-0.7, 2.6)
    ax.legend(loc="lower right", fontsize=7.5)

    fig.tight_layout()
    save(fig, "auto_vs_manual")
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
    print("Figure 5: auto_vs_manual")
    f5 = fig_auto_vs_manual()

    print(f"\nAll saved to: {FIGS}/")

    plt.show()
