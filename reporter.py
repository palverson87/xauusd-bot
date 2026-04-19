"""
Weekly PDF report — saved to reports/<date>.pdf
Uses matplotlib with a dark theme matching the dashboard.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import db

log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"

# ── Colours ────────────────────────────────────────────────────────────────────
BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#30363d"
TEXT  = "#e6edf3"
DIM   = "#8b949e"
GREEN = "#3fb950"
RED   = "#f85149"
AMBER = "#d29922"
BLUE  = "#58a6ff"


def _style(ax, title, xlabel="", ylabel="Win Rate (%)"):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, fontfamily="monospace", pad=8)
    ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    ax.set_ylabel(ylabel, color=DIM, fontsize=8)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)


def _bar(ax, labels, values, totals, title, xlabel=""):
    if not labels:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, color=DIM, fontsize=10)
        _style(ax, title, xlabel)
        return
    colors = [GREEN if v >= 55 else RED if v < 45 else AMBER for v in values]
    x = range(len(labels))
    ax.bar(x, values, color=colors, zorder=3, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    for i, (v, n) in enumerate(zip(values, totals)):
        ax.text(i, v + 1.5, f"n={n}", ha="center", color=DIM, fontsize=8)
    ax.axhline(50, color=DIM, linestyle="--", linewidth=0.8, zorder=4)
    ax.set_ylim(0, 115)
    _style(ax, title, xlabel)


def generate_weekly_report() -> str:
    REPORTS_DIR.mkdir(exist_ok=True)
    stats   = db.get_performance_stats()
    weights = db.load_weights()
    now     = datetime.now(timezone.utc)
    fname   = REPORTS_DIR / f"xauusd_{now.strftime('%Y-%m-%d')}.pdf"

    with PdfPages(str(fname)) as pdf:

        # ── Page 1: Win-rate overview ──────────────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor(BG)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

        # Title row
        ax0 = fig.add_subplot(gs[0, :])
        ax0.axis("off")
        ax0.set_facecolor(BG)
        ax0.text(0.5, 0.70, "XAU/USD  ·  Weekly Performance Report",
                 transform=ax0.transAxes, color=TEXT, fontsize=18,
                 ha="center", fontweight="bold", fontfamily="monospace")
        ax0.text(0.5, 0.20, f"Week ending {now.strftime('%B %d, %Y')}  ·  UTC",
                 transform=ax0.transAxes, color=DIM, fontsize=11,
                 ha="center", fontfamily="monospace")

        # By session
        ax1 = fig.add_subplot(gs[1, 0])
        sess = stats["by_session"]
        _bar(ax1,
             [r["session"] for r in sess],
             [r["wins"] / r["total"] * 100 if r["total"] else 0 for r in sess],
             [r["total"] for r in sess],
             "Win Rate by Session", "Session")

        # By confluence score
        ax2 = fig.add_subplot(gs[1, 1])
        sc = stats["by_score"]
        _bar(ax2,
             [str(r["score"]) for r in sc],
             [r["wins"] / r["total"] * 100 if r["total"] else 0 for r in sc],
             [r["total"] for r in sc],
             "Win Rate by Confluence Score", "Score / 5")

        # By indicator
        ax3 = fig.add_subplot(gs[1, 2])
        ind = stats["indicator_acc"]
        _bar(ax3,
             [n.replace(" ", "\n") for n in ind],
             [d["pct"] if d["pct"] is not None else 0 for d in ind.values()],
             [d["total"] for d in ind.values()],
             "Win Rate by Indicator", "Indicator")

        pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        # ── Page 2: Equity curve + weights ────────────────────────────────
        fig2 = plt.figure(figsize=(11.69, 8.27))
        fig2.patch.set_facecolor(BG)
        gs2  = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

        # Equity curve
        ax_eq = fig2.add_subplot(gs2[0, 0])
        eq_data = stats["equity"]
        if eq_data:
            equity = 10_000.0
            curve  = [equity]
            for row in eq_data:
                equity *= 1 + (row["pnl_pct"] or 0) / 100
                curve.append(equity)
            final_ret = (curve[-1] / 10_000 - 1) * 100
            col = GREEN if curve[-1] >= 10_000 else RED
            xs  = range(len(curve))
            ax_eq.plot(xs, curve, color=col, linewidth=2, zorder=3)
            ax_eq.fill_between(xs, 10_000, curve, alpha=0.15, color=col, zorder=2)
            ax_eq.axhline(10_000, color=DIM, linestyle="--", linewidth=0.8, zorder=4)
            _style(ax_eq, f"Equity Curve  ({final_ret:+.1f}%)",
                   "Signal #", "Portfolio Value ($)")
        else:
            ax_eq.text(0.5, 0.5, "No completed signals yet",
                       ha="center", va="center", transform=ax_eq.transAxes,
                       color=DIM, fontsize=10)
            _style(ax_eq, "Equity Curve", "Signal #", "Portfolio Value ($)")

        # Adaptive weights
        ax_w  = fig2.add_subplot(gs2[0, 1])
        names = list(weights.keys())
        vals  = list(weights.values())
        wcols = [GREEN if v >= 1.0 else RED for v in vals]
        bars  = ax_w.barh(names, vals, color=wcols, zorder=3, height=0.5)
        ax_w.axvline(1.0, color=DIM, linestyle="--", linewidth=0.8, zorder=4)
        for bar, v in zip(bars, vals):
            ax_w.text(v + 0.03, bar.get_y() + bar.get_height() / 2,
                      f"{v:.2f}", va="center", color=TEXT, fontsize=9)
        ax_w.set_xlim(0, 2.3)
        _style(ax_w, "Adaptive Indicator Weights", "Weight", "")
        ax_w.set_ylabel("", color=DIM)
        ax_w.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
        ax_w.grid(axis="y", visible=False)

        d = pdf.infodict()
        d["Title"]   = "XAU/USD Weekly Performance Report"
        d["Subject"] = f"Week ending {now.strftime('%Y-%m-%d')}"

        pdf.savefig(fig2, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)

    log.info("Report written: %s", fname)
    return str(fname)
