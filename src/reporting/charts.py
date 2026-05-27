"""
Charts
───────
All matplotlib chart generators matching the target report style.

Charts produced:
  equity_curve.png     : Cumulative return + drawdown subplot
  rolling_sharpe.png   : Rolling 12-month annualised Sharpe
  monthly_heatmap.png  : Calendar heatmap of monthly returns
  report.png           : Full composite report (all three + summary table)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.reporting.analytics import PerformanceMetrics, rolling_sharpe

# ── Colour palette ────────────────────────────────────────────────────────────
# Maps commodity key → line colour (matches reference images)
PALETTE: Dict[str, str] = {
    "crude_oil":   "#1f77b4",   # blue
    "natural_gas": "#ff7f0e",   # orange
    "gold":        "#2ca02c",   # green
    "silver":      "#d62728",   # red
    "copper":      "#9467bd",   # purple
    "wheat":       "#8c564b",   # brown
    "corn":        "#e377c2",   # pink
    "soybeans":    "#bcbd22",   # yellow-green
    "combined":    "#000000",   # black (bold)
}

DISPLAY_NAMES: Dict[str, str] = {
    "crude_oil":   "Crude Oil",
    "natural_gas": "Natural Gas",
    "gold":        "Gold",
    "silver":      "Silver",
    "copper":      "Copper",
    "wheat":       "Wheat",
    "corn":        "Corn",
    "soybeans":    "Soybeans",
    "combined":    "COMBINED",
}

FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def _label(key: str) -> str:
    return DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def _color(key: str) -> str:
    return PALETTE.get(key, "#333333")


# ── Equity Curve ──────────────────────────────────────────────────────────────

def plot_equity_curve(
    daily_pnl: pd.DataFrame,
    title: str = "Equity Curve",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Two-panel chart:
      Top    : Cumulative return from 1.0 base
      Bottom : Drawdown area (red fill)
    """
    equity = (1 + daily_pnl).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax()

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

    cols = [c for c in daily_pnl.columns if c != "combined"] + ["combined"]

    for col in cols:
        if col not in equity.columns:
            continue
        lw = 2.5 if col == "combined" else 1.2
        ls = "-" if col == "combined" else "--"
        ax_eq.plot(equity.index, equity[col],
                   color=_color(col), linewidth=lw, linestyle=ls,
                   label=_label(col), zorder=3 if col == "combined" else 2)

    ax_eq.set_ylabel("Cumulative Return", fontsize=10)
    ax_eq.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax_eq.axhline(1.0, color="grey", linewidth=0.7, linestyle=":")
    ax_eq.legend(loc="upper left", fontsize=8, framealpha=0.85)
    # Show as percentage gain from start: 1.25 → "+25%"
    ax_eq.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{(x - 1) * 100:+.0f}%")
    )
    ax_eq.grid(True)

    # Drawdown (combined only)
    if "combined" in drawdown.columns:
        dd = drawdown["combined"]
        ax_dd.fill_between(dd.index, dd, 0, color="#e74c3c", alpha=0.55, label="Drawdown")
        ax_dd.plot(dd.index, dd, color="#e74c3c", linewidth=0.7)
    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax_dd.grid(True)
    ax_dd.set_xlabel("Date", fontsize=9)
    plt.setp(ax_dd.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    plt.setp(ax_eq.get_xticklabels(), visible=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Rolling Sharpe ────────────────────────────────────────────────────────────

def plot_rolling_sharpe(
    daily_pnl: pd.DataFrame,
    window_days: int = 252,
    title: str = "Rolling 12-Month Sharpe Ratio",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    roll_sh = rolling_sharpe(daily_pnl, window_days)

    fig, ax = plt.subplots(figsize=(14, 5))

    cols = [c for c in daily_pnl.columns if c != "combined"] + ["combined"]
    for col in cols:
        if col not in roll_sh.columns:
            continue
        lw = 2.5 if col == "combined" else 1.2
        ls = "-" if col == "combined" else "--"
        ax.plot(roll_sh.index, roll_sh[col],
                color=_color(col), linewidth=lw, linestyle=ls,
                label=_label(col), zorder=3 if col == "combined" else 2)

    ax.axhline(1.0, color="grey", linewidth=0.9, linestyle=":",
               label="Sharpe = 1")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_ylabel("Sharpe (annualised)", fontsize=10)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Monthly Heatmap ────────────────────────────────────────────────────────────

def plot_monthly_heatmap(
    monthly_returns: pd.DataFrame,
    column: str = "combined",
    title: str = "Combined Monthly Returns Heatmap (Sharpe-Weighted)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Calendar heatmap — rows=years, cols=months (Jan–Dec) + Total column."""
    if column not in monthly_returns.columns:
        column = monthly_returns.columns[0]

    mr = monthly_returns[column].dropna()
    if mr.empty:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

    # Pivot into year × month grid
    df = mr.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    # Total column (cumulative product - 1)
    pivot["Total"] = (1 + pivot.fillna(0)).prod(axis=1) - 1

    n_rows = len(pivot)
    fig_h = max(2.5, 0.55 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.set_aspect("auto")

    vmax = 0.08
    vmin = -0.08

    # Custom red-white-green colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rwg", ["#c0392b", "#ffffff", "#1a7a4a"], N=256
    )

    month_cols = [c for c in pivot.columns if c != "Total"]
    n_cols = len(month_cols)
    col_w = 1.0
    total_w = 1.2

    for r, (year, row) in enumerate(pivot.iterrows()):
        # Month cells
        for c, mcol in enumerate(month_cols):
            val = row.get(mcol, np.nan)
            normed = np.clip((val - vmin) / (vmax - vmin), 0, 1) if not np.isnan(val) else 0.5
            color = cmap(normed)
            rect = mpatches.FancyBboxPatch(
                (c * col_w, r), col_w * 0.95, 0.85,
                boxstyle="round,pad=0.02", facecolor=color,
                edgecolor="white", linewidth=0.5
            )
            ax.add_patch(rect)
            if not np.isnan(val):
                txt_color = "white" if abs(val) > 0.04 else "black"
                ax.text(c * col_w + col_w * 0.475, r + 0.42,
                        f"{val * 100:.1f}%", ha="center", va="center",
                        fontsize=7.5, color=txt_color, fontweight="bold")

        # Total cell
        total_val = row.get("Total", np.nan)
        tx = n_cols * col_w + 0.1
        if not np.isnan(total_val):
            t_color = "#1a7a4a" if total_val >= 0 else "#c0392b"
            rect_t = mpatches.FancyBboxPatch(
                (tx, r), total_w * 0.9, 0.85,
                boxstyle="round,pad=0.02", facecolor=t_color,
                edgecolor="white", linewidth=0.5
            )
            ax.add_patch(rect_t)
            ax.text(tx + total_w * 0.45, r + 0.42,
                    f"{'+' if total_val >= 0 else ''}{total_val * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

        # Year label
        ax.text(-0.6, r + 0.42, str(year), ha="right", va="center",
                fontsize=9, fontweight="bold")

    # Column headers
    for c, mcol in enumerate(month_cols):
        ax.text(c * col_w + col_w * 0.475, n_rows + 0.15, mcol,
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(n_cols * col_w + 0.1 + total_w * 0.45, n_rows + 0.15, "Total",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="#1a7a4a")

    ax.set_xlim(-1, n_cols * col_w + total_w + 0.3)
    ax.set_ylim(-0.3, n_rows + 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin * 100, vmax * 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.set_label("%", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Composite Report ──────────────────────────────────────────────────────────

def plot_report(
    daily_pnl: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    metrics: Dict[str, PerformanceMetrics],
    nav_weights: Dict[str, float],
    strategy_name: str = "Lone Star Strategy",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Full composite A3-style report matching the target layout:
      Row 1: Performance table (left) + Equity curve (right)
      Row 2: Rolling Sharpe
      Row 3: Monthly heatmap
    """
    fig = plt.figure(figsize=(20, 22))
    gs_outer = gridspec.GridSpec(3, 1, figure=fig,
                                 height_ratios=[2.2, 1.5, 1.8],
                                 hspace=0.35)

    # ── Row 1: table + equity curve ───────────────────────────────────────
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[0], width_ratios=[1, 2.2], wspace=0.08
    )
    ax_tbl = fig.add_subplot(gs_top[0])
    ax_eq_top = fig.add_subplot(gs_top[1])

    _draw_summary_table(ax_tbl, metrics, nav_weights, strategy_name, daily_pnl)
    _draw_equity_in_ax(ax_eq_top, daily_pnl, title="Equity Curve — Combined (Backtest → Live)")

    # ── Row 2: rolling Sharpe ─────────────────────────────────────────────
    ax_sharpe = fig.add_subplot(gs_outer[1])
    roll_sh = rolling_sharpe(daily_pnl, 252)
    cols = [c for c in daily_pnl.columns if c != "combined"] + ["combined"]
    for col in cols:
        if col not in roll_sh.columns:
            continue
        ax_sharpe.plot(roll_sh.index, roll_sh[col],
                       color=_color(col),
                       linewidth=2.5 if col == "combined" else 1.2,
                       linestyle="-" if col == "combined" else "--",
                       label=_label(col),
                       zorder=3 if col == "combined" else 2)
    ax_sharpe.axhline(1.0, color="grey", linewidth=0.9, linestyle=":", label="Sharpe = 1")
    ax_sharpe.axhline(0.0, color="black", linewidth=0.5)
    ax_sharpe.set_title("Rolling 12-Month Sharpe Ratio", fontsize=12, fontweight="bold")
    ax_sharpe.set_ylabel("Sharpe (annualised)", fontsize=9)
    ax_sharpe.set_xlabel("Date", fontsize=9)
    ax_sharpe.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_sharpe.grid(True, alpha=0.3, linestyle="--")

    # ── Row 3: monthly heatmap ────────────────────────────────────────────
    ax_hm = fig.add_subplot(gs_outer[2])
    _draw_heatmap_in_ax(ax_hm, monthly_returns, title="Combined Monthly Returns Heatmap (Sharpe-Weighted)")

    fig.suptitle(f"{strategy_name}", fontsize=16, fontweight="bold", y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Internal helpers ───────────────────────────────────────────────────────────

def _draw_summary_table(
    ax: plt.Axes,
    metrics: Dict[str, PerformanceMetrics],
    nav_weights: Dict[str, float],
    strategy_name: str,
    daily_pnl: pd.DataFrame,
) -> None:
    ax.axis("off")

    ordered = [k for k in metrics if k != "combined"]
    ordered_all = ordered + ["combined"]

    y = 0.97
    ax.text(0.0, y, strategy_name, fontsize=11, fontweight="bold",
            va="top", transform=ax.transAxes)
    m = metrics.get("combined") or metrics.get(list(metrics.keys())[0])
    if m:
        ax.text(0.0, y - 0.06, f"{m.start_date} → {m.end_date}",
                fontsize=9, color="#555", va="top", transform=ax.transAxes)

    headers = ["Universe", "Return", "Annual", "Sharpe", "MaxDD"]
    col_x = [0.0, 0.32, 0.48, 0.64, 0.80]
    row_h = 0.062
    y0 = y - 0.14

    # Header row
    for hdr, cx in zip(headers, col_x):
        ax.text(cx, y0, hdr, fontsize=8, fontweight="bold", va="top",
                transform=ax.transAxes, color="#333")
    y0 -= 0.005
    ax.plot([0, 1], [y0, y0], color="#aaa", linewidth=0.8,
            transform=ax.transAxes, clip_on=False)
    y0 -= row_h * 0.6

    for key in ordered_all:
        m_row = metrics.get(key)
        if m_row is None:
            continue
        is_combined = (key == "combined")
        fw = "bold" if is_combined else "normal"
        fcolor = "#1a6b1a" if is_combined else "black"
        bg_color = "#e8f5e9" if is_combined else None

        if is_combined:
            rect = mpatches.FancyBboxPatch(
                (-0.02, y0 - row_h * 0.7), 1.04, row_h * 0.85,
                boxstyle="round,pad=0.01", facecolor=bg_color,
                edgecolor="none", transform=ax.transAxes, zorder=0
            )
            ax.add_patch(rect)

        vals = [_label(key), m_row.fmt_return(), m_row.fmt_annual(),
                m_row.fmt_sharpe(), m_row.fmt_maxdd()]
        for val, cx in zip(vals, col_x):
            ax.text(cx, y0, val, fontsize=8, fontweight=fw, color=fcolor,
                    va="top", transform=ax.transAxes)
        y0 -= row_h

    # NAV Allocation
    y0 -= row_h * 0.5
    ax.plot([0, 1], [y0 + row_h * 0.2, y0 + row_h * 0.2], color="#aaa",
            linewidth=0.5, transform=ax.transAxes, clip_on=False)
    y0 -= row_h * 0.3
    ax.text(0.0, y0, "NAV Allocation", fontsize=8, fontweight="bold",
            va="top", transform=ax.transAxes, color="#333")
    y0 -= row_h * 0.8

    total_w = sum(nav_weights.values()) or 1
    sorted_nav = sorted(nav_weights.items(), key=lambda x: -x[1])
    for commod, w in sorted_nav:
        pct = w / total_w * 100
        ax.text(0.0, y0, _label(commod), fontsize=8, va="top",
                transform=ax.transAxes)
        ax.text(0.70, y0, f"{pct:.1f}%", fontsize=8, va="top",
                transform=ax.transAxes, ha="right")
        y0 -= row_h * 0.9


def _draw_equity_in_ax(ax: plt.Axes, daily_pnl: pd.DataFrame, title: str) -> None:
    """Equity curve with embedded drawdown band at bottom."""
    equity = (1 + daily_pnl).cumprod()
    cols = [c for c in daily_pnl.columns if c != "combined"] + ["combined"]

    for col in cols:
        if col not in equity.columns:
            continue
        ax.plot(equity.index, equity[col],
                color=_color(col),
                linewidth=2.5 if col == "combined" else 1.2,
                linestyle="-" if col == "combined" else "--",
                label=_label(col),
                zorder=3 if col == "combined" else 2)

    ax.set_ylabel("Cumulative Return", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle=":")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{(x - 1) * 100:+.0f}%")
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Date", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


def _draw_heatmap_in_ax(ax: plt.Axes, monthly_returns: pd.DataFrame, title: str) -> None:
    """Inline heatmap for the composite report."""
    col = "combined" if "combined" in monthly_returns.columns else monthly_returns.columns[0]
    mr = monthly_returns[col].dropna()
    if mr.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = mr.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    import seaborn as sns
    pivot_display = pivot.copy()
    pivot_display.columns = month_labels[:len(pivot_display.columns)]

    mask = pivot_display.isna()
    annot = pivot_display.applymap(
        lambda x: f"{x*100:.1f}%" if not pd.isna(x) else ""
    )

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rwg", ["#c0392b", "#ffffff", "#1a7a4a"], N=256
    )

    sns.heatmap(
        pivot_display, ax=ax, cmap=cmap, center=0,
        vmin=-0.08, vmax=0.08,
        annot=annot, fmt="", annot_kws={"size": 7.5},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "%", "shrink": 0.6, "format": "%.0f%%"},
        mask=mask,
    )

    # Add Total column annotation
    totals = (1 + pivot.fillna(0)).prod(axis=1) - 1
    for i, (year, total) in enumerate(totals.items()):
        color = "#1a7a4a" if total >= 0 else "#c0392b"
        ax.text(len(pivot_display.columns) + 0.9, i + 0.5,
                f"{'+' if total >= 0 else ''}{total*100:.1f}%",
                ha="left", va="center", fontsize=7.5,
                color=color, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
