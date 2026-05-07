import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BG       = "#060b14"
SURFACE  = "#0d1626"
CARD     = "#111d30"
BORDER   = "#1e3050"
ACCENT   = "#3b82f6"
ACCENT2  = "#06b6d4"
SUCCESS  = "#10b981"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"
GRID     = "#1e3050"

def _style(fig, axes):
    """Apply dark theme to figure and one or more axes."""
    fig.patch.set_facecolor(BG)
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.title.set_fontsize(11)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.yaxis.grid(True, color=GRID, linestyle='--', alpha=0.6)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)


# ── Single-column charts ──────────────────────────────────────────────

def plot_histogram(df, column):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df[column], kde=True, ax=ax,
                 color=ACCENT, alpha=0.75, edgecolor=BORDER,
                 line_kws={"color": ACCENT2, "lw": 2})
    ax.set_title(f"Distribution — {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    _style(fig, ax)
    plt.tight_layout()
    return fig


def plot_bar(df, column):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = df[column].value_counts()
    palette = sns.color_palette("cool", len(counts))
    counts.plot(kind='bar', ax=ax, color=palette, edgecolor=BORDER, width=0.65)
    ax.set_title(f"Value Counts — {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    plt.xticks(rotation=40, ha='right', fontsize=8)
    _style(fig, ax)
    plt.tight_layout()
    return fig


def plot_heatmap(df):
    corr = df.corr(numeric_only=True)
    size = max(7, len(corr) * 0.9)
    fig, ax = plt.subplots(figsize=(size, size * 0.75))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        ax=ax, linewidths=0.4, linecolor=BG,
        annot_kws={"size": 8, "color": "white"},
        vmin=-1, vmax=1
    )
    ax.set_title("Correlation Heatmap")
    fig.patch.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.title.set_color(TEXT)
    plt.tight_layout()
    return fig


# ── Two-column charts ─────────────────────────────────────────────────

def plot_scatter(df, col_x, col_y):
    """Scatter plot between two numeric columns."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df[col_x], df[col_y],
        color=ACCENT, alpha=0.55, s=25, edgecolors=BORDER, linewidths=0.4
    )
    # regression line
    try:
        import numpy as np
        m, b = np.polyfit(df[col_x].dropna(), df[col_y].dropna(), 1)
        x_line = sorted(df[col_x].dropna())
        ax.plot(x_line, [m * x + b for x in x_line],
                color=ACCENT2, lw=1.8, linestyle='--', alpha=0.8)
    except Exception:
        pass
    ax.set_title(f"Scatter — {col_x}  vs  {col_y}")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    _style(fig, ax)
    plt.tight_layout()
    return fig


def plot_boxplot(df, col_cat, col_num):
    """Box plot: numeric column grouped by categorical column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    order   = df[col_cat].value_counts().index.tolist()[:15]  # max 15 categories
    palette = sns.color_palette("cool", len(order))
    sns.boxplot(
        data=df, x=col_cat, y=col_num, order=order,
        palette=palette, ax=ax,
        flierprops=dict(marker='o', color=ACCENT2, alpha=0.4, markersize=3),
        medianprops=dict(color=ACCENT2, lw=2),
        whiskerprops=dict(color=TEXT),
        capprops=dict(color=TEXT),
        boxprops=dict(edgecolor=TEXT)
    )
    ax.set_title(f"Box Plot — {col_num}  by  {col_cat}")
    ax.set_xlabel(col_cat)
    ax.set_ylabel(col_num)
    plt.xticks(rotation=35, ha='right', fontsize=8)
    _style(fig, ax)
    plt.tight_layout()
    return fig 