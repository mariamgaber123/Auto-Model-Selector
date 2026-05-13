import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

BG       = "#060b14"
CARD     = "#111d30"
BORDER   = "#1e3050"
ACCENT   = "#3b82f6"
ACCENT2  = "#06b6d4"
TEXT     = "#e2e8f0"
GRID     = "#1e3050"

FIGSIZE = (8, 5)


def _style(fig, ax):
    fig.patch.set_facecolor(BG)

    if not isinstance(ax, (list, tuple, np.ndarray)):
        ax = [ax]

    for a in ax:
        a.set_facecolor(CARD)
        a.tick_params(colors=TEXT, labelsize=9)
        a.xaxis.label.set_color(TEXT)
        a.yaxis.label.set_color(TEXT)
        a.title.set_color(TEXT)
        a.title.set_fontsize(11)

        for spine in a.spines.values():
            spine.set_edgecolor(BORDER)

        a.yaxis.grid(True, color=GRID, linestyle='--', alpha=0.6)
        a.xaxis.grid(False)
        a.set_axisbelow(True)


def plot_histogram(df, column):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.histplot(
        df[column],
        kde=True,
        ax=ax,
        color=ACCENT,
        alpha=0.75,
        edgecolor=BORDER,
        line_kws={"color": ACCENT2, "lw": 2}
    )

    ax.set_title(f"Distribution — {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    _style(fig, ax)
    plt.tight_layout()
    return fig


def plot_bar(df, column):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    counts = df[column].value_counts()
    palette = sns.color_palette("cool", len(counts))

    counts.plot(
        kind='bar',
        ax=ax,
        color=palette,
        edgecolor=BORDER,
        width=0.65
    )

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
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        linewidths=0.4,
        linecolor=BG,
        vmin=-1,
        vmax=1
    )

    ax.set_title("Correlation Heatmap")
    _style(fig, ax)

    plt.tight_layout()
    return fig


def plot_scatter(df, col_x, col_y):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.scatter(
        df[col_x],
        df[col_y],
        color=ACCENT,
        alpha=0.55,
        s=25,
        edgecolors=BORDER,
        linewidths=0.4
    )

    try:
        temp = df[[col_x, col_y]].dropna()
        m, b = np.polyfit(temp[col_x], temp[col_y], 1)

        x_line = sorted(temp[col_x])
        ax.plot(
            x_line,
            [m * x + b for x in x_line],
            color=ACCENT2,
            lw=1.8,
            linestyle='--',
            alpha=0.8
        )
    except:
        pass

    ax.set_title(f"Scatter — {col_x} vs {col_y}")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)

    _style(fig, ax)
    plt.tight_layout()
    return fig


def plot_boxplot(df, col_cat, col_num):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    order = df[col_cat].value_counts().index.tolist()[:15]
    palette = sns.color_palette("cool", len(order))

    sns.boxplot(
        data=df,
        x=col_cat,
        y=col_num,
        order=order,
        palette=palette,
        ax=ax,
        flierprops=dict(marker='o', color=ACCENT2, alpha=0.4, markersize=3),
        medianprops=dict(color=ACCENT2, lw=2),
        whiskerprops=dict(color=TEXT),
        capprops=dict(color=TEXT),
        boxprops=dict(edgecolor=TEXT)
    )

    ax.set_title(f"Box Plot — {col_num} by {col_cat}")
    ax.set_xlabel(col_cat)
    ax.set_ylabel(col_num)

    plt.xticks(rotation=35, ha='right', fontsize=8)

    _style(fig, ax)
    plt.tight_layout()
    return fig


def plot_pie(df, column):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    counts = df[column].value_counts().head(10)
    palette = sns.color_palette("cool", len(counts))

    ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=palette,
        textprops={'color': 'white', 'fontsize': 9},
        wedgeprops={'edgecolor': BORDER, 'linewidth': 1}
    )

    ax.set_title(f"Distribution Percentage — {column}")
    ax.axis('off')

    _style(fig, ax)
    plt.tight_layout()
    return fig


def plot_stacked_bar(df, col_main, col_sub):
    fig, ax = plt.subplots(figsize=(10, 6))

    ct = pd.crosstab(df[col_main], df[col_sub])

    ct.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=sns.color_palette("cool", len(ct.columns)),
        edgecolor=BORDER
    )

    ax.set_title(f"Distribution of {col_sub} across {col_main}")
    ax.set_xlabel(col_main)
    ax.set_ylabel("Count")

    ax.legend(
        title=col_sub,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        facecolor=CARD,
        edgecolor=BORDER
    )

    plt.xticks(rotation=45, ha='right')

    _style(fig, ax)
    plt.tight_layout()
    return fig