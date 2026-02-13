"""
Chart generation for research output.

Generates three charts:
  1. Sentiment pie chart (positive / negative / neutral)
  2. Subreddit distribution pie chart (top N subreddits)
  3. Sentiment trend line chart (weekly buckets over 3 months)
"""

import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from config import CHART_OUTPUT_DIR, SENTIMENT_COLORS

# --------------------------------------------------------------------------
# Global style tweaks
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

DONUT_WIDTH = 0.45  # ring thickness for donut charts


def _ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# --------------------------------------------------------------------------
# 1. Sentiment Donut Chart
# --------------------------------------------------------------------------

def generate_sentiment_pie(
    results: list[dict],
    brand_name: str,
    output_dir: str = CHART_OUTPUT_DIR,
) -> str:
    """Donut chart: Positive / Negative / Neutral split with center total."""
    _ensure_output_dir(output_dir)

    sentiments = Counter(r["sentiment"] for r in results)
    total = len(results)

    labels, sizes, colors = [], [], []
    for sentiment in ("positive", "negative", "neutral"):
        count = sentiments.get(sentiment, 0)
        if count > 0:
            labels.append(sentiment.capitalize())
            sizes.append(count)
            colors.append(SENTIMENT_COLORS[sentiment])

    fig, ax = plt.subplots(figsize=(8, 6))

    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"width": DONUT_WIDTH, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12, "fontweight": "bold", "color": "#333333"},
    )

    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")

    # Center text: total count
    ax.text(
        0, 0, f"{total}\nposts",
        ha="center", va="center",
        fontsize=22, fontweight="bold", color="#333333",
    )

    # Legend below chart with counts
    legend_labels = [
        f"{lbl}  ({sz})" for lbl, sz in zip(labels, sizes)
    ]
    legend = ax.legend(
        wedges, legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(labels),
        fontsize=11,
        frameon=False,
        handlelength=1.2,
        handleheight=1.2,
    )

    ax.set_title(
        f"Sentiment Analysis — {brand_name}",
        fontsize=16, fontweight="bold", color="#222222", pad=20,
    )
    fig.text(
        0.5, 0.88, "Last 3 months",
        ha="center", fontsize=11, color="#888888",
    )

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_sentiment.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# --------------------------------------------------------------------------
# 2. Subreddit Distribution Donut Chart (with clean legend)
# --------------------------------------------------------------------------

# Distinct palette that works well for categorical data
_SUBREDDIT_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#86BCB6", "#8CD17D",
]


def generate_subreddit_pie(
    results: list[dict],
    brand_name: str,
    output_dir: str = CHART_OUTPUT_DIR,
    top_n: int = 10,
) -> str:
    """Donut chart: top N subreddits with a side legend (no overlapping labels)."""
    _ensure_output_dir(output_dir)

    subreddits = Counter(r["subreddit"] for r in results)
    top = subreddits.most_common(top_n)

    labels = [f"r/{sub}" for sub, _ in top]
    sizes = [count for _, count in top]

    top_total = sum(sizes)
    others = len(results) - top_total
    if others > 0:
        labels.append("Others")
        sizes.append(others)

    total = sum(sizes)
    colors = [_SUBREDDIT_PALETTE[i % len(_SUBREDDIT_PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 7))

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops={"width": DONUT_WIDTH, "edgecolor": "white", "linewidth": 2},
    )

    # Center text
    ax.text(
        0, 0, f"{total}\nposts",
        ha="center", va="center",
        fontsize=22, fontweight="bold", color="#333333",
    )

    # Side legend with counts & percentages — avoids label overlap entirely
    legend_labels = [
        f"{lbl}  —  {sz} ({sz / total * 100:.1f}%)"
        for lbl, sz in zip(labels, sizes)
    ]
    legend = ax.legend(
        wedges, legend_labels,
        title="Subreddits",
        title_fontproperties=fm.FontProperties(weight="bold", size=12),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    ax.set_title(
        f"Subreddit Distribution — {brand_name}",
        fontsize=16, fontweight="bold", color="#222222", pad=20,
    )
    fig.text(
        0.40, 0.88, "Last 3 months",
        ha="center", fontsize=11, color="#888888",
    )

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_subreddits.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# --------------------------------------------------------------------------
# 3. Sentiment Trend Line Chart (polished)
# --------------------------------------------------------------------------

def generate_sentiment_trend(
    results: list[dict],
    brand_name: str,
    output_dir: str = CHART_OUTPUT_DIR,
) -> Optional[str]:
    """Line chart: weekly sentiment counts over time."""
    _ensure_output_dir(output_dir)

    weekly: dict[str, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0, "neutral": 0}
    )

    for r in results:
        dt = datetime.strptime(r["created_date"], "%Y-%m-%d")
        week_start = dt - timedelta(days=dt.weekday())
        key = week_start.strftime("%Y-%m-%d")
        sentiment = r["sentiment"] if r["sentiment"] in SENTIMENT_COLORS else "neutral"
        weekly[key][sentiment] += 1

    if not weekly:
        return None

    weeks = sorted(weekly.keys())
    pos = [weekly[w]["positive"] for w in weeks]
    neg = [weekly[w]["negative"] for w in weeks]
    neu = [weekly[w]["neutral"] for w in weeks]
    labels = [datetime.strptime(w, "%Y-%m-%d").strftime("%b %d") for w in weeks]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(labels, pos, "o-", color=SENTIMENT_COLORS["positive"],
            label="Positive", linewidth=2.5, markersize=6, alpha=0.9)
    ax.plot(labels, neg, "o-", color=SENTIMENT_COLORS["negative"],
            label="Negative", linewidth=2.5, markersize=6, alpha=0.9)
    ax.plot(labels, neu, "o-", color=SENTIMENT_COLORS["neutral"],
            label="Neutral", linewidth=2.5, markersize=6, alpha=0.9)

    # Light fill under curves
    ax.fill_between(range(len(labels)), pos, alpha=0.08, color=SENTIMENT_COLORS["positive"])
    ax.fill_between(range(len(labels)), neg, alpha=0.08, color=SENTIMENT_COLORS["negative"])

    ax.set_xlabel("Week", fontsize=12, color="#555555")
    ax.set_ylabel("Number of Posts", fontsize=12, color="#555555")
    ax.set_title(
        f"Sentiment Trend — {brand_name}",
        fontsize=16, fontweight="bold", color="#222222",
    )
    fig.text(
        0.5, 0.90, "Weekly, last 3 months",
        ha="center", fontsize=11, color="#888888",
    )
    ax.legend(fontsize=11, frameon=True, fancybox=True, edgecolor="#CCCCCC")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right")

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_trend.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Sanitize brand name for use in file paths."""
    return name.lower().replace(" ", "_").replace("&", "and").replace("'", "")
