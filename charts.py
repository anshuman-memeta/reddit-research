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

from config import CHART_OUTPUT_DIR, SENTIMENT_COLORS


def _ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_sentiment_pie(
    results: list[dict],
    brand_name: str,
    output_dir: str = CHART_OUTPUT_DIR,
) -> str:
    """Pie chart: Positive / Negative / Neutral split."""
    _ensure_output_dir(output_dir)

    sentiments = Counter(r["sentiment"] for r in results)
    total = len(results)

    labels, sizes, colors = [], [], []
    for sentiment in ("positive", "negative", "neutral"):
        count = sentiments.get(sentiment, 0)
        if count > 0:
            pct = count / total * 100
            labels.append(f"{sentiment.capitalize()}\n({count}, {pct:.1f}%)")
            sizes.append(count)
            colors.append(SENTIMENT_COLORS[sentiment])

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")

    ax.set_title(
        f"Sentiment Analysis: {brand_name}\n({total} relevant posts, last 3 months)",
        fontsize=14, fontweight="bold", pad=20,
    )

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_sentiment.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def generate_subreddit_pie(
    results: list[dict],
    brand_name: str,
    output_dir: str = CHART_OUTPUT_DIR,
    top_n: int = 10,
) -> str:
    """Pie chart: top N subreddits by mention count."""
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

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.Set3
    colors = [cmap(i / max(len(labels), 1)) for i in range(len(labels))]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10}, pctdistance=0.85,
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.set_title(
        f"Subreddit Distribution: {brand_name}\n({len(results)} relevant posts, last 3 months)",
        fontsize=14, fontweight="bold", pad=20,
    )

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_subreddits.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


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
        weekly[key][r["sentiment"]] += 1

    if not weekly:
        return None

    weeks = sorted(weekly.keys())
    pos = [weekly[w]["positive"] for w in weeks]
    neg = [weekly[w]["negative"] for w in weeks]
    neu = [weekly[w]["neutral"] for w in weeks]
    labels = [datetime.strptime(w, "%Y-%m-%d").strftime("%b %d") for w in weeks]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(labels, pos, "o-", color=SENTIMENT_COLORS["positive"],
            label="Positive", linewidth=2, markersize=6)
    ax.plot(labels, neg, "o-", color=SENTIMENT_COLORS["negative"],
            label="Negative", linewidth=2, markersize=6)
    ax.plot(labels, neu, "o-", color=SENTIMENT_COLORS["neutral"],
            label="Neutral", linewidth=2, markersize=6)

    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_title(
        f"Sentiment Trend: {brand_name}\n(Weekly, last 3 months)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_trend.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _safe_name(name: str) -> str:
    """Sanitize brand name for use in file paths."""
    return name.lower().replace(" ", "_").replace("&", "and").replace("'", "")
