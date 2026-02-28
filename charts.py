"""
Chart generation for research output.

Quick mode charts:
  1. Sentiment pie chart (positive / negative / neutral)
  2. Subreddit distribution pie chart (top N subreddits)

Detailed mode adds:
  3. Post type distribution donut chart
  4. Recommendation strength bar chart
  5. Share of voice bar chart
"""

import os
from collections import Counter
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
# 3. Post Type Distribution Donut (detailed mode)
# --------------------------------------------------------------------------

_POST_TYPE_PALETTE = {
    "review": "#4E79A7",
    "question": "#F28E2B",
    "complaint": "#E15759",
    "recommendation": "#59A14F",
    "comparison": "#76B7B2",
    "discussion": "#EDC948",
}


def generate_post_type_chart(
    results: list[dict],
    brand_name: str,
    output_dir: str = CHART_OUTPUT_DIR,
) -> Optional[str]:
    """Donut chart: distribution of post types."""
    _ensure_output_dir(output_dir)

    types = Counter(r.get("post_type", "discussion") for r in results)
    if not types:
        return None

    labels, sizes, colors = [], [], []
    for ptype, count in types.most_common():
        labels.append(ptype.capitalize())
        sizes.append(count)
        colors.append(_POST_TYPE_PALETTE.get(ptype, "#BAB0AC"))

    total = sum(sizes)
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

    ax.text(0, 0, f"{total}\nposts", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#333333")

    legend_labels = [f"{lbl}  ({sz})" for lbl, sz in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=min(3, len(labels)),
              fontsize=10, frameon=False)

    ax.set_title(f"Post Types — {brand_name}",
                 fontsize=16, fontweight="bold", color="#222222", pad=20)
    fig.text(0.5, 0.88, "Last 3 months",
             ha="center", fontsize=11, color="#888888")

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_post_types.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# --------------------------------------------------------------------------
# 4. Recommendation Strength Bar Chart (detailed mode)
# --------------------------------------------------------------------------

_REC_ORDER = ["strong_recommend", "recommend", "neutral", "caution", "strong_negative"]
_REC_COLORS = {
    "strong_recommend": "#2E7D32",
    "recommend": "#66BB6A",
    "neutral": "#9E9E9E",
    "caution": "#EF5350",
    "strong_negative": "#C62828",
}
_REC_LABELS = {
    "strong_recommend": "Strong Recommend",
    "recommend": "Recommend",
    "neutral": "Neutral",
    "caution": "Caution",
    "strong_negative": "Strong Negative",
}


def generate_recommendation_chart(
    results: list[dict],
    brand_name: str,
    output_dir: str = CHART_OUTPUT_DIR,
) -> Optional[str]:
    """Horizontal bar chart: recommendation strength distribution."""
    _ensure_output_dir(output_dir)

    recs = Counter(r.get("recommendation_strength", "neutral") for r in results)
    labels, sizes, colors = [], [], []
    for rec in _REC_ORDER:
        count = recs.get(rec, 0)
        if count > 0:
            labels.append(_REC_LABELS.get(rec, rec))
            sizes.append(count)
            colors.append(_REC_COLORS.get(rec, "#9E9E9E"))

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.8)))
    bars = ax.barh(labels, sizes, color=colors, edgecolor="white", linewidth=1.5, height=0.6)

    for bar, count in zip(bars, sizes):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=11, fontweight="bold", color="#333333")

    ax.set_title(f"Recommendation Strength — {brand_name}",
                 fontsize=16, fontweight="bold", color="#222222")
    ax.set_xlabel("Number of Posts", fontsize=12, color="#555555")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_recommendations.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# --------------------------------------------------------------------------
# 5. Share of Voice Bar Chart (detailed mode)
# --------------------------------------------------------------------------

def generate_share_of_voice_chart(
    sov_data: dict[str, dict],
    brand_name: str,
    output_dir: str = CHART_OUTPUT_DIR,
) -> Optional[str]:
    """Horizontal bar chart: share of voice (brand vs competitors)."""
    _ensure_output_dir(output_dir)

    if not sov_data or all(v["mentions"] == 0 for v in sov_data.values()):
        return None

    # Sort by mentions descending
    sorted_items = sorted(sov_data.items(), key=lambda x: x[1]["mentions"], reverse=True)
    names = [name for name, _ in sorted_items]
    shares = [data["share_pct"] for _, data in sorted_items]
    mentions = [data["mentions"] for _, data in sorted_items]

    colors = ["#4CAF50" if n == brand_name else "#90CAF9" for n in names]

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.8)))
    bars = ax.barh(names, shares, color=colors, edgecolor="white", linewidth=1.5, height=0.6)

    for bar, mention_count, share in zip(bars, mentions, shares):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{share}% ({mention_count} mentions)",
                va="center", fontsize=10, color="#333333")

    ax.set_title(f"Share of Voice — {brand_name}",
                 fontsize=16, fontweight="bold", color="#222222")
    ax.set_xlabel("Share of Voice (%)", fontsize=12, color="#555555")
    ax.set_xlim(0, max(shares) * 1.35 if shares else 100)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    path = os.path.join(output_dir, f"{_safe_name(brand_name)}_sov.png")
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
