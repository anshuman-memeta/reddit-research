"""
LLM-powered relevance filtering and sentiment analysis.

Two-step pipeline:
  1. Relevance check  - Is this post actually about the brand?
  2. Sentiment analysis - Sentiment, theme, competitor co-mentions.

Uses Groq (Llama 3.3 70B) with keyword-based fallbacks.
"""

import json
import logging
import time
from typing import Callable, Optional

import requests

from config import GROQ_API_KEY, GROQ_MODEL
from fetcher import RedditPost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Single combined prompt: relevance + sentiment in one call (halves API usage)
COMBINED_PROMPT = """You are analyzing a Reddit post to determine if it's about a specific brand, and if so, what the sentiment is.

Brand: {brand_name}
Description: {brand_description}
Category: {category}

Reddit post title: {title}
Reddit post body: {body}
Subreddit: r/{subreddit}
Upvotes: {score}

The brand name may also be a common word (e.g. "Sahi" means "correct" in Hindi). Only mark as relevant if the post is actually discussing the brand/product.

Known competitors: {competitors}

Respond with ONLY a JSON object:
- If the post is NOT about the brand: {{"relevant": false}}
- If the post IS about the brand: {{"relevant": true, "sentiment": "positive", "theme": "great sunscreen formula", "summary": "User recommends the brand's sunscreen for oily skin.", "competitor_mentions": ["Minimalist"]}}"""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class BrandAnalyzer:
    """Two-step LLM pipeline: relevance filter -> sentiment analysis."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    # ----- LLM call -------------------------------------------------------

    def _call_llm(self, prompt: str, max_retries: int = 3) -> Optional[dict]:
        """Call Groq API, parse JSON response. Returns None on failure."""
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 256,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()

                # Strip ```json fences if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                parsed = json.loads(content)
                logger.info(f"LLM response OK: relevant={parsed.get('relevant', '?')}")
                return parsed

            except requests.exceptions.HTTPError:
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Groq rate-limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"Groq HTTP error: {resp.status_code} {resp.text[:200]}")
                return None
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.error(f"Failed to parse LLM response: {e}")
                return None
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        logger.warning("LLM call exhausted all retries, falling back to keywords")
        return None

    # ----- Single-call analysis (relevance + sentiment) --------------------

    def analyze_post(self, post: RedditPost, brand_config: dict) -> Optional[dict]:
        """
        Combined relevance check + sentiment in ONE LLM call.
        Returns analysis dict if relevant, None if not.
        Falls back to keyword heuristics if LLM unavailable.
        """
        competitors = brand_config.get("competitors", [])
        prompt = COMBINED_PROMPT.format(
            brand_name=brand_config["name"],
            brand_description=brand_config.get("description", ""),
            category=brand_config.get("category", "general"),
            title=post.title[:500],
            body=(post.selftext or "")[:1000],
            subreddit=post.subreddit,
            score=post.score,
            competitors=", ".join(competitors) if competitors else "none known",
        )

        result = self._call_llm(prompt)
        if result is not None:
            if not result.get("relevant", False):
                return None
            return {
                "sentiment": result.get("sentiment", "neutral"),
                "theme": result.get("theme", "general discussion"),
                "summary": result.get("summary", ""),
                "competitor_mentions": result.get("competitor_mentions", []),
            }

        # Fallback: keyword heuristics
        return self._keyword_fallback(post, brand_config)

    def _keyword_fallback(self, post: RedditPost, brand_config: dict) -> Optional[dict]:
        """Keyword-based relevance + sentiment when LLM is unavailable."""
        text = f"{post.title} {post.selftext}".lower()
        product_terms = [t.lower() for t in brand_config.get("product_terms", [])]
        hint_subs = [s.lower() for s in brand_config.get("subreddit_hints", [])]

        # Relevance check
        has_product_term = any(term in text for term in product_terms)
        in_relevant_sub = post.subreddit.lower() in hint_subs
        if not (has_product_term or in_relevant_sub):
            return None

        # Sentiment
        positive = ["love", "great", "amazing", "best", "awesome", "excellent",
                     "recommend", "good", "fantastic", "happy", "satisfied", "smooth"]
        negative = ["hate", "worst", "terrible", "bad", "awful", "scam", "fraud",
                     "disappointed", "horrible", "poor", "waste", "trash", "bug",
                     "crash", "slow", "stuck", "useless"]

        pos = sum(1 for w in positive if w in text)
        neg = sum(1 for w in negative if w in text)

        if pos > neg:
            sentiment = "positive"
        elif neg > pos:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        competitors_found = [c for c in brand_config.get("competitors", []) if c.lower() in text]

        return {
            "sentiment": sentiment,
            "theme": "general discussion",
            "summary": post.title[:100],
            "competitor_mentions": competitors_found,
        }

    # ----- Full pipeline --------------------------------------------------

    def process_posts(
        self,
        posts: list[RedditPost],
        brand_config: dict,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> list[dict]:
        """
        Run combined relevance + sentiment analysis on all posts.
        Uses a single LLM call per post (not two).
        """
        results: list[dict] = []
        total = len(posts)
        logger.info(f"Starting analysis of {total} posts...")

        for i, post in enumerate(posts):
            logger.info(f"[{i+1}/{total}] Analyzing: {post.title[:60]}...")
            analysis = self.analyze_post(post, brand_config)

            if analysis is None:
                logger.info(f"  -> Filtered out (not relevant)")
            else:
                logger.info(f"  -> Relevant! sentiment={analysis['sentiment']}, theme={analysis['theme'][:40]}")
                results.append({
                    "post_id": post.post_id,
                    "title": post.title,
                    "selftext": (post.selftext or "")[:500],
                    "subreddit": post.subreddit,
                    "author": post.author,
                    "url": post.permalink,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_date": post.created_date.strftime("%Y-%m-%d"),
                    "created_utc": post.created_utc,
                    "relevance_confidence": 0.9,
                    "sentiment": analysis["sentiment"],
                    "theme": analysis["theme"],
                    "summary": analysis["summary"],
                    "competitor_mentions": ", ".join(analysis["competitor_mentions"]),
                })

            # Groq free tier: ~30 req/min. Wait 2.5s to stay under.
            time.sleep(2.5)

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total, len(results))

        logger.info(f"Analysis complete: {len(results)} relevant out of {total} posts")
        return results
