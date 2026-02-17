"""
LLM-powered relevance filtering and sentiment analysis.

Uses batch processing: up to 50 posts per Groq API call to minimise
rate-limit hits and wall-clock time.  Falls back to keyword heuristics
when the LLM is unavailable.
"""

import json
import logging
import time
from typing import Callable, Optional

import requests

from config import GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDERS
from fetcher import RedditPost

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------

# Single-post prompt (used as fallback if a batch fails)
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

# Batch prompt — analyses many posts in one LLM call
BATCH_PROMPT = """You are analyzing multiple Reddit posts to determine if each one is about a specific brand, and if so, what the sentiment is.

Brand: {brand_name}
Description: {brand_description}
Category: {category}
Known competitors: {competitors}

The brand name may also be a common word (e.g. "Sahi" means "correct" in Hindi). Only mark as relevant if the post is ACTUALLY discussing the brand/product.

Below are the posts. Each has an "id" you must include in your response.

{posts_block}

Respond with ONLY a JSON array. One object per post, in the same order:
- If NOT about the brand: {{"id": "<post_id>", "relevant": false}}
- If about the brand: {{"id": "<post_id>", "relevant": true, "sentiment": "positive|negative|neutral", "theme": "short theme", "summary": "One-line summary.", "competitor_mentions": ["Competitor"]}}

Return ONLY the JSON array, no other text."""

BATCH_SIZE = 10  # posts per LLM call

_VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def _normalize_sentiment(value: str) -> str:
    """Clamp sentiment to one of the three valid values."""
    v = value.strip().lower()
    if v in _VALID_SENTIMENTS:
        return v
    return "neutral"


# --------------------------------------------------------------------------
# Analyzer
# --------------------------------------------------------------------------

_UNSET = object()  # sentinel to distinguish "no arg" from "passed None"


class BrandAnalyzer:
    """Batch LLM pipeline: relevance filter + sentiment analysis."""

    def __init__(self, api_key=_UNSET, model=_UNSET):
        # Build the provider chain.  When explicit api_key/model are passed
        # (e.g. tests with api_key=None), use only that single provider.
        if api_key is not _UNSET or model is not _UNSET:
            key = api_key if api_key is not _UNSET else GROQ_API_KEY
            mdl = model if model is not _UNSET else GROQ_MODEL
            self.providers = [
                {
                    "name": "Groq",
                    "api_url": "https://api.groq.com/openai/v1/chat/completions",
                    "api_key": key,
                    "model": mdl,
                }
            ]
        else:
            # Use the configured fallback chain, skip providers without keys
            self.providers = [p for p in LLM_PROVIDERS if p.get("api_key")]

        # Per-provider rate-limit tracking
        self._rate_limited: dict[str, bool] = {
            p["name"]: False for p in self.providers
        }

    # ----- Low-level LLM call ---------------------------------------------

    def _call_single_provider(
        self,
        provider: dict,
        prompt: str,
        max_retries: int,
        max_tokens: int,
    ) -> Optional[str]:
        """Call one provider's API. Returns content string or None."""
        name = provider["name"]

        if not provider.get("api_key") or self._rate_limited.get(name):
            return None

        headers = {
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": provider["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    provider["api_url"], headers=headers, json=payload, timeout=60,
                )
                resp.raise_for_status()
                self._rate_limited[name] = False
                content = resp.json()["choices"][0]["message"]["content"].strip()
                return content

            except requests.exceptions.HTTPError:
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        f"{name} rate-limited (attempt {attempt+1}/{max_retries}), "
                        f"waiting {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"{name} HTTP error: {resp.status_code} {resp.text[:200]}")
                return None
            except Exception as e:
                logger.error(f"{name} call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        logger.warning(f"{name} exhausted all retries")
        self._rate_limited[name] = True
        return None

    def _call_llm(
        self,
        prompt: str,
        max_retries: int = 3,
        max_tokens: int = 256,
    ) -> Optional[str]:
        """Try each provider in order until one succeeds."""
        for provider in self.providers:
            name = provider["name"]
            if self._rate_limited.get(name):
                continue

            result = self._call_single_provider(
                provider, prompt, max_retries, max_tokens,
            )
            if result is not None:
                return result

            # Provider failed — log and try the next one
            if len(self.providers) > 1:
                logger.info(f"  {name} failed, trying next provider...")

        return None

    @property
    def _all_rate_limited(self) -> bool:
        """True when every configured provider is rate-limited."""
        return all(self._rate_limited.get(p["name"]) for p in self.providers)

    @staticmethod
    def _parse_json(raw: str):
        """Strip markdown fences and parse JSON."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    # ----- Batch analysis -------------------------------------------------

    def _build_posts_block(self, posts: list[RedditPost]) -> str:
        """Format a list of posts for the batch prompt."""
        parts = []
        for p in posts:
            parts.append(
                f'---\nid: {p.post_id}\n'
                f'title: {p.title[:300]}\n'
                f'body: {(p.selftext or "")[:600]}\n'
                f'subreddit: r/{p.subreddit}\n'
                f'upvotes: {p.score}'
            )
        return "\n".join(parts)

    def analyze_batch(
        self,
        posts: list[RedditPost],
        brand_config: dict,
    ) -> dict[str, Optional[dict]]:
        """
        Analyse up to BATCH_SIZE posts in a single LLM call.
        Returns {post_id: analysis_dict_or_None}.
        """
        competitors = brand_config.get("competitors", [])
        prompt = BATCH_PROMPT.format(
            brand_name=brand_config["name"],
            brand_description=brand_config.get("description", ""),
            category=brand_config.get("category", "general"),
            competitors=", ".join(competitors) if competitors else "none known",
            posts_block=self._build_posts_block(posts),
        )

        # Allocate ~120 tokens per post for the response
        max_tokens = min(len(posts) * 150, 4096)

        raw = self._call_llm(prompt, max_retries=4, max_tokens=max_tokens)
        if raw is None:
            return {}

        try:
            parsed = self._parse_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse batch response: {e}")
            return {}

        if not isinstance(parsed, list):
            logger.error("Batch response is not a JSON array")
            return {}

        results: dict[str, Optional[dict]] = {}
        for item in parsed:
            pid = item.get("id")
            if not pid:
                continue
            if not item.get("relevant", False):
                results[pid] = None
            else:
                results[pid] = {
                    "sentiment": _normalize_sentiment(item.get("sentiment", "neutral")),
                    "theme": item.get("theme", "general discussion"),
                    "summary": item.get("summary", ""),
                    "competitor_mentions": item.get("competitor_mentions", []),
                }
        return results

    # ----- Single-post analysis (fallback) --------------------------------

    def analyze_post(self, post: RedditPost, brand_config: dict) -> Optional[dict]:
        """
        Combined relevance check + sentiment in ONE LLM call (single post).
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

        raw = self._call_llm(prompt, max_tokens=256)
        if raw is not None:
            try:
                result = self._parse_json(raw)
                if not result.get("relevant", False):
                    return None
                return {
                    "sentiment": _normalize_sentiment(result.get("sentiment", "neutral")),
                    "theme": result.get("theme", "general discussion"),
                    "summary": result.get("summary", ""),
                    "competitor_mentions": result.get("competitor_mentions", []),
                }
            except (json.JSONDecodeError, ValueError):
                logger.error("Failed to parse single-post LLM response")

        return self._keyword_fallback(post, brand_config)

    # ----- Keyword fallback -----------------------------------------------

    def _keyword_fallback(self, post: RedditPost, brand_config: dict) -> Optional[dict]:
        """Keyword-based relevance + sentiment when LLM is unavailable."""
        text = f"{post.title} {post.selftext}".lower()
        product_terms = [t.lower() for t in brand_config.get("product_terms", [])]
        hint_subs = [s.lower() for s in brand_config.get("subreddit_hints", [])]

        has_product_term = any(term in text for term in product_terms)
        in_relevant_sub = post.subreddit.lower() in hint_subs
        if not (has_product_term or in_relevant_sub):
            return None

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
        Batch analysis pipeline. Sends up to BATCH_SIZE posts per LLM call.
        Falls back to single-post calls for any posts missing from the
        batch response, then keyword heuristics as last resort.

        When Groq rate-limits us, we stop hammering the API and use keyword
        fallback for the rest.  Each new batch resets the flag to try again.
        """
        results: list[dict] = []
        total = len(posts)
        logger.info(f"Starting batch analysis of {total} posts (batch size={BATCH_SIZE})...")

        processed = 0
        fallback_count = 0

        for batch_start in range(0, total, BATCH_SIZE):
            batch = posts[batch_start : batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1

            # Reset rate-limit flags each batch so we retry providers
            for name in self._rate_limited:
                self._rate_limited[name] = False

            logger.info(
                f"[Batch {batch_num}] Analysing posts "
                f"{batch_start + 1}-{batch_start + len(batch)} of {total}..."
            )

            # --- Try batch LLM call ---
            batch_results = self.analyze_batch(batch, brand_config)
            batch_was_rate_limited = self._all_rate_limited

            for post in batch:
                processed += 1

                if post.post_id in batch_results:
                    analysis = batch_results[post.post_id]
                elif batch_was_rate_limited:
                    # Batch failed due to rate limits — don't hammer with
                    # individual calls, go straight to keyword fallback
                    analysis = self._keyword_fallback(post, brand_config)
                    fallback_count += 1
                else:
                    # Batch succeeded but this post was missing from response
                    logger.info(
                        f"  Post {post.post_id} missing from batch, "
                        f"trying single call..."
                    )
                    analysis = self.analyze_post(post, brand_config)
                    time.sleep(2)  # small delay for single fallback calls

                if analysis is None:
                    logger.info(f"  [{processed}/{total}] Filtered: {post.title[:50]}")
                else:
                    logger.info(
                        f"  [{processed}/{total}] Relevant: "
                        f"sentiment={analysis['sentiment']}, "
                        f"theme={analysis['theme'][:40]}"
                    )
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

            # Progress update after each batch
            if progress_callback:
                progress_callback(processed, total, len(results))

            # Pause between batches — longer if we just got rate-limited
            if batch_start + BATCH_SIZE < total:
                cooldown = 30 if batch_was_rate_limited else 5
                if batch_was_rate_limited:
                    logger.info(f"  Rate-limited — cooling down {cooldown}s before next batch")
                time.sleep(cooldown)

        if fallback_count:
            logger.info(f"  ({fallback_count} posts used keyword fallback due to rate limits)")
        logger.info(f"Analysis complete: {len(results)} relevant out of {total} posts")
        return results
