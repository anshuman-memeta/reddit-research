"""
Multi-source Reddit post fetcher.

Sources (in priority order):
  1. Arctic Shift API (primary — reliable from datacenter IPs)
  2. Reddit search JSON endpoint (secondary — often 403 from VPS IPs)
  3. Pullpush.io API (fallback, unreliable)

All sources are deduplicated by post ID before returning.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import requests

from config import REDDIT_RATE_LIMIT_DELAY, REDDIT_USER_AGENT

logger = logging.getLogger(__name__)


# Default subreddits to search when brand has no subreddit_hints.
# Covers general Indian consumer, tech, and lifestyle communities.
DEFAULT_SUBREDDITS = [
    "india", "AskIndia", "indiasocial",
    "IndianGaming", "IndianConsumer", "IndiaTech",
    "gadgets", "technology", "BuyItForLife",
    "IndianSkincareAddicts", "IndianFashionAddicts",
    "IndiaInvestments", "CreditCardsIndia",
    "Fitness", "SkincareAddiction",
]


@dataclass
class RedditPost:
    """A single Reddit post with metadata."""

    post_id: str
    title: str
    selftext: str
    subreddit: str
    author: str
    url: str
    permalink: str
    score: int
    num_comments: int
    created_utc: float
    created_date: datetime = field(init=False)

    def __post_init__(self):
        self.created_date = datetime.fromtimestamp(self.created_utc, tz=timezone.utc)

    def __eq__(self, other):
        return isinstance(other, RedditPost) and self.post_id == other.post_id

    def __hash__(self):
        return hash(self.post_id)


# ---------------------------------------------------------------------------
# Source 1 (PRIMARY): Arctic Shift API
# ---------------------------------------------------------------------------

class ArcticShiftFetcher:
    """
    Fetches posts from the Arctic Shift Reddit archive.
    This is the most reliable source from datacenter IPs since it's a proper
    API that doesn't block server requests.

    NOTE: Full-text search requires a subreddit or author filter.
    """

    BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

    def __init__(self, rate_limit: float = 1.0):
        self.session = requests.Session()
        self.rate_limit = rate_limit

    def search_subreddit(self, subreddit: str, query: str,
                         after_date: Optional[str] = None,
                         limit: int = 100,
                         max_pages: int = 10,
                         max_retries: int = 2) -> list[RedditPost]:
        """
        Search within a specific subreddit with retry logic.

        Args:
            subreddit: Subreddit name (without r/).
            query: Search query.
            after_date: Date string like "2024-11-09". Defaults to 90 days ago.
            limit: Results per page (max 100).
            max_retries: Number of retries per request on failure.
        """
        if after_date is None:
            after_date = (datetime.now(tz=timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")

        posts: list[RedditPost] = []
        before_date: Optional[str] = None

        for page in range(max_pages):
            params = {
                "query": query,
                "subreddit": subreddit,
                "after": after_date,
                "limit": limit,
                "sort": "desc",
            }
            if before_date:
                params["before"] = before_date

            data = None
            for attempt in range(max_retries + 1):
                try:
                    resp = self.session.get(self.BASE_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if attempt < max_retries:
                        wait = 2 * (attempt + 1)
                        logger.warning(f"Arctic Shift r/{subreddit} attempt {attempt + 1} failed: {e}, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"Arctic Shift error (r/{subreddit}, page {page}): {e}")

            if data is None:
                break

            results = data.get("data", [])
            if not results:
                break

            for d in results:
                created = d.get("created_utc", 0)
                posts.append(RedditPost(
                    post_id=d.get("id", ""),
                    title=d.get("title", ""),
                    selftext=d.get("selftext", ""),
                    subreddit=d.get("subreddit", ""),
                    author=d.get("author", "[deleted]"),
                    url=d.get("url", ""),
                    permalink=f"https://reddit.com/r/{d.get('subreddit', '')}/comments/{d.get('id', '')}",
                    score=d.get("score", 0),
                    num_comments=d.get("num_comments", 0),
                    created_utc=created,
                ))

            # Paginate: use last post's date as the new before cursor
            if results:
                last_ts = results[-1].get("created_utc", 0)
                before_date = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d")

            if len(results) < limit:
                break

            time.sleep(self.rate_limit)

        return posts


# ---------------------------------------------------------------------------
# Source 2: Reddit's own search JSON endpoint (often blocked from VPS IPs)
# ---------------------------------------------------------------------------

class RedditSearchFetcher:
    """
    Fetches posts via Reddit search JSON endpoints.
    Falls back from www.reddit.com -> old.reddit.com on persistent 403s.
    NOTE: Frequently returns 403 from datacenter/VPS IPs.
    """

    ENDPOINTS = [
        "https://www.reddit.com",
        "https://old.reddit.com",
    ]

    def __init__(self, user_agent: str = REDDIT_USER_AGENT, rate_limit: float = REDDIT_RATE_LIMIT_DELAY):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        self.rate_limit = rate_limit
        self._working_endpoint_idx = 0

    def _get_with_retry(self, url: str, params: dict, max_retries: int = 2) -> Optional[dict]:
        """GET with retry on 403/429 errors."""
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code in (403, 429):
                    wait = 3 * (attempt + 1)
                    logger.warning(f"Reddit {resp.status_code} on {url} (attempt {attempt + 1})")
                    if attempt >= 1:
                        return None
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
        return None

    def _search_with_fallback(self, path: str, params: dict) -> Optional[dict]:
        """Try each endpoint until one succeeds."""
        endpoints = self.ENDPOINTS[self._working_endpoint_idx:] + self.ENDPOINTS[:self._working_endpoint_idx]
        for base in endpoints:
            url = f"{base}{path}"
            data = self._get_with_retry(url, params)
            if data is not None:
                self._working_endpoint_idx = self.ENDPOINTS.index(base)
                return data
        return None

    def search(self, query: str, sort: str = "new", time_filter: str = "year",
               limit: int = 100, max_pages: int = 10) -> list[RedditPost]:
        """Global Reddit search with pagination."""
        posts: list[RedditPost] = []
        after: Optional[str] = None

        for page in range(max_pages):
            params = {
                "q": query, "sort": sort, "t": time_filter,
                "limit": limit, "type": "link",
            }
            if after:
                params["after"] = after

            try:
                data = self._search_with_fallback("/search.json", params)
                if data is None:
                    break
            except Exception as e:
                logger.error(f"Reddit search error (page {page}): {e}")
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                d = child.get("data", {})
                posts.append(RedditPost(
                    post_id=d.get("id", ""),
                    title=d.get("title", ""),
                    selftext=d.get("selftext", ""),
                    subreddit=d.get("subreddit", ""),
                    author=d.get("author", ""),
                    url=d.get("url", ""),
                    permalink=f"https://reddit.com{d.get('permalink', '')}",
                    score=d.get("score", 0),
                    num_comments=d.get("num_comments", 0),
                    created_utc=d.get("created_utc", 0),
                ))

            after = data.get("data", {}).get("after")
            if not after:
                break
            time.sleep(self.rate_limit)

        return posts

    def search_subreddit(self, subreddit: str, query: str, sort: str = "new",
                         time_filter: str = "year", limit: int = 100,
                         max_pages: int = 5) -> list[RedditPost]:
        """Search within a specific subreddit."""
        path = f"/r/{subreddit}/search.json"
        posts: list[RedditPost] = []
        after: Optional[str] = None

        for page in range(max_pages):
            params = {
                "q": query, "sort": sort, "t": time_filter,
                "limit": limit, "restrict_sr": "on", "type": "link",
            }
            if after:
                params["after"] = after

            try:
                data = self._search_with_fallback(path, params)
                if data is None:
                    break
            except Exception as e:
                logger.error(f"Subreddit search error (r/{subreddit}, page {page}): {e}")
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                d = child.get("data", {})
                posts.append(RedditPost(
                    post_id=d.get("id", ""),
                    title=d.get("title", ""),
                    selftext=d.get("selftext", ""),
                    subreddit=d.get("subreddit", ""),
                    author=d.get("author", ""),
                    url=d.get("url", ""),
                    permalink=f"https://reddit.com{d.get('permalink', '')}",
                    score=d.get("score", 0),
                    num_comments=d.get("num_comments", 0),
                    created_utc=d.get("created_utc", 0),
                ))

            after = data.get("data", {}).get("after")
            if not after:
                break
            time.sleep(self.rate_limit)

        return posts


# ---------------------------------------------------------------------------
# Source 3: Pullpush.io (unreliable fallback)
# ---------------------------------------------------------------------------

class PullpushFetcher:
    """Fetches posts from Pullpush.io (Pushshift mirror). May be down."""

    BASE_URL = "https://api.pullpush.io/reddit/search/submission"

    def __init__(self, rate_limit: float = 1.0):
        self.session = requests.Session()
        self.rate_limit = rate_limit

    def search(self, query: str, after_ts: Optional[int] = None,
               before_ts: Optional[int] = None, limit: int = 100,
               max_pages: int = 10) -> list[RedditPost]:

        now = int(datetime.now(tz=timezone.utc).timestamp())
        if after_ts is None:
            after_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=90)).timestamp())
        if before_ts is None:
            before_ts = now

        posts: list[RedditPost] = []

        for page in range(max_pages):
            params = {
                "q": query,
                "after": after_ts,
                "before": before_ts,
                "size": limit,
                "sort": "desc",
                "sort_type": "created_utc",
            }

            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Pullpush error (page {page}): {e}")
                break

            results = data.get("data", [])
            if not results:
                break

            for d in results:
                posts.append(RedditPost(
                    post_id=d.get("id", ""),
                    title=d.get("title", ""),
                    selftext=d.get("selftext", ""),
                    subreddit=d.get("subreddit", ""),
                    author=d.get("author", "[deleted]"),
                    url=d.get("url", ""),
                    permalink=f"https://reddit.com{d.get('permalink', '')}",
                    score=d.get("score", 0),
                    num_comments=d.get("num_comments", 0),
                    created_utc=d.get("created_utc", 0),
                ))

            if results:
                before_ts = int(results[-1].get("created_utc", before_ts)) - 1

            if len(results) < limit:
                break

            time.sleep(self.rate_limit)

        return posts


# ---------------------------------------------------------------------------
# Orchestrator: fetch from all sources, deduplicate
# ---------------------------------------------------------------------------

class MultiSourceFetcher:
    """
    Orchestrates fetching from all sources and deduplicates by post ID.

    Arctic Shift is now the PRIMARY source since Reddit blocks VPS IPs.
    Reddit scraping is tried second as a bonus. Pullpush is last resort.
    """

    def __init__(self, user_agent: str = REDDIT_USER_AGENT):
        self.arctic = ArcticShiftFetcher()
        self.reddit = RedditSearchFetcher(user_agent=user_agent)
        self.pullpush = PullpushFetcher()

    def fetch_all(
        self,
        brand_config: dict,
        lookback_days: int = 90,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> list[RedditPost]:
        """
        Fetch posts for a brand from all sources, deduplicate, and return
        sorted by date (newest first).

        Reports per-source diagnostics via progress_callback so failures
        are visible in the Telegram chat.
        """
        all_posts: dict[str, RedditPost] = {}
        keywords = brand_config.get("keywords", [])
        subreddit_hints = brand_config.get("subreddit_hints", [])
        diagnostics: list[str] = []  # Track per-source results for reporting

        after_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).timestamp())
        after_date = (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # --- Source 1 (PRIMARY): Arctic Shift ---------------------------------
        # Always search subreddit_hints + default subs for broad coverage.
        if progress_callback:
            progress_callback("Searching Arctic Shift archive (primary)...")

        # Build the list of subreddits to search
        arctic_subs = list(subreddit_hints)
        for s in DEFAULT_SUBREDDITS:
            if s.lower() not in [x.lower() for x in arctic_subs]:
                arctic_subs.append(s)

        arctic_errors = 0
        for sub in arctic_subs:
            for kw in keywords:
                try:
                    posts = self.arctic.search_subreddit(
                        subreddit=sub, query=kw, after_date=after_date, max_pages=5)
                    new = 0
                    for p in posts:
                        if p.post_id not in all_posts:
                            all_posts[p.post_id] = p
                            new += 1
                    if posts:
                        logger.info(f"  Arctic Shift r/{sub} '{kw}': {len(posts)} raw, {new} new")
                except Exception as e:
                    arctic_errors += 1
                    logger.error(f"  Arctic Shift r/{sub} failed for '{kw}': {e}")

        arctic_count = len(all_posts)
        diag = f"Arctic Shift: {arctic_count} posts from {len(arctic_subs)} subs"
        if arctic_errors:
            diag += f" ({arctic_errors} errors)"
        diagnostics.append(diag)
        logger.info(f"Arctic Shift total: {arctic_count} unique posts ({arctic_errors} errors)")

        if progress_callback:
            progress_callback(f"Arctic Shift: {arctic_count} posts. Now trying Reddit...")

        # --- Source 2: Reddit search JSON (bonus, often blocked) ---------------
        for kw in keywords:
            try:
                posts = self.reddit.search(query=f'"{kw}"', sort="new",
                                           time_filter="year", max_pages=3)
                for p in posts:
                    if p.created_utc >= after_ts and p.post_id not in all_posts:
                        all_posts[p.post_id] = p
                logger.info(f"  Reddit '{kw}': {len(posts)} raw")
            except Exception as e:
                logger.error(f"  Reddit search failed for '{kw}': {e}")

        # Targeted subreddit searches on Reddit
        for sub in subreddit_hints:
            for kw in keywords:
                try:
                    posts = self.reddit.search_subreddit(
                        subreddit=sub, query=f'"{kw}"', sort="new",
                        time_filter="year", max_pages=3)
                    for p in posts:
                        if p.created_utc >= after_ts and p.post_id not in all_posts:
                            all_posts[p.post_id] = p
                except Exception as e:
                    logger.error(f"  r/{sub} search failed for '{kw}': {e}")

        reddit_added = len(all_posts) - arctic_count
        diagnostics.append(f"Reddit: +{reddit_added} new posts")
        logger.info(f"Reddit added: {reddit_added} new posts")

        # --- Source 3: Pullpush (fallback) ------------------------------------
        if progress_callback:
            progress_callback(f"Total so far: {len(all_posts)}. Trying Pullpush...")

        pre_pullpush = len(all_posts)
        for kw in keywords:
            try:
                posts = self.pullpush.search(query=kw, after_ts=after_ts, max_pages=5)
                for p in posts:
                    if p.post_id not in all_posts:
                        all_posts[p.post_id] = p
                logger.info(f"  Pullpush '{kw}': {len(posts)} raw")
            except Exception as e:
                logger.error(f"  Pullpush failed for '{kw}': {e}")

        pullpush_added = len(all_posts) - pre_pullpush
        diagnostics.append(f"Pullpush: +{pullpush_added} new posts")

        total = len(all_posts)
        logger.info(f"Grand total: {total} unique posts across all sources")

        if progress_callback:
            summary = " | ".join(diagnostics)
            progress_callback(f"Fetched {total} unique posts. [{summary}]")

        # Sort newest first
        return sorted(all_posts.values(), key=lambda p: p.created_utc, reverse=True)
