"""
Multi-source Reddit post fetcher.

Sources (in priority order):
  1. Reddit search JSON endpoint (primary, free, no auth)
  2. Arctic Shift API (secondary archive)
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
# Source 1: Reddit's own search JSON endpoint
# ---------------------------------------------------------------------------

class RedditSearchFetcher:
    """
    Fetches posts via https://www.reddit.com/search.json
    No authentication required.  Paginated via `after` token.
    """

    BASE_URL = "https://www.reddit.com/search.json"

    def __init__(self, user_agent: str = REDDIT_USER_AGENT, rate_limit: float = REDDIT_RATE_LIMIT_DELAY):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.rate_limit = rate_limit

    def _get_with_retry(self, url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        """GET with retry on 403/429 errors."""
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code in (403, 429):
                    wait = 3 * (attempt + 1)
                    logger.warning(f"Reddit {resp.status_code}, retrying in {wait}s...")
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

    def search(self, query: str, sort: str = "new", time_filter: str = "year",
               limit: int = 100, max_pages: int = 10) -> list[RedditPost]:
        """Global Reddit search with pagination."""
        posts: list[RedditPost] = []
        after: Optional[str] = None

        for page in range(max_pages):
            params = {
                "q": query,
                "sort": sort,
                "t": time_filter,
                "limit": limit,
                "type": "link",
            }
            if after:
                params["after"] = after

            try:
                data = self._get_with_retry(self.BASE_URL, params)
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
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        posts: list[RedditPost] = []
        after: Optional[str] = None

        for page in range(max_pages):
            params = {
                "q": query,
                "sort": sort,
                "t": time_filter,
                "limit": limit,
                "restrict_sr": "on",
                "type": "link",
            }
            if after:
                params["after"] = after

            try:
                data = self._get_with_retry(url, params)
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
# Source 2: Arctic Shift API (community Pushshift replacement)
# ---------------------------------------------------------------------------

class ArcticShiftFetcher:
    """
    Fetches posts from the Arctic Shift Reddit archive.
    NOTE: Full-text search requires a subreddit or author filter.
    """

    BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

    def __init__(self, rate_limit: float = 1.0):
        self.session = requests.Session()
        self.rate_limit = rate_limit

    def search_subreddit(self, subreddit: str, query: str,
                         after_date: Optional[str] = None,
                         limit: int = 100,
                         max_pages: int = 10) -> list[RedditPost]:
        """
        Search within a specific subreddit. Arctic Shift requires
        subreddit or author for full-text search.

        Args:
            subreddit: Subreddit name (without r/).
            query: Search query.
            after_date: Date string like "2024-11-09". Defaults to 90 days ago.
            limit: Results per page (max 100).
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

            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Arctic Shift error (r/{subreddit}, page {page}): {e}")
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
    Orchestrates fetching from all three sources and deduplicates by post ID.
    Sends progress updates via an optional callback.
    """

    def __init__(self, user_agent: str = REDDIT_USER_AGENT):
        self.reddit = RedditSearchFetcher(user_agent=user_agent)
        self.arctic = ArcticShiftFetcher()
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
        """
        all_posts: dict[str, RedditPost] = {}
        keywords = brand_config.get("keywords", [])
        subreddit_hints = brand_config.get("subreddit_hints", [])

        after_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).timestamp())

        # --- Source 1: Reddit search JSON (primary) -----------------------
        if progress_callback:
            progress_callback("Searching Reddit...")

        for kw in keywords:
            try:
                posts = self.reddit.search(query=f'"{kw}"', sort="new",
                                           time_filter="year", max_pages=10)
                for p in posts:
                    if p.created_utc >= after_ts:
                        all_posts[p.post_id] = p
                logger.info(f"  Reddit '{kw}': {len(posts)} raw, {len(all_posts)} unique total")
            except Exception as e:
                logger.error(f"  Reddit search failed for '{kw}': {e}")

        # Targeted subreddit searches
        for sub in subreddit_hints:
            for kw in keywords:
                try:
                    posts = self.reddit.search_subreddit(
                        subreddit=sub, query=f'"{kw}"', sort="new",
                        time_filter="year", max_pages=3)
                    for p in posts:
                        if p.created_utc >= after_ts:
                            all_posts[p.post_id] = p
                    logger.info(f"  r/{sub} '{kw}': {len(posts)} posts")
                except Exception as e:
                    logger.error(f"  r/{sub} search failed for '{kw}': {e}")

        reddit_count = len(all_posts)
        logger.info(f"Reddit total: {reddit_count} unique posts")

        # --- Source 2: Arctic Shift (secondary, subreddit-scoped) ----------
        # Arctic Shift only supports full-text search within a subreddit
        if progress_callback:
            progress_callback(f"Found {reddit_count} from Reddit. Checking Arctic Shift...")

        after_date = (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        for sub in subreddit_hints:
            for kw in keywords:
                try:
                    posts = self.arctic.search_subreddit(
                        subreddit=sub, query=kw, after_date=after_date, max_pages=5)
                    new = 0
                    for p in posts:
                        if p.post_id not in all_posts:
                            all_posts[p.post_id] = p
                            new += 1
                    logger.info(f"  Arctic Shift r/{sub} '{kw}': {len(posts)} raw, {new} new")
                except Exception as e:
                    logger.error(f"  Arctic Shift r/{sub} failed for '{kw}': {e}")

        arctic_total = len(all_posts) - reddit_count
        logger.info(f"Arctic Shift added: {arctic_total} new posts")

        # --- Source 3: Pullpush (fallback) --------------------------------
        if progress_callback:
            progress_callback(f"Total so far: {len(all_posts)}. Trying Pullpush...")

        for kw in keywords:
            try:
                posts = self.pullpush.search(query=kw, after_ts=after_ts, max_pages=10)
                new = 0
                for p in posts:
                    if p.post_id not in all_posts:
                        all_posts[p.post_id] = p
                        new += 1
                logger.info(f"  Pullpush '{kw}': {len(posts)} raw, {new} new")
            except Exception as e:
                logger.error(f"  Pullpush failed for '{kw}': {e}")

        total = len(all_posts)
        logger.info(f"Grand total: {total} unique posts across all sources")

        if progress_callback:
            progress_callback(f"Fetched {total} unique posts from all sources.")

        # Sort newest first
        return sorted(all_posts.values(), key=lambda p: p.created_utc, reverse=True)
