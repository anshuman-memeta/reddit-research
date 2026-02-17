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

from config import (
    REDDIT_RATE_LIMIT_DELAY,
    REDDIT_USER_AGENT,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
)

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
    Fetches posts via Reddit search JSON endpoints.
    No authentication required.  Paginated via `after` token.
    Falls back from www.reddit.com → old.reddit.com on persistent 403s.
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
        # Track which endpoint works; start with first
        self._working_endpoint_idx = 0

    def _get_with_retry(self, url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        """GET with retry on 403/429 errors. Falls back to old.reddit.com on persistent 403."""
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 3 * (attempt + 1)
                    logger.warning(f"Reddit 429 rate-limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 403:
                    logger.warning(f"Reddit 403 on {url} (attempt {attempt + 1})")
                    # Try next endpoint on second failure
                    if attempt >= 1:
                        return None  # Signal caller to try fallback endpoint
                    time.sleep(3 * (attempt + 1))
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
        for i, base in enumerate(endpoints):
            url = f"{base}{path}"
            data = self._get_with_retry(url, params)
            if data is not None:
                # Remember which endpoint worked
                self._working_endpoint_idx = self.ENDPOINTS.index(base)
                return data
            logger.warning(f"Endpoint {base} failed for {path}, trying next...")
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
# Source 1b: Reddit OAuth API (reliable, works from datacenter IPs)
# ---------------------------------------------------------------------------

class RedditOAuthFetcher:
    """
    Fetches posts via Reddit's official OAuth API (oauth.reddit.com).
    Uses the 'Application Only' flow (client_credentials) — no user login needed.
    This is the reliable method for datacenter/server IPs where www.reddit.com
    returns 403.

    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET from a Reddit 'script' app.
    Create one at: https://www.reddit.com/prefs/apps
    """

    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    API_BASE = "https://oauth.reddit.com"

    def __init__(self, client_id: str, client_secret: str,
                 user_agent: str = "ResearchBot/1.0", rate_limit: float = 1.0):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.rate_limit = rate_limit
        self._token: Optional[str] = None
        self._token_expires: float = 0

    def _ensure_token(self):
        """Obtain or refresh the OAuth2 bearer token."""
        if self._token and time.time() < self._token_expires - 60:
            return
        try:
            resp = requests.post(
                self.TOKEN_URL,
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": self.user_agent},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._token_expires = time.time() + data.get("expires_in", 3600)
            logger.info("Reddit OAuth token obtained")
        except Exception as e:
            logger.error(f"Reddit OAuth token error: {e}")
            raise

    def _get(self, path: str, params: dict) -> Optional[dict]:
        """Authenticated GET to oauth.reddit.com."""
        self._ensure_token()
        url = f"{self.API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": self.user_agent,
        }
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                if resp.status_code == 401:
                    # Token expired mid-request, refresh once
                    self._token = None
                    self._ensure_token()
                    headers["Authorization"] = f"Bearer {self._token}"
                    continue
                if resp.status_code == 429:
                    wait = 3 * (attempt + 1)
                    logger.warning(f"Reddit OAuth 429, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise
        return None

    def _parse_posts(self, data: Optional[dict]) -> tuple[list[RedditPost], Optional[str]]:
        """Parse listing response into posts + after token."""
        if not data:
            return [], None
        children = data.get("data", {}).get("children", [])
        posts = []
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
        return posts, after

    def search(self, query: str, sort: str = "new", time_filter: str = "year",
               limit: int = 100, max_pages: int = 10) -> list[RedditPost]:
        """Global search via OAuth API."""
        all_posts: list[RedditPost] = []
        after: Optional[str] = None

        for page in range(max_pages):
            params = {"q": query, "sort": sort, "t": time_filter,
                      "limit": limit, "type": "link"}
            if after:
                params["after"] = after

            data = self._get("/search", params)
            posts, after = self._parse_posts(data)
            all_posts.extend(posts)

            if not posts or not after:
                break
            time.sleep(self.rate_limit)

        return all_posts

    def search_subreddit(self, subreddit: str, query: str, sort: str = "new",
                         time_filter: str = "year", limit: int = 100,
                         max_pages: int = 5) -> list[RedditPost]:
        """Search within a specific subreddit via OAuth API."""
        all_posts: list[RedditPost] = []
        after: Optional[str] = None

        for page in range(max_pages):
            params = {"q": query, "sort": sort, "t": time_filter,
                      "limit": limit, "restrict_sr": "on", "type": "link"}
            if after:
                params["after"] = after

            data = self._get(f"/r/{subreddit}/search", params)
            posts, after = self._parse_posts(data)
            all_posts.extend(posts)

            if not posts or not after:
                break
            time.sleep(self.rate_limit)

        return all_posts


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
    Orchestrates fetching from all sources and deduplicates by post ID.
    Uses Reddit OAuth API (reliable) if credentials are configured,
    falls back to scraping www/old.reddit.com otherwise.
    """

    def __init__(self, user_agent: str = REDDIT_USER_AGENT):
        self.reddit_scraper = RedditSearchFetcher(user_agent=user_agent)
        # Use OAuth if configured (works from datacenter IPs)
        self.reddit_oauth: Optional[RedditOAuthFetcher] = None
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            self.reddit_oauth = RedditOAuthFetcher(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
            )
            logger.info("Reddit OAuth credentials found — will use OAuth API")
        else:
            logger.warning("No Reddit OAuth credentials — falling back to scraping (may get 403)")
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

        # --- Source 1: Reddit (OAuth preferred, scraping fallback) ---------
        if progress_callback:
            progress_callback("Searching Reddit...")

        # Pick the best available Reddit source
        reddit = self.reddit_oauth or self.reddit_scraper
        source_name = "OAuth" if self.reddit_oauth else "scraper"

        for kw in keywords:
            try:
                posts = reddit.search(query=f'"{kw}"', sort="new",
                                      time_filter="year", max_pages=10)
                for p in posts:
                    if p.created_utc >= after_ts:
                        all_posts[p.post_id] = p
                logger.info(f"  Reddit({source_name}) '{kw}': {len(posts)} raw, {len(all_posts)} unique total")
            except Exception as e:
                logger.error(f"  Reddit({source_name}) search failed for '{kw}': {e}")

        # Targeted subreddit searches
        for sub in subreddit_hints:
            for kw in keywords:
                try:
                    posts = reddit.search_subreddit(
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
        # Arctic Shift requires a subreddit filter for full-text search.
        # Always include common subs when Reddit returned 0, to maximize coverage.
        if progress_callback:
            progress_callback(f"Found {reddit_count} from Reddit. Checking Arctic Shift...")

        COMMON_SUBS = [
            "india", "IndianGaming", "IndianConsumer",
            "gadgets", "technology", "BuyItForLife",
        ]
        arctic_subs = list(subreddit_hints)
        if reddit_count == 0:
            # Add common subs that aren't already in hints
            for s in COMMON_SUBS:
                if s.lower() not in [x.lower() for x in arctic_subs]:
                    arctic_subs.append(s)
            logger.info(f"Reddit returned 0 — expanded Arctic Shift to {len(arctic_subs)} subs")

        after_date = (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
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
