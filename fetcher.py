"""
Multi-source Reddit post fetcher.

Sources (in priority order):
  1. Arctic Shift API (primary — reliable from datacenter IPs)
  2. Reddit search JSON endpoint (secondary — often 403 from VPS IPs)
  3. Reddit RSS feeds (alternate — sometimes bypasses 403 blocking)
  4. Pullpush.io API (fallback, unreliable)

All sources are deduplicated by post ID before returning.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Optional

import requests

from config import REDDIT_RATE_LIMIT_DELAY, REDDIT_USER_AGENT, MAX_COMMENTS_PER_POST

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


@dataclass
class RedditComment:
    """A single Reddit comment."""

    comment_id: str
    post_id: str
    body: str
    author: str
    score: int
    created_utc: float


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
        self.session.headers.update({
            "User-Agent": REDDIT_USER_AGENT,
            "Accept": "application/json",
        })
        self.rate_limit = rate_limit
        self._reachable: Optional[bool] = None  # cached connectivity result

    def check_connectivity(self, timeout: float = 15) -> bool:
        """Quick connectivity test — returns True if Arctic Shift responds."""
        if self._reachable is not None:
            return self._reachable
        for attempt in range(3):
            try:
                resp = self.session.get(
                    self.BASE_URL,
                    params={"query": "test", "subreddit": "all", "limit": 1},
                    timeout=timeout,
                )
                resp.raise_for_status()
                self._reachable = True
                return True
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    logger.warning(f"Arctic Shift connectivity check attempt {attempt + 1} failed: {e}")
                else:
                    logger.error(f"Arctic Shift unreachable after 3 connectivity checks: {e}")
        self._reachable = False
        return False

    def search_subreddit(self, subreddit: str, query: str,
                         after_date: Optional[str] = None,
                         limit: int = 100,
                         max_pages: int = 10,
                         max_retries: int = 3) -> list[RedditPost]:
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
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    resp = self.session.get(self.BASE_URL, params=params, timeout=45)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = 2 * (attempt + 1)
                        logger.warning(f"Arctic Shift r/{subreddit} attempt {attempt + 1} failed: {e}, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"Arctic Shift error (r/{subreddit}, page {page}): {e}")

            if data is None:
                # Return what we have so far instead of raising — let the caller
                # handle partial results and decide whether to retry this sub.
                if last_error:
                    logger.warning(f"Arctic Shift r/{subreddit} page {page} failed, returning {len(posts)} posts collected so far")
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
# Source 3: Reddit RSS feeds (sometimes bypasses JSON 403 blocking)
# ---------------------------------------------------------------------------

class RedditRSSFetcher:
    """
    Fetches posts from Reddit's RSS/Atom feeds.
    RSS endpoints are sometimes accessible even when JSON search returns 403.
    Limited to ~25 results per request (no deep pagination), but useful as
    a supplementary source.
    """

    ENDPOINTS = [
        "https://www.reddit.com",
        "https://old.reddit.com",
    ]

    def __init__(self, user_agent: str = REDDIT_USER_AGENT, rate_limit: float = REDDIT_RATE_LIMIT_DELAY):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.rate_limit = rate_limit

    def _extract_post_id(self, link: str) -> str:
        """Extract post ID from a Reddit URL like /r/sub/comments/ID/..."""
        match = re.search(r"/comments/([a-z0-9]+)", link)
        return match.group(1) if match else ""

    def _parse_rss(self, content: str) -> list[RedditPost]:
        """Parse Reddit RSS XML into RedditPost objects."""
        posts = []
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.warning(f"RSS XML parse error: {e}")
            return posts

        # Handle both RSS 2.0 (<item>) and Atom (<entry>) formats
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Try Atom format first (Reddit typically returns Atom)
        entries = root.findall(".//atom:entry", ns)
        if entries:
            for entry in entries:
                title = entry.findtext("atom:title", "", ns)
                link = entry.findtext("atom:link[@href]", "", ns)
                # Get href from link element
                link_elem = entry.find("atom:link", ns)
                if link_elem is not None:
                    link = link_elem.get("href", "")
                content_elem = entry.find("atom:content", ns)
                selftext = content_elem.text if content_elem is not None and content_elem.text else ""
                updated = entry.findtext("atom:updated", "", ns)
                author_elem = entry.find("atom:author/atom:name", ns)
                author = author_elem.text if author_elem is not None else "[unknown]"
                # Strip /u/ prefix from author
                if author.startswith("/u/"):
                    author = author[3:]

                post_id = self._extract_post_id(link)
                if not post_id:
                    continue

                # Parse date
                created_utc = 0.0
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        created_utc = dt.timestamp()
                    except (ValueError, TypeError):
                        pass

                # Extract subreddit from link
                sub_match = re.search(r"/r/([^/]+)", link)
                subreddit = sub_match.group(1) if sub_match else ""

                posts.append(RedditPost(
                    post_id=post_id,
                    title=title,
                    selftext=selftext,
                    subreddit=subreddit,
                    author=author,
                    url=link,
                    permalink=link,
                    score=0,  # RSS doesn't include score
                    num_comments=0,
                    created_utc=created_utc,
                ))
            return posts

        # Fallback: RSS 2.0 format (<item>)
        for item in root.findall(".//item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            selftext = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")

            post_id = self._extract_post_id(link)
            if not post_id:
                continue

            created_utc = 0.0
            if pub_date:
                try:
                    dt = parsedate_to_datetime(pub_date)
                    created_utc = dt.timestamp()
                except (ValueError, TypeError):
                    pass

            sub_match = re.search(r"/r/([^/]+)", link)
            subreddit = sub_match.group(1) if sub_match else ""

            posts.append(RedditPost(
                post_id=post_id,
                title=title,
                selftext=selftext or "",
                subreddit=subreddit,
                author="[rss]",
                url=link,
                permalink=link,
                score=0,
                num_comments=0,
                created_utc=created_utc,
            ))

        return posts

    def search(self, query: str, max_retries: int = 2) -> list[RedditPost]:
        """Global search via RSS. Returns up to ~25 results."""
        path = f"/search.rss?q={requests.utils.quote(query)}&sort=new&t=year"
        for base in self.ENDPOINTS:
            url = f"{base}{path}"
            for attempt in range(max_retries):
                try:
                    resp = self.session.get(url, timeout=20)
                    if resp.status_code in (403, 429):
                        logger.warning(f"RSS {resp.status_code} on {url}")
                        break  # Try next endpoint
                    resp.raise_for_status()
                    return self._parse_rss(resp.text)
                except requests.exceptions.HTTPError:
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        logger.warning(f"RSS search error: {e}")
            time.sleep(self.rate_limit)
        return []

    def search_subreddit(self, subreddit: str, query: str, max_retries: int = 2) -> list[RedditPost]:
        """Search within a subreddit via RSS. Returns up to ~25 results."""
        path = f"/r/{subreddit}/search.rss?q={requests.utils.quote(query)}&restrict_sr=on&sort=new&t=year"
        for base in self.ENDPOINTS:
            url = f"{base}{path}"
            for attempt in range(max_retries):
                try:
                    resp = self.session.get(url, timeout=20)
                    if resp.status_code in (403, 429):
                        logger.warning(f"RSS {resp.status_code} on {url}")
                        break
                    resp.raise_for_status()
                    return self._parse_rss(resp.text)
                except requests.exceptions.HTTPError:
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        logger.warning(f"RSS r/{subreddit} search error: {e}")
            time.sleep(self.rate_limit)
        return []


# ---------------------------------------------------------------------------
# Source 4: Pullpush.io (unreliable fallback)
# ---------------------------------------------------------------------------

class PullpushFetcher:
    """Fetches posts from Pullpush.io (Pushshift mirror). May be down."""

    BASE_URL = "https://api.pullpush.io/reddit/search/submission"

    def __init__(self, rate_limit: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": REDDIT_USER_AGENT,
            "Accept": "application/json",
        })
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
# Comment fetcher (used in detailed mode)
# ---------------------------------------------------------------------------

class CommentFetcher:
    """Fetches top comments for specific Reddit posts using the JSON API."""

    ENDPOINTS = [
        "https://old.reddit.com",
        "https://www.reddit.com",
    ]

    def __init__(self, rate_limit: float = REDDIT_RATE_LIMIT_DELAY):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": REDDIT_USER_AGENT,
            "Accept": "application/json",
        })
        self.rate_limit = rate_limit

    def fetch_comments(self, subreddit: str, post_id: str,
                       limit: int = MAX_COMMENTS_PER_POST) -> list[RedditComment]:
        """Fetch top-level comments for a single post, sorted by score."""
        path = f"/r/{subreddit}/comments/{post_id}.json"
        params = {"sort": "top", "limit": limit, "depth": 1}

        for base in self.ENDPOINTS:
            url = f"{base}{path}"
            try:
                resp = self.session.get(url, params=params, timeout=20)
                if resp.status_code in (403, 429):
                    continue  # try next endpoint
                resp.raise_for_status()
                data = resp.json()
                return self._parse_comments(data, post_id)
            except Exception as e:
                logger.warning(f"Comment fetch failed for {post_id} via {base}: {e}")
        return []

    @staticmethod
    def _parse_comments(data, post_id: str) -> list[RedditComment]:
        """Parse Reddit JSON response into RedditComment objects."""
        comments: list[RedditComment] = []
        if not isinstance(data, list) or len(data) < 2:
            return comments

        children = data[1].get("data", {}).get("children", [])
        for child in children:
            if child.get("kind") != "t1":
                continue
            d = child.get("data", {})
            body = d.get("body", "")
            if not body or body in ("[deleted]", "[removed]"):
                continue
            comments.append(RedditComment(
                comment_id=d.get("id", ""),
                post_id=post_id,
                body=body[:500],
                author=d.get("author", "[deleted]"),
                score=d.get("score", 0),
                created_utc=d.get("created_utc", 0),
            ))
        return comments

    def fetch_comments_batch(
        self,
        posts: list[dict],
        limit: int = MAX_COMMENTS_PER_POST,
        progress_callback: Optional[Callable] = None,
    ) -> dict[str, list[RedditComment]]:
        """Fetch comments for multiple posts with rate limiting.

        Args:
            posts: list of result dicts (must have 'post_id' and 'subreddit')
            limit: max comments per post
            progress_callback: called with (done, total) after every 10 posts

        Returns:
            {post_id: [RedditComment, ...]}
        """
        all_comments: dict[str, list[RedditComment]] = {}
        total = len(posts)

        for i, post in enumerate(posts):
            pid = post["post_id"]
            sub = post["subreddit"]
            comments = self.fetch_comments(sub, pid, limit=limit)
            all_comments[pid] = comments

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total)

            if i < total - 1:
                time.sleep(self.rate_limit)

        return all_comments


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
        self.rss = RedditRSSFetcher(user_agent=user_agent)
        self.pullpush = PullpushFetcher()
        self.errors: list[str] = []  # Collects error details for diagnostics

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
        self.errors = []  # Reset error log

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

        # Quick connectivity check before committing to the full search loop.
        # This saves minutes of timeouts if Arctic Shift is unreachable.
        arctic_errors = 0
        arctic_reachable = self.arctic.check_connectivity()
        if not arctic_reachable:
            logger.warning("Arctic Shift failed connectivity check, skipping entirely")
            if progress_callback:
                progress_callback("Arctic Shift unreachable (failed connectivity check), skipping to other sources...")
            self.errors.append("Arctic Shift: unreachable (failed connectivity pre-check)")
        else:
            consecutive_sub_failures = 0  # count per-subreddit, not per-keyword
            last_arctic_error = ""
            for sub in arctic_subs:
                # If Arctic Shift is consistently failing across subreddits, skip the rest
                if consecutive_sub_failures >= 5:
                    remaining = len(arctic_subs) - arctic_subs.index(sub)
                    logger.warning(f"Arctic Shift: 5 consecutive subreddit failures, skipping {remaining} remaining subs")
                    if progress_callback:
                        progress_callback(f"Arctic Shift failing ({last_arctic_error}), skipping {remaining} remaining subs...")
                    break

                sub_had_success = False
                for kw in keywords:
                    try:
                        posts = self.arctic.search_subreddit(
                            subreddit=sub, query=kw, after_date=after_date, max_pages=5)
                        sub_had_success = True  # at least one keyword succeeded for this sub
                        new = 0
                        for p in posts:
                            if p.post_id not in all_posts:
                                all_posts[p.post_id] = p
                                new += 1
                        if posts:
                            logger.info(f"  Arctic Shift r/{sub} '{kw}': {len(posts)} raw, {new} new")
                    except Exception as e:
                        arctic_errors += 1
                        last_arctic_error = str(e)
                        self.errors.append(f"Arctic Shift r/{sub} '{kw}': {e}")
                        logger.error(f"  Arctic Shift r/{sub} failed for '{kw}': {e}")

                if sub_had_success:
                    consecutive_sub_failures = 0
                else:
                    consecutive_sub_failures += 1
                    # Brief pause before trying next subreddit to let transient issues clear
                    time.sleep(2)

        arctic_count = len(all_posts)
        diag = f"Arctic Shift: {arctic_count} posts from {len(arctic_subs)} subs"
        if arctic_errors:
            diag += f" ({arctic_errors} errors)"
        diagnostics.append(diag)
        logger.info(f"Arctic Shift total: {arctic_count} unique posts ({arctic_errors} errors)")

        if progress_callback:
            progress_callback(f"Arctic Shift: {arctic_count} posts. Now trying Reddit...")

        # --- Source 2: Reddit search JSON (bonus, often blocked) ---------------
        reddit_errors = 0
        for kw in keywords:
            try:
                posts = self.reddit.search(query=f'"{kw}"', sort="new",
                                           time_filter="year", max_pages=3)
                for p in posts:
                    if p.created_utc >= after_ts and p.post_id not in all_posts:
                        all_posts[p.post_id] = p
                logger.info(f"  Reddit '{kw}': {len(posts)} raw")
            except Exception as e:
                reddit_errors += 1
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
                    reddit_errors += 1
                    logger.error(f"  r/{sub} search failed for '{kw}': {e}")

        reddit_added = len(all_posts) - arctic_count
        diag = f"Reddit JSON: +{reddit_added} new posts"
        if reddit_errors:
            diag += f" ({reddit_errors} errors)"
        diagnostics.append(diag)
        logger.info(f"Reddit JSON added: {reddit_added} new posts ({reddit_errors} errors)")

        # --- Source 3: Reddit RSS feeds (alternate, bypasses some 403s) --------
        pre_rss = len(all_posts)
        rss_errors = 0
        for kw in keywords:
            try:
                posts = self.rss.search(query=kw)
                for p in posts:
                    if p.created_utc >= after_ts and p.post_id and p.post_id not in all_posts:
                        all_posts[p.post_id] = p
                logger.info(f"  RSS '{kw}': {len(posts)} raw")
            except Exception as e:
                rss_errors += 1
                logger.error(f"  RSS search failed for '{kw}': {e}")

        # Targeted subreddit RSS searches
        for sub in subreddit_hints[:5]:  # limit to avoid excessive requests
            for kw in keywords:
                try:
                    posts = self.rss.search_subreddit(subreddit=sub, query=kw)
                    for p in posts:
                        if p.created_utc >= after_ts and p.post_id and p.post_id not in all_posts:
                            all_posts[p.post_id] = p
                except Exception as e:
                    rss_errors += 1
                    logger.error(f"  RSS r/{sub} search failed for '{kw}': {e}")

        rss_added = len(all_posts) - pre_rss
        diag = f"RSS: +{rss_added} new posts"
        if rss_errors:
            diag += f" ({rss_errors} errors)"
        diagnostics.append(diag)
        logger.info(f"Reddit RSS added: {rss_added} new posts ({rss_errors} errors)")

        # --- Source 4: Pullpush (fallback) ------------------------------------
        if progress_callback:
            progress_callback(f"Total so far: {len(all_posts)}. Trying Pullpush...")

        pre_pullpush = len(all_posts)
        pullpush_errors = 0
        for kw in keywords:
            try:
                posts = self.pullpush.search(query=kw, after_ts=after_ts, max_pages=5)
                for p in posts:
                    if p.post_id not in all_posts:
                        all_posts[p.post_id] = p
                logger.info(f"  Pullpush '{kw}': {len(posts)} raw")
            except Exception as e:
                pullpush_errors += 1
                logger.error(f"  Pullpush failed for '{kw}': {e}")

        pullpush_added = len(all_posts) - pre_pullpush
        diag = f"Pullpush: +{pullpush_added} new posts"
        if pullpush_errors:
            diag += f" ({pullpush_errors} errors)"
        diagnostics.append(diag)

        total = len(all_posts)
        logger.info(f"Grand total: {total} unique posts across all sources")

        if progress_callback:
            summary = " | ".join(diagnostics)
            progress_callback(f"Fetched {total} unique posts. [{summary}]")

        # Sort newest first
        return sorted(all_posts.values(), key=lambda p: p.created_utc, reverse=True)
