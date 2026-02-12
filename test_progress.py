"""Tests for progress callback and file logging changes."""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from fetcher import RedditPost
from analyzer import BrandAnalyzer


def _make_post(i: int) -> RedditPost:
    return RedditPost(
        post_id=f"post_{i}",
        title=f"Test post about BrandX product #{i}",
        selftext="I love the BrandX sunscreen, it's great for oily skin.",
        subreddit="SkincareAddiction",
        author=f"user_{i}",
        url=f"https://reddit.com/r/SkincareAddiction/comments/abc{i}",
        permalink=f"/r/SkincareAddiction/comments/abc{i}",
        score=10,
        num_comments=5,
        created_utc=datetime(2025, 12, 1, tzinfo=timezone.utc).timestamp(),
    )


BRAND_CONFIG = {
    "name": "BrandX",
    "description": "A skincare brand",
    "category": "beauty",
    "keywords": ["BrandX"],
    "product_terms": ["sunscreen", "moisturizer"],
    "competitors": ["Minimalist"],
    "subreddit_hints": ["SkincareAddiction"],
}


class TestProgressCallback:
    """Verify process_posts fires progress_callback at the right intervals."""

    def test_callback_fires_every_25_posts(self):
        """With 60 posts, callback should fire at post 25 and 50."""
        analyzer = BrandAnalyzer(api_key=None)  # No API key = keyword fallback
        posts = [_make_post(i) for i in range(60)]

        fired_at = []

        def callback(done, total, relevant):
            fired_at.append((done, total, relevant))

        # Patch sleep so test runs fast
        with patch("analyzer.time.sleep"):
            analyzer.process_posts(posts, BRAND_CONFIG, progress_callback=callback)

        # Should fire at 25 and 50 (every 25), not at 10, 20, 30... (old every-10)
        assert len(fired_at) == 2, f"Expected 2 callbacks, got {len(fired_at)}: {fired_at}"
        assert fired_at[0][0] == 25
        assert fired_at[1][0] == 50

    def test_callback_not_fired_under_25_posts(self):
        """With fewer than 25 posts, callback should never fire."""
        analyzer = BrandAnalyzer(api_key=None)
        posts = [_make_post(i) for i in range(20)]

        fired = []

        with patch("analyzer.time.sleep"):
            analyzer.process_posts(posts, BRAND_CONFIG, progress_callback=lambda d, t, r: fired.append(d))

        assert fired == [], f"Callback fired unexpectedly: {fired}"

    def test_callback_receives_correct_relevant_count(self):
        """The 'relevant' arg should reflect how many posts passed filtering."""
        analyzer = BrandAnalyzer(api_key=None)
        posts = [_make_post(i) for i in range(30)]

        fired_at = []

        def callback(done, total, relevant):
            fired_at.append((done, total, relevant))

        with patch("analyzer.time.sleep"):
            results = analyzer.process_posts(posts, BRAND_CONFIG, progress_callback=callback)

        assert len(fired_at) == 1
        assert fired_at[0][0] == 25  # done
        assert fired_at[0][1] == 30  # total
        # relevant count at post 25 should be <= 25 and > 0 (keyword fallback matches all)
        assert fired_at[0][2] > 0
        assert fired_at[0][2] <= 25


class TestProgressCallbackAsync:
    """Verify the async bridge works (run_coroutine_threadsafe pattern)."""

    def test_callback_from_thread_sends_message(self):
        """Simulate the bot.py pattern: callback in thread -> async message."""
        messages_sent = []

        async def fake_send(chat_id, text):
            messages_sent.append(text)

        async def run():
            loop = asyncio.get_running_loop()

            def progress_callback(done, total, relevant):
                future = asyncio.run_coroutine_threadsafe(
                    fake_send("chat123", f"Analyzed {done}/{total} posts ({relevant} relevant so far)..."),
                    loop,
                )
                future.result(timeout=5)  # Wait for it to actually execute

            # Run callback from a thread, just like asyncio.to_thread would
            await asyncio.to_thread(progress_callback, 25, 100, 5)

            # Give event loop a moment to process
            await asyncio.sleep(0.1)

        asyncio.run(run())

        assert len(messages_sent) == 1
        assert "25/100" in messages_sent[0]
        assert "5 relevant" in messages_sent[0]
