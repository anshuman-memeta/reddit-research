"""
Google Sheets writer for research output.

Creates a new spreadsheet per research run with:
  - "Research Data" sheet: full post-level data
  - "Summary" sheet: aggregated stats
"""

import logging
from collections import Counter
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

from config import GOOGLE_SHEETS_CREDS_FILE

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DATA_HEADERS = [
    "Date", "Subreddit", "Title", "Sentiment", "Theme",
    "Summary", "Upvotes", "Comments", "Competitor Mentions", "URL",
]


class SheetsWriter:
    """Creates and populates a Google Sheet with research results."""

    def __init__(self, creds_file: str = GOOGLE_SHEETS_CREDS_FILE):
        creds = Credentials.from_service_account_file(creds_file, scopes=SCOPES)
        self.client = gspread.authorize(creds)

    def create_research_sheet(self, brand_name: str, results: list[dict]) -> str:
        """
        Create a new Google Sheet and return its URL.

        Args:
            brand_name: Name of the brand researched.
            results: List of analyzed post dicts from BrandAnalyzer.

        Returns:
            URL of the created spreadsheet.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = f"{brand_name} - Reddit Research - {timestamp}"

        spreadsheet = self.client.create(title)

        # --- Data sheet ---------------------------------------------------
        ws = spreadsheet.sheet1
        ws.update_title("Research Data")

        ws.update("A1", [DATA_HEADERS])
        ws.format("A1:J1", {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
        })

        rows = []
        for r in results:
            rows.append([
                r["created_date"],
                f"r/{r['subreddit']}",
                r["title"],
                r["sentiment"].capitalize(),
                r["theme"],
                r["summary"],
                r["score"],
                r["num_comments"],
                r["competitor_mentions"],
                r["url"],
            ])

        if rows:
            ws.update(f"A2:J{len(rows) + 1}", rows)

        # Color-code sentiment cells
        color_map = {
            "positive": {"red": 0.85, "green": 0.95, "blue": 0.85},
            "negative": {"red": 0.95, "green": 0.85, "blue": 0.85},
            "neutral":  {"red": 0.93, "green": 0.93, "blue": 0.93},
        }
        for i, r in enumerate(results):
            c = color_map.get(r["sentiment"])
            if c:
                ws.format(f"D{i + 2}", {"backgroundColor": c})

        # --- Summary sheet ------------------------------------------------
        summary_ws = spreadsheet.add_worksheet("Summary", rows=30, cols=5)
        summary_data = self._build_summary(brand_name, results)
        summary_ws.update("A1", summary_data)
        summary_ws.format("A1:B1", {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
        })

        # Make viewable by anyone with the link
        spreadsheet.share("", perm_type="anyone", role="reader")

        logger.info(f"Created Google Sheet: {spreadsheet.url}")
        return spreadsheet.url

    @staticmethod
    def _build_summary(brand_name: str, results: list[dict]) -> list[list]:
        total = len(results)
        sentiments = Counter(r["sentiment"] for r in results)
        subreddits = Counter(r["subreddit"] for r in results)
        themes = Counter(r["theme"] for r in results)

        rows = [
            ["Metric", "Value"],
            ["Brand", brand_name],
            ["Total Relevant Posts", total],
            ["Positive", sentiments.get("positive", 0)],
            ["Negative", sentiments.get("negative", 0)],
            ["Neutral", sentiments.get("neutral", 0)],
            [],
            ["Top Subreddits", "Count"],
        ]
        for sub, count in subreddits.most_common(10):
            rows.append([f"r/{sub}", count])

        rows.append([])
        rows.append(["Top Themes", "Count"])
        for theme, count in themes.most_common(10):
            rows.append([theme, count])

        return rows
