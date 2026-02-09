"""
Configuration for the Reddit Research Bot.
All secrets loaded from environment variables.
"""

import os
import json

# Telegram Bot
TELEGRAM_BOT_TOKEN = os.getenv("RESEARCH_BOT_TOKEN")
AUTHORIZED_USERS = json.loads(os.getenv("AUTHORIZED_USERS", "[]"))

# Groq API (LLM for relevance + sentiment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Google Sheets
GOOGLE_SHEETS_CREDS_FILE = os.getenv(
    "GOOGLE_SHEETS_CREDS_FILE",
    os.path.join(os.path.dirname(__file__), "credentials.json"),
)

# Reddit search settings
REDDIT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
SEARCH_LOOKBACK_DAYS = 90
REDDIT_RATE_LIMIT_DELAY = 2.0  # seconds between requests

# Brands config path
BRANDS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "brands.json")

# Chart styling
SENTIMENT_COLORS = {
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#9E9E9E",
}

CHART_OUTPUT_DIR = "/tmp/research_charts"
