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
GROQ_MODEL = "llama-3.1-8b-instant"

# Multi-provider LLM fallback chain (tried in order)
# Each provider uses the OpenAI-compatible chat completions API.
# Only providers with a configured API key are active.
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

LLM_PROVIDERS = [
    {
        "name": "Groq",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": GROQ_API_KEY,
        "model": GROQ_MODEL,
    },
    {
        "name": "Cerebras",
        "api_url": "https://api.cerebras.ai/v1/chat/completions",
        "api_key": CEREBRAS_API_KEY,
        "model": "llama-3.3-70b",
    },
    {
        "name": "SambaNova",
        "api_url": "https://api.sambanova.ai/v1/chat/completions",
        "api_key": SAMBANOVA_API_KEY,
        "model": "Meta-Llama-3.3-70B-Instruct",
    },
    {
        "name": "Mistral",
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "api_key": MISTRAL_API_KEY,
        "model": "mistral-small-latest",
    },
]

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

# Comment fetching (detailed mode)
COMMENT_SCORE_THRESHOLD = 20    # only fetch comments for posts with this many upvotes
MAX_COMMENTS_PER_POST = 3       # top comments to fetch per post
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")

# OAuth2 user credentials
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
GOOGLE_OAUTH_REFRESH_TOKEN = os.getenv("GOOGLE_OAUTH_REFRESH_TOKEN", "")
