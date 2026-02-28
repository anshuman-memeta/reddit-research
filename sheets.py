"""
Google Sheets writer — OAuth2 user credentials (preferred) or service account fallback.
"""
import csv, json, logging, os, time
from collections import Counter
from datetime import datetime
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials as SACredentials
from config import (CHART_OUTPUT_DIR, GOOGLE_DRIVE_FOLDER_ID, GOOGLE_OAUTH_CLIENT_ID,
    GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REFRESH_TOKEN, GOOGLE_SHEETS_CREDS_FILE)

logger = logging.getLogger(__name__)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
DATA_HEADERS = ["Date","Subreddit","Title","Sentiment","Theme","Summary","Upvotes","Comments","Competitor Mentions","URL"]
DETAILED_DATA_HEADERS = DATA_HEADERS + [
    "Post Type", "Purchase Intent", "Recommendation Strength",
    "Pain Points", "Feature Requests",
    "Head-to-Head", "Competitor Sentiment", "Top Comment Sentiment",
]

def _build_gspread_client():
    if GOOGLE_OAUTH_REFRESH_TOKEN and GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET:
        logger.info("Using OAuth2 user credentials for Google Sheets")
        client, _ = gspread.oauth_from_dict(authorized_user_info={"type":"authorized_user","client_id":GOOGLE_OAUTH_CLIENT_ID,"client_secret":GOOGLE_OAUTH_CLIENT_SECRET,"refresh_token":GOOGLE_OAUTH_REFRESH_TOKEN}, scopes=SCOPES)
        return client
    logger.info("Using service account credentials for Google Sheets")
    return gspread.authorize(SACredentials.from_service_account_file(GOOGLE_SHEETS_CREDS_FILE, scopes=SCOPES))

def _retry_on_quota(func, *args, max_retries=5, **kwargs):
    """Retry a Google Sheets API call with exponential backoff on 429 quota errors."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                logger.warning(f"Sheets API quota hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


def _format_list_field(value):
    """Format a list or string field for display in sheets."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value if v)
    return str(value) if value else ""


def _format_json_field(value):
    """Format a dict/object field as JSON string for display."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value) if value else ""


class SheetsWriter:
    def __init__(self):
        self.client = _build_gspread_client()

    def create_research_sheet(self, brand_name, results, detailed=False, sov_data=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        spreadsheet = self.client.create(f"{brand_name} - Reddit Research - {timestamp}", folder_id=GOOGLE_DRIVE_FOLDER_ID or None)
        ws = spreadsheet.sheet1
        _retry_on_quota(ws.update_title, "Research Data")

        headers = DETAILED_DATA_HEADERS if detailed else DATA_HEADERS
        last_col = chr(ord('A') + len(headers) - 1)  # e.g. 'J' or 'R'

        _retry_on_quota(ws.update, "A1", [headers])
        _retry_on_quota(ws.format, f"A1:{last_col}1", {"textFormat":{"bold":True},"backgroundColor":{"red":0.9,"green":0.9,"blue":0.9}})

        rows = []
        for r in results:
            row = [
                r["created_date"], f"r/{r['subreddit']}", r["title"],
                r["sentiment"].capitalize(), r["theme"], r["summary"],
                r["score"], r["num_comments"], r["competitor_mentions"], r["url"],
            ]
            if detailed:
                row.extend([
                    r.get("post_type", ""),
                    r.get("purchase_intent", ""),
                    r.get("recommendation_strength", ""),
                    _format_list_field(r.get("pain_points", [])),
                    _format_list_field(r.get("feature_requests", [])),
                    _format_json_field(r.get("head_to_head")),
                    _format_json_field(r.get("competitor_sentiment")),
                    r.get("top_comment_sentiment", ""),
                ])
            rows.append(row)

        if rows:
            _retry_on_quota(ws.update, f"A2:{last_col}{len(rows)+1}", rows)

        # Color-code sentiment column
        color_map = {"positive":{"red":0.85,"green":0.95,"blue":0.85},"negative":{"red":0.95,"green":0.85,"blue":0.85},"neutral":{"red":0.93,"green":0.93,"blue":0.93}}
        batch_formats = []
        for i, r in enumerate(results):
            c = color_map.get(r["sentiment"])
            if c:
                batch_formats.append({"range": f"D{i+2}", "format": {"backgroundColor": c}})
        if batch_formats:
            _retry_on_quota(ws.batch_format, batch_formats)

        # Summary worksheet
        summary_rows = max(30, 60 if detailed else 30)
        summary_ws = _retry_on_quota(spreadsheet.add_worksheet, "Summary", rows=summary_rows, cols=5)
        _retry_on_quota(summary_ws.update, "A1", self._build_summary(brand_name, results, detailed=detailed, sov_data=sov_data))
        _retry_on_quota(summary_ws.format, "A1:B1", {"textFormat":{"bold":True},"backgroundColor":{"red":0.9,"green":0.9,"blue":0.9}})

        _retry_on_quota(spreadsheet.share, "", perm_type="anyone", role="reader")
        logger.info(f"Created Google Sheet: {spreadsheet.url}")
        return spreadsheet.url

    @staticmethod
    def _build_summary(brand_name, results, detailed=False, sov_data=None):
        sentiments = Counter(r["sentiment"] for r in results)
        subreddits = Counter(r["subreddit"] for r in results)
        themes = Counter(r["theme"] for r in results)
        rows = [["Metric","Value"],["Brand",brand_name],["Total Relevant Posts",len(results)],["Positive",sentiments.get("positive",0)],["Negative",sentiments.get("negative",0)],["Neutral",sentiments.get("neutral",0)],[],["Top Subreddits","Count"]]
        for sub, count in subreddits.most_common(10): rows.append([f"r/{sub}", count])
        rows.extend([[], ["Top Themes","Count"]])
        for theme, count in themes.most_common(10): rows.append([theme, count])

        if detailed:
            # Post Type distribution
            post_types = Counter(r.get("post_type", "discussion") for r in results)
            rows.extend([[], ["Post Type", "Count"]])
            for pt, count in post_types.most_common():
                rows.append([pt.capitalize(), count])

            # Purchase Intent distribution
            intents = Counter(r.get("purchase_intent", "none") for r in results)
            active_intents = {k: v for k, v in intents.items() if k != "none"}
            if active_intents:
                rows.extend([[], ["Purchase Intent", "Count"]])
                for intent, count in sorted(active_intents.items(), key=lambda x: x[1], reverse=True):
                    rows.append([intent.capitalize(), count])

            # Recommendation Strength distribution
            recs = Counter(r.get("recommendation_strength", "neutral") for r in results)
            rows.extend([[], ["Recommendation Strength", "Count"]])
            for level in ["strong_recommend", "recommend", "neutral", "caution", "strong_negative"]:
                count = recs.get(level, 0)
                if count:
                    rows.append([level.replace("_", " ").title(), count])

            # Top Pain Points (aggregated)
            all_pain = []
            for r in results:
                pp = r.get("pain_points", [])
                if isinstance(pp, list):
                    all_pain.extend(p for p in pp if p)
                elif pp:
                    all_pain.extend(p.strip() for p in str(pp).split(";") if p.strip())
            if all_pain:
                rows.extend([[], ["Top Pain Points", "Count"]])
                for pain, count in Counter(all_pain).most_common(10):
                    rows.append([pain, count])

            # Top Feature Requests (aggregated)
            all_feat = []
            for r in results:
                fr = r.get("feature_requests", [])
                if isinstance(fr, list):
                    all_feat.extend(f for f in fr if f)
                elif fr:
                    all_feat.extend(f.strip() for f in str(fr).split(";") if f.strip())
            if all_feat:
                rows.extend([[], ["Top Feature Requests", "Count"]])
                for feat, count in Counter(all_feat).most_common(10):
                    rows.append([feat, count])

            # Share of Voice
            if sov_data:
                rows.extend([[], ["Share of Voice", "Mentions", "Share %"]])
                for name, data in sorted(sov_data.items(), key=lambda x: x[1]["mentions"], reverse=True):
                    rows.append([name, data["mentions"], f"{data['share_pct']}%"])

        return rows


def export_results_csv(brand_name, results, detailed=False):
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
    safe = brand_name.lower().replace(" ","_").replace("&","and")
    path = os.path.join(CHART_OUTPUT_DIR, f"{safe}_research_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    headers = DETAILED_DATA_HEADERS if detailed else DATA_HEADERS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in results:
            row = [
                r["created_date"], f"r/{r['subreddit']}", r["title"],
                r["sentiment"].capitalize(), r["theme"], r["summary"],
                r["score"], r["num_comments"], r["competitor_mentions"], r["url"],
            ]
            if detailed:
                row.extend([
                    r.get("post_type", ""),
                    r.get("purchase_intent", ""),
                    r.get("recommendation_strength", ""),
                    _format_list_field(r.get("pain_points", [])),
                    _format_list_field(r.get("feature_requests", [])),
                    _format_json_field(r.get("head_to_head")),
                    _format_json_field(r.get("competitor_sentiment")),
                    r.get("top_comment_sentiment", ""),
                ])
            w.writerow(row)
    logger.info(f"Exported CSV: {path} ({len(results)} rows)")
    return path
