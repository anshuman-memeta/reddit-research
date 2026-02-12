"""
Google Sheets writer — OAuth2 user credentials (preferred) or service account fallback.
"""
import csv, logging, os, time
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

class SheetsWriter:
    def __init__(self):
        self.client = _build_gspread_client()
    def create_research_sheet(self, brand_name, results):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        spreadsheet = self.client.create(f"{brand_name} - Reddit Research - {timestamp}", folder_id=GOOGLE_DRIVE_FOLDER_ID or None)
        ws = spreadsheet.sheet1
        _retry_on_quota(ws.update_title, "Research Data")
        _retry_on_quota(ws.update, "A1", [DATA_HEADERS])
        _retry_on_quota(ws.format, "A1:J1", {"textFormat":{"bold":True},"backgroundColor":{"red":0.9,"green":0.9,"blue":0.9}})
        rows = [[r["created_date"],f"r/{r['subreddit']}",r["title"],r["sentiment"].capitalize(),r["theme"],r["summary"],r["score"],r["num_comments"],r["competitor_mentions"],r["url"]] for r in results]
        if rows:
            _retry_on_quota(ws.update, f"A2:J{len(rows)+1}", rows)
        color_map = {"positive":{"red":0.85,"green":0.95,"blue":0.85},"negative":{"red":0.95,"green":0.85,"blue":0.85},"neutral":{"red":0.93,"green":0.93,"blue":0.93}}
        batch_formats = []
        for i, r in enumerate(results):
            c = color_map.get(r["sentiment"])
            if c:
                batch_formats.append({"range": f"D{i+2}", "format": {"backgroundColor": c}})
        if batch_formats:
            _retry_on_quota(ws.batch_format, batch_formats)
        summary_ws = _retry_on_quota(spreadsheet.add_worksheet, "Summary", rows=30, cols=5)
        _retry_on_quota(summary_ws.update, "A1", self._build_summary(brand_name, results))
        _retry_on_quota(summary_ws.format, "A1:B1", {"textFormat":{"bold":True},"backgroundColor":{"red":0.9,"green":0.9,"blue":0.9}})
        _retry_on_quota(spreadsheet.share, "", perm_type="anyone", role="reader")
        logger.info(f"Created Google Sheet: {spreadsheet.url}")
        return spreadsheet.url
    @staticmethod
    def _build_summary(brand_name, results):
        sentiments = Counter(r["sentiment"] for r in results)
        subreddits = Counter(r["subreddit"] for r in results)
        themes = Counter(r["theme"] for r in results)
        rows = [["Metric","Value"],["Brand",brand_name],["Total Relevant Posts",len(results)],["Positive",sentiments.get("positive",0)],["Negative",sentiments.get("negative",0)],["Neutral",sentiments.get("neutral",0)],[],["Top Subreddits","Count"]]
        for sub, count in subreddits.most_common(10): rows.append([f"r/{sub}", count])
        rows.extend([[], ["Top Themes","Count"]])
        for theme, count in themes.most_common(10): rows.append([theme, count])
        return rows

def export_results_csv(brand_name, results):
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
    safe = brand_name.lower().replace(" ","_").replace("&","and")
    path = os.path.join(CHART_OUTPUT_DIR, f"{safe}_research_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DATA_HEADERS)
        for r in results:
            w.writerow([r["created_date"],f"r/{r['subreddit']}",r["title"],r["sentiment"].capitalize(),r["theme"],r["summary"],r["score"],r["num_comments"],r["competitor_mentions"],r["url"]])
    logger.info(f"Exported CSV: {path} ({len(results)} rows)")
    return path
