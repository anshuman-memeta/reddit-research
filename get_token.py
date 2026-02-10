from google_auth_oauthlib.flow import InstalledAppFlow
flow = InstalledAppFlow.from_client_secrets_file(
    "client_secret.json",
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
creds = flow.run_local_server(port=0)
print(f"\nGOOGLE_OAUTH_REFRESH_TOKEN={creds.refresh_token}")
