# Project Context

## Deployment

- The bot runs on a **Vultr VPS** (Mumbai region): `ssh root@139.84.173.63`
- **Do NOT run `bot.py` in this container** — it will conflict with the VPS instance and cause Telegram 409 Conflict errors
- Only one bot instance can poll Telegram at a time

## Deploying Changes

1. Push code changes to the repo
2. SSH into the VPS: `ssh root@139.84.173.63`
3. Pull changes: `cd ~/reddit-research && git pull`
4. Restart the bot: `pkill -f bot.py && sleep 2 && nohup ./run-bot.sh > bot.log 2>&1 &`

## Configuration

- All secrets are in `.env` (not committed — see `.env.example` for the template)
- Required env vars: `RESEARCH_BOT_TOKEN`, `AUTHORIZED_USERS`, `GROQ_API_KEY`
- Optional: `GOOGLE_SHEETS_CREDS_FILE`

## Authorized Users

- 804130532 (Anshuman)
- 7564871164 (Colleague)
