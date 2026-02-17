#!/usr/bin/env python3
"""
Reddit Research Bot - Telegram interface.

Commands:
  /research <brand>   - Run a 3-month deep dive on a brand
  /research_add       - Onboard a new brand (interactive)
  /research_list      - List all configured brands
  /help               - Show help
"""

import asyncio
import json
import logging
import os
import sys

from collections import Counter
from datetime import datetime

from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import (
    AUTHORIZED_USERS,
    BRANDS_CONFIG_PATH,
    
    TELEGRAM_BOT_TOKEN,
)
from fetcher import MultiSourceFetcher
from analyzer import BrandAnalyzer
from charts import generate_sentiment_pie, generate_subreddit_pie, generate_sentiment_trend
from sheets import SheetsWriter, export_results_csv

# ---- Logging -------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---- Conversation states for /research_add --------------------------------
BRAND_NAME, CATEGORY, KEYWORDS, PRODUCT_TERMS, COMPETITORS, SUBREDDIT_HINTS, DESCRIPTION = range(7)


# ---- Helpers --------------------------------------------------------------

def load_brands() -> dict:
    with open(BRANDS_CONFIG_PATH, "r") as f:
        return json.load(f)


def save_brands(data: dict) -> None:
    with open(BRANDS_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def is_authorized(user_id: int) -> bool:
    return user_id in AUTHORIZED_USERS


# ---- /start & /help -------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return

    await update.message.reply_text(
        "*Reddit Research Bot*\n\n"
        "Commands:\n"
        "/research <brand> \u2014 Deep dive (3 months)\n"
        "/research\\_add \u2014 Add a new brand\n"
        "/research\\_list \u2014 List configured brands\n"
        "/help \u2014 Show this message",
        parse_mode="Markdown",
    )


# ---- /research_list -------------------------------------------------------

async def cmd_research_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return

    brands = load_brands().get("brands", {})
    if not brands:
        await update.message.reply_text("No brands configured. Use /research\\_add to add one.")
        return

    lines = ["*Configured Brands:*\n"]
    for name, cfg in brands.items():
        cat = cfg.get("category", "general")
        kws = ", ".join(cfg.get("keywords", []))
        lines.append(f"\u2022 *{name}* [{cat}]\n  Keywords: {kws}\n")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ---- /research <brand> ----------------------------------------------------

async def cmd_research(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return

    if not context.args:
        await update.message.reply_text("Usage: `/research <brand_name>`", parse_mode="Markdown")
        return

    brand_name = " ".join(context.args)
    brands = load_brands().get("brands", {})

    # Case-insensitive lookup
    brand_config = None
    matched_name = None
    for name, cfg in brands.items():
        if name.lower() == brand_name.lower():
            brand_config = dict(cfg)
            brand_config["name"] = name
            matched_name = name
            break

    if not brand_config:
        available = ", ".join(brands.keys())
        await update.message.reply_text(
            f"Brand '{brand_name}' not found.\n\n"
            f"Available: {available}\n\n"
            f"Use /research\\_add to add a new brand.",
        )
        return

    await update.message.reply_text(
        f"*Starting deep dive on {matched_name}...*\n\n"
        f"Category: {brand_config.get('category', 'general')}\n"
        f"Keywords: {', '.join(brand_config.get('keywords', []))}\n"
        f"Lookback: 3 months\n\n"
        f"This may take a few minutes. I'll send results when ready.",
        parse_mode="Markdown",
    )

    # Run heavy pipeline in background so the bot stays responsive
    task = asyncio.create_task(_run_research_pipeline(update, matched_name, brand_config))
    task.add_done_callback(_task_done_callback)


def _task_done_callback(task: asyncio.Task):
    """Log any unhandled exceptions from background research tasks."""
    if task.cancelled():
        logger.warning("Research task was cancelled")
    elif task.exception():
        logger.error(f"Research task crashed: {task.exception()}", exc_info=task.exception())


async def _run_research_pipeline(update: Update, brand_name: str, brand_config: dict):
    """Execute fetch -> analyze -> chart -> sheet -> deliver."""
    chat_id = update.effective_chat.id
    bot = update.get_bot()

    try:
        # -- Step 1: Fetch --------------------------------------------------
        await bot.send_message(chat_id, "Fetching posts from Arctic Shift, Reddit & Pullpush...")

        fetcher = MultiSourceFetcher()

        # Send fetcher progress to Telegram so we can see per-source results
        loop = asyncio.get_running_loop()

        def fetch_progress(msg: str):
            try:
                asyncio.run_coroutine_threadsafe(
                    bot.send_message(chat_id, msg), loop
                )
            except Exception:
                pass

        posts = await asyncio.to_thread(
            fetcher.fetch_all, brand_config, lookback_days=90,
            progress_callback=fetch_progress,
        )

        if not posts:
            error_msg = f"No posts found for {brand_name} in the last 3 months."
            if fetcher.errors:
                # Show first few errors so user knows what went wrong
                error_msg += "\n\nSource errors encountered:"
                for err in fetcher.errors[:5]:
                    error_msg += f"\n- {err}"
                if len(fetcher.errors) > 5:
                    error_msg += f"\n... and {len(fetcher.errors) - 5} more"
            await bot.send_message(chat_id, error_msg)
            return

        await bot.send_message(
            chat_id,
            f"Found {len(posts)} posts. Filtering for relevance & analyzing sentiment...\n"
            f"(processing in batches of 10 via LLM)",
        )

        # -- Step 2: Analyze ------------------------------------------------
        analyzer = BrandAnalyzer()
        loop = asyncio.get_running_loop()

        def progress_callback(done, total, relevant):
            try:
                asyncio.run_coroutine_threadsafe(
                    bot.send_message(
                        chat_id,
                        f"Analyzed {done}/{total} posts ({relevant} relevant so far)...",
                    ),
                    loop,
                )
            except Exception:
                pass

        results = await asyncio.to_thread(
            analyzer.process_posts, posts, brand_config, progress_callback
        )

        if not results:
            await bot.send_message(
                chat_id,
                f"No relevant posts found for {brand_name} after filtering.\n"
                f"The posts found were not actually about the brand.",
            )
            return

        await bot.send_message(chat_id, f"{len(results)} relevant posts analyzed. Generating outputs...")

        # -- Step 3: Charts -------------------------------------------------
        chart_paths = []

        sentiment_chart = await asyncio.to_thread(generate_sentiment_pie, results, brand_name)
        if sentiment_chart:
            chart_paths.append(sentiment_chart)

        subreddit_chart = await asyncio.to_thread(generate_subreddit_pie, results, brand_name)
        if subreddit_chart:
            chart_paths.append(subreddit_chart)

        trend_chart = await asyncio.to_thread(generate_sentiment_trend, results, brand_name)
        if trend_chart:
            chart_paths.append(trend_chart)

        # -- Step 4: Google Sheet -------------------------------------------
        sheet_url = None
        csv_path = None
        try:
            writer = SheetsWriter()
            sheet_url = await asyncio.to_thread(writer.create_research_sheet, brand_name, results)
        except Exception as e:
            logger.error(f"Google Sheets error: {e}")
            await bot.send_message(chat_id, f"Could not create Google Sheet: {e}\nExporting CSV instead...")
            try:
                csv_path = await asyncio.to_thread(export_results_csv, brand_name, results)
            except Exception as csv_err:
                logger.error(f"CSV export also failed: {csv_err}")

        # -- Step 5: Summary message ----------------------------------------
        sentiments = Counter(r["sentiment"] for r in results)
        subreddits = Counter(r["subreddit"] for r in results)
        themes = Counter(r["theme"] for r in results)

        total = len(results)
        pos = sentiments.get("positive", 0)
        neg = sentiments.get("negative", 0)
        neu = sentiments.get("neutral", 0)

        summary = (
            f"*Research Complete: {brand_name}*\n"
            f"{'=' * 30}\n\n"
            f"*Total relevant posts:* {total} (last 3 months)\n\n"
            f"*Sentiment breakdown:*\n"
            f"  Positive: {pos} ({pos / total * 100:.1f}%)\n"
            f"  Negative: {neg} ({neg / total * 100:.1f}%)\n"
            f"  Neutral: {neu} ({neu / total * 100:.1f}%)\n\n"
            f"*Top subreddits:*\n"
        )
        for sub, count in subreddits.most_common(5):
            summary += f"  r/{sub}: {count} posts\n"

        summary += "\n*Top themes:*\n"
        for theme, count in themes.most_common(5):
            summary += f"  {theme}: {count}\n"

        if sheet_url:
            summary += f"\n*Google Sheet:* [Open Research Appendix]({sheet_url})"

        await bot.send_message(chat_id, summary, parse_mode="Markdown", disable_web_page_preview=True)

        # -- Step 6: Send charts --------------------------------------------
        for path in chart_paths:
            with open(path, "rb") as f:
                await bot.send_photo(chat_id, photo=f)

        # -- Step 7: Send CSV if Sheets failed ------------------------------
        if csv_path:
            with open(csv_path, "rb") as f:
                await bot.send_document(chat_id, document=f, filename=os.path.basename(csv_path))

        # Cleanup
        for path in chart_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        if csv_path:
            try:
                os.remove(csv_path)
            except OSError:
                pass

    except Exception as e:
        logger.error(f"Research pipeline error: {e}", exc_info=True)
        await bot.send_message(chat_id, f"Research failed: {e}")


# ---- /research_add (conversation) -----------------------------------------

async def add_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return ConversationHandler.END
    context.user_data.pop("new_brand", None)  # clear any stale state
    await update.message.reply_text(
        "Let's add a new brand.\n(Send /cancel anytime to abort)\n\n*Brand name?*",
        parse_mode="Markdown",
    )
    return BRAND_NAME


async def add_brand_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["new_brand"] = {"name": update.message.text.strip()}
    await update.message.reply_text(
        "*Category?*\n(beauty / finance / health\\_fitness / food / footwear / tech\\_gadgets / general)\n\n_Or type your own custom category_",
        parse_mode="Markdown",
    )
    return CATEGORY


async def add_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["new_brand"]["category"] = update.message.text.strip().lower()
    await update.message.reply_text(
        "*Keywords to search?*\n(comma-separated)\nExample: Sahi app, Sahi trading, Sahi invest",
        parse_mode="Markdown",
    )
    return KEYWORDS


async def add_keywords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.lower() == "skip":
        context.user_data["new_brand"]["keywords"] = []
    else:
        context.user_data["new_brand"]["keywords"] = [
            k.strip() for k in text.split(",") if k.strip()
        ]
    await update.message.reply_text(
        "*Product terms?*\n(comma-separated context words)\nExample: stock, trading, mutual fund, demat\n\nType `skip` to skip.",
        parse_mode="Markdown",
    )
    return PRODUCT_TERMS


async def add_product_terms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.lower() == "skip":
        context.user_data["new_brand"]["product_terms"] = []
    else:
        context.user_data["new_brand"]["product_terms"] = [
            t.strip() for t in text.split(",") if t.strip()
        ]
    await update.message.reply_text(
        "*Competitors?*\n(comma-separated)\nExample: Groww, Zerodha, Angel One\n\nType `skip` to skip.",
        parse_mode="Markdown",
    )
    return COMPETITORS


async def add_competitors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.lower() == "skip":
        context.user_data["new_brand"]["competitors"] = []
    else:
        context.user_data["new_brand"]["competitors"] = [
            c.strip() for c in text.split(",") if c.strip()
        ]
    await update.message.reply_text(
        "*Relevant subreddits?*\n(comma-separated, without r/)\nExample: IndiaInvestments, DesiStreetBets\n\nType `skip` to skip.",
        parse_mode="Markdown",
    )
    return SUBREDDIT_HINTS


async def add_subreddit_hints(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.lower() == "skip":
        context.user_data["new_brand"]["subreddit_hints"] = []
    else:
        context.user_data["new_brand"]["subreddit_hints"] = [
            s.strip() for s in text.split(",") if s.strip()
        ]
    await update.message.reply_text(
        "*Brief description of the brand?*\nExample: Fintech app for trading in the Indian stock market",
        parse_mode="Markdown",
    )
    return DESCRIPTION


async def add_description(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        brand_data = context.user_data.pop("new_brand")
        brand_name = brand_data.pop("name")
        brand_data["description"] = update.message.text.strip()

        # Persist
        brands = load_brands()
        brands["brands"][brand_name] = brand_data
        save_brands(brands)
        logger.info(f"Brand '{brand_name}' saved to {BRANDS_CONFIG_PATH}")

        await update.message.reply_text(
            f"{brand_name} added!\n\n"
            f"Category: {brand_data['category']}\n"
            f"Keywords: {', '.join(brand_data['keywords'])}\n"
            f"Product terms: {', '.join(brand_data['product_terms'])}\n"
            f"Competitors: {', '.join(brand_data.get('competitors', [])) or 'None'}\n"
            f"Subreddits: {', '.join(brand_data.get('subreddit_hints', [])) or 'None'}\n\n"
            f"Run /research {brand_name} to start.",
        )
    except Exception as e:
        logger.error(f"add_description failed: {e}", exc_info=True)
        await update.message.reply_text(f"Error saving brand: {e}")
    return ConversationHandler.END


async def add_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("new_brand", None)
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# ---- Bot setup & main -----------------------------------------------------

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("Error: RESEARCH_BOT_TOKEN environment variable not set")
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Simple commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("research_list", cmd_research_list))
    app.add_handler(CommandHandler("research", cmd_research))

    # Conversation: /research_add
    conv = ConversationHandler(
        entry_points=[CommandHandler("research_add", add_start)],
        states={
            BRAND_NAME:     [MessageHandler(filters.TEXT & ~filters.COMMAND, add_brand_name)],
            CATEGORY:       [MessageHandler(filters.TEXT & ~filters.COMMAND, add_category)],
            KEYWORDS:       [MessageHandler(filters.TEXT & ~filters.COMMAND, add_keywords)],
            PRODUCT_TERMS:  [MessageHandler(filters.TEXT & ~filters.COMMAND, add_product_terms)],
            COMPETITORS:    [MessageHandler(filters.TEXT & ~filters.COMMAND, add_competitors)],
            SUBREDDIT_HINTS:[MessageHandler(filters.TEXT & ~filters.COMMAND, add_subreddit_hints)],
            DESCRIPTION:    [MessageHandler(filters.TEXT & ~filters.COMMAND, add_description)],
        },
        fallbacks=[
            CommandHandler("cancel", add_cancel),
            CommandHandler("research_add", add_start),  # restart mid-conversation
        ],
    )
    app.add_handler(conv)

    # Register commands in Telegram menu
    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("research", "Deep dive on a brand (3 months)"),
            BotCommand("research_add", "Add a new brand"),
            BotCommand("research_list", "List configured brands"),
            BotCommand("help", "Show help"),
        ])

    app.post_init = post_init

    # Global error handler - log all unhandled errors
    async def error_handler(update, context):
        logger.error(f"Unhandled error: {context.error}", exc_info=context.error)
        if update and update.effective_chat:
            try:
                await context.bot.send_message(
                    update.effective_chat.id,
                    f"Something went wrong: {context.error}",
                )
            except Exception:
                pass

    app.add_error_handler(error_handler)

    logger.info("Research bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
