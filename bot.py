#!/usr/bin/env python3
"""
Reddit Research Bot - Telegram interface.

Commands:
  /research_quick <brand>    - Quick research (sentiment, themes, subreddits)
  /research_detailed <brand> - Detailed research (+ comments, pain points, etc.)
  /research_add              - Onboard a new brand (interactive)
  /research_edit <brand>     - Edit a brand's config (interactive)
  /research_delete <brand>   - Delete a brand
  /research_list             - List configured brands
  /research_stop             - Cancel a running research
  /help                      - Show help
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
    COMMENT_SCORE_THRESHOLD,
    TELEGRAM_BOT_TOKEN,
)
from fetcher import MultiSourceFetcher, CommentFetcher
from analyzer import BrandAnalyzer, compute_share_of_voice
from charts import (
    generate_sentiment_pie, generate_subreddit_pie,
    generate_post_type_chart, generate_recommendation_chart,
    generate_share_of_voice_chart,
)
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
DELETE_CONFIRM = 0  # single state for delete conversation

# ---- Conversation states for /research_edit -------------------------------
EDIT_FIELD_SELECT, EDIT_FIELD_VALUE = range(2)

# ---- Active research tasks (chat_id -> asyncio.Task) ----------------------
_active_research: dict[int, asyncio.Task] = {}


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
        "/research\\_quick <brand> \u2014 Quick research (3 months)\n"
        "/research\\_detailed <brand> \u2014 Detailed research with comments, pain points, competitive intel\n"
        "/research\\_add \u2014 Add a new brand\n"
        "/research\\_edit <brand> \u2014 Edit a brand's config\n"
        "/research\\_delete <brand> \u2014 Delete a brand\n"
        "/research\\_list \u2014 List configured brands\n"
        "/research\\_stop \u2014 Cancel running research\n"
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


# ---- /research_quick & /research_detailed -----------------------------------

def _lookup_brand(brand_name: str, brands: dict):
    """Case-insensitive brand lookup. Returns (matched_name, brand_config) or (None, None)."""
    for name, cfg in brands.items():
        if name.lower() == brand_name.lower():
            brand_config = dict(cfg)
            brand_config["name"] = name
            return name, brand_config
    return None, None


async def _start_research(update: Update, context: ContextTypes.DEFAULT_TYPE, detailed: bool):
    """Shared entry point for quick and detailed research commands."""
    if not is_authorized(update.effective_user.id):
        return

    cmd = "research_detailed" if detailed else "research_quick"
    if not context.args:
        await update.message.reply_text(f"Usage: `/{cmd} <brand_name>`", parse_mode="Markdown")
        return

    brand_name = " ".join(context.args)
    brands = load_brands().get("brands", {})
    matched_name, brand_config = _lookup_brand(brand_name, brands)

    if not brand_config:
        available = ", ".join(brands.keys())
        await update.message.reply_text(
            f"Brand '{brand_name}' not found.\n\n"
            f"Available: {available}\n\n"
            f"Use /research\\_add to add a new brand.",
        )
        return

    mode_label = "detailed" if detailed else "quick"
    extra = ""
    if detailed:
        extra = "\nThis will take longer as I'll analyze comments, pain points, and competitive data."

    await update.message.reply_text(
        f"*Starting {mode_label} research on {matched_name}...*\n\n"
        f"Category: {brand_config.get('category', 'general')}\n"
        f"Keywords: {', '.join(brand_config.get('keywords', []))}\n"
        f"Lookback: 3 months{extra}\n\n"
        f"I'll send results when ready.",
        parse_mode="Markdown",
    )

    chat_id = update.effective_chat.id

    # Cancel any already-running research for this chat
    existing = _active_research.get(chat_id)
    if existing and not existing.done():
        existing.cancel()
        await update.message.reply_text("Previous research cancelled. Starting new one...")

    task = asyncio.create_task(
        _run_research_pipeline(update, matched_name, brand_config, detailed=detailed)
    )
    _active_research[chat_id] = task
    task.add_done_callback(lambda t: _task_done_callback(t, chat_id))


async def cmd_research_quick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _start_research(update, context, detailed=False)


async def cmd_research_detailed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _start_research(update, context, detailed=True)


def _task_done_callback(task: asyncio.Task, chat_id: int):
    """Log any unhandled exceptions from background research tasks."""
    _active_research.pop(chat_id, None)
    if task.cancelled():
        logger.warning("Research task was cancelled")
    elif task.exception():
        logger.error(f"Research task crashed: {task.exception()}", exc_info=task.exception())


async def _send_long_message(bot, chat_id, text, parse_mode="Markdown"):
    """Send a message, splitting if it exceeds Telegram's 4096 char limit."""
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        await bot.send_message(chat_id, text, parse_mode=parse_mode, disable_web_page_preview=True)
        return

    parts = text.split("\n\n")
    current = ""
    for part in parts:
        if len(current) + len(part) + 2 > MAX_LEN:
            if current:
                await bot.send_message(chat_id, current, parse_mode=parse_mode, disable_web_page_preview=True)
            current = part
        else:
            current = current + "\n\n" + part if current else part
    if current:
        await bot.send_message(chat_id, current, parse_mode=parse_mode, disable_web_page_preview=True)


async def _run_research_pipeline(
    update: Update,
    brand_name: str,
    brand_config: dict,
    detailed: bool = False,
):
    """Execute fetch -> analyze -> (optional: SOV, comments) -> chart -> sheet -> deliver."""
    chat_id = update.effective_chat.id
    bot = update.get_bot()

    try:
        def _check_cancelled():
            task = _active_research.get(chat_id)
            if task and task.cancelled():
                raise asyncio.CancelledError()

        # -- Step 1: Fetch --------------------------------------------------
        await bot.send_message(chat_id, "Fetching posts from Arctic Shift, Reddit & Pullpush...")

        fetcher = MultiSourceFetcher()
        loop = asyncio.get_running_loop()

        def fetch_progress(msg: str):
            try:
                asyncio.run_coroutine_threadsafe(bot.send_message(chat_id, msg), loop)
            except Exception:
                pass

        posts = await asyncio.to_thread(
            fetcher.fetch_all, brand_config, lookback_days=90,
            progress_callback=fetch_progress,
        )

        _check_cancelled()

        if not posts:
            error_msg = f"No posts found for {brand_name} in the last 3 months."
            if fetcher.errors:
                error_msg += "\n\nSource errors encountered:"
                for err in fetcher.errors[:5]:
                    error_msg += f"\n- {err}"
                if len(fetcher.errors) > 5:
                    error_msg += f"\n... and {len(fetcher.errors) - 5} more"
            await bot.send_message(chat_id, error_msg)
            return

        mode_label = "detailed" if detailed else "quick"
        await bot.send_message(
            chat_id,
            f"Found {len(posts)} posts. Running {mode_label} analysis...\n"
            f"(processing in batches of 10 via LLM)",
        )

        _check_cancelled()

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
            analyzer.process_posts, posts, brand_config, progress_callback,
            detailed=detailed,
        )

        if not results:
            await bot.send_message(
                chat_id,
                f"No relevant posts found for {brand_name} after filtering.\n"
                f"The posts found were not actually about the brand.",
            )
            return

        await bot.send_message(chat_id, f"{len(results)} relevant posts analyzed. Generating outputs...")

        _check_cancelled()

        # -- Detailed-only steps: SOV, comments, comment analysis -----------
        sov_data = None
        if detailed:
            # Step 3: Share of Voice
            keywords = brand_config.get("keywords", [])
            competitors = brand_config.get("competitors", [])
            sov_data = await asyncio.to_thread(
                compute_share_of_voice, posts, brand_name, keywords, competitors
            )

            _check_cancelled()

            # Step 4: Fetch comments for high-engagement posts
            comment_posts = [r for r in results if r["score"] >= COMMENT_SCORE_THRESHOLD]
            comments_by_post = {}
            if comment_posts:
                await bot.send_message(
                    chat_id,
                    f"Fetching comments for {len(comment_posts)} posts with 20+ upvotes...",
                )

                comment_fetcher = CommentFetcher()

                def comment_progress(done, total):
                    try:
                        asyncio.run_coroutine_threadsafe(
                            bot.send_message(
                                chat_id,
                                f"Fetched comments for {done}/{total} posts...",
                            ),
                            loop,
                        )
                    except Exception:
                        pass

                comments_by_post = await asyncio.to_thread(
                    comment_fetcher.fetch_comments_batch,
                    comment_posts,
                    progress_callback=comment_progress,
                )

                _check_cancelled()

            # Step 5: Analyze comments via LLM
            posts_with_comments = {
                pid: comms for pid, comms in comments_by_post.items() if comms
            }
            if posts_with_comments:
                await bot.send_message(
                    chat_id,
                    f"Analyzing comments for {len(posts_with_comments)} posts...",
                )

                comment_analysis = await asyncio.to_thread(
                    analyzer.analyze_comments, posts_with_comments, brand_config
                )

                # Step 6: Merge comment analysis into results
                for r in results:
                    ca = comment_analysis.get(r["post_id"])
                    if ca:
                        r["top_comment_sentiment"] = ca["top_comment_sentiment"]
                        # Merge comment-discovered pain points
                        existing_pp = r.get("pain_points", [])
                        if isinstance(existing_pp, list):
                            r["pain_points"] = list(set(existing_pp + ca.get("comment_pain_points", [])))
                        # Merge comment-discovered feature requests
                        existing_fr = r.get("feature_requests", [])
                        if isinstance(existing_fr, list):
                            r["feature_requests"] = list(set(existing_fr + ca.get("comment_feature_requests", [])))

                _check_cancelled()

        # -- Charts ---------------------------------------------------------
        chart_paths = []

        sentiment_chart = await asyncio.to_thread(generate_sentiment_pie, results, brand_name)
        if sentiment_chart:
            chart_paths.append(sentiment_chart)

        subreddit_chart = await asyncio.to_thread(generate_subreddit_pie, results, brand_name)
        if subreddit_chart:
            chart_paths.append(subreddit_chart)

        if detailed:
            post_type_chart = await asyncio.to_thread(generate_post_type_chart, results, brand_name)
            if post_type_chart:
                chart_paths.append(post_type_chart)

            rec_chart = await asyncio.to_thread(generate_recommendation_chart, results, brand_name)
            if rec_chart:
                chart_paths.append(rec_chart)

            if sov_data:
                sov_chart = await asyncio.to_thread(generate_share_of_voice_chart, sov_data, brand_name)
                if sov_chart:
                    chart_paths.append(sov_chart)

        _check_cancelled()

        # -- Google Sheet ---------------------------------------------------
        sheet_url = None
        csv_path = None
        try:
            writer = SheetsWriter()
            sheet_url = await asyncio.to_thread(
                writer.create_research_sheet, brand_name, results,
                detailed=detailed, sov_data=sov_data,
            )
        except Exception as e:
            logger.error(f"Google Sheets error: {e}")
            await bot.send_message(chat_id, f"Could not create Google Sheet: {e}\nExporting CSV instead...")
            try:
                csv_path = await asyncio.to_thread(
                    export_results_csv, brand_name, results, detailed=detailed
                )
            except Exception as csv_err:
                logger.error(f"CSV export also failed: {csv_err}")

        # -- Summary message ------------------------------------------------
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

        if detailed:
            # Share of Voice
            if sov_data:
                summary += "\n*Share of Voice:*\n"
                for name, data in sorted(sov_data.items(), key=lambda x: x[1]["share_pct"], reverse=True):
                    marker = " (brand)" if name == brand_name else ""
                    summary += f"  {name}{marker}: {data['share_pct']}% ({data['mentions']} mentions)\n"

            # Post Types
            post_types = Counter(r.get("post_type", "discussion") for r in results)
            summary += "\n*Post Types:*\n"
            for pt, count in post_types.most_common():
                summary += f"  {pt.capitalize()}: {count}\n"

            # Purchase Intent
            intents = Counter(r.get("purchase_intent", "none") for r in results)
            active_intents = {k: v for k, v in intents.items() if k != "none"}
            if active_intents:
                summary += "\n*Purchase Intent Signals:*\n"
                for intent, count in sorted(active_intents.items(), key=lambda x: x[1], reverse=True):
                    summary += f"  {intent.capitalize()}: {count}\n"

            # Recommendation Strength
            recs = Counter(r.get("recommendation_strength", "neutral") for r in results)
            summary += "\n*Recommendation Strength:*\n"
            for level in ["strong_recommend", "recommend", "neutral", "caution", "strong_negative"]:
                count = recs.get(level, 0)
                if count:
                    summary += f"  {level.replace('_', ' ').title()}: {count}\n"

            # Top Pain Points
            all_pain = []
            for r in results:
                pp = r.get("pain_points", [])
                if isinstance(pp, list):
                    all_pain.extend(p for p in pp if p)
            if all_pain:
                summary += "\n*Top Pain Points:*\n"
                for pain, count in Counter(all_pain).most_common(5):
                    summary += f"  \u2022 {pain} ({count}x)\n"

            # Top Feature Requests
            all_feat = []
            for r in results:
                fr = r.get("feature_requests", [])
                if isinstance(fr, list):
                    all_feat.extend(f for f in fr if f)
            if all_feat:
                summary += "\n*Feature Requests:*\n"
                for feat, count in Counter(all_feat).most_common(5):
                    summary += f"  \u2022 {feat} ({count}x)\n"

            # Head-to-Head Comparisons
            h2h_results = []
            for r in results:
                h2h = r.get("head_to_head")
                if h2h and isinstance(h2h, dict):
                    h2h_results.append(h2h)
            if h2h_results:
                summary += "\n*Head-to-Head Comparisons:*\n"
                wins = Counter()
                for h in h2h_results:
                    comp = h.get("competitor", "?")
                    winner = h.get("winner", "tie")
                    wins[(comp, winner)] += 1
                for (comp, winner), count in wins.most_common(5):
                    summary += f"  vs {comp}: {winner} ({count}x)\n"

        if sheet_url:
            summary += f"\n*Google Sheet:* [Open Research Appendix]({sheet_url})"

        await _send_long_message(bot, chat_id, summary)

        # -- Send charts ----------------------------------------------------
        for path in chart_paths:
            with open(path, "rb") as f:
                await bot.send_photo(chat_id, photo=f)

        # -- Send CSV if Sheets failed --------------------------------------
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

    except asyncio.CancelledError:
        logger.info(f"Research for {brand_name} was cancelled by user")
        await bot.send_message(chat_id, f"Research on *{brand_name}* has been stopped.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Research pipeline error: {e}", exc_info=True)
        await bot.send_message(chat_id, f"Research failed: {e}")


# ---- /research_stop -------------------------------------------------------

async def cmd_research_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return

    chat_id = update.effective_chat.id
    task = _active_research.get(chat_id)

    if not task or task.done():
        await update.message.reply_text("No research is currently running.")
        return

    task.cancel()
    await update.message.reply_text("Stopping research... Please wait.")


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
            f"Run /research\\_quick {brand_name} or /research\\_detailed {brand_name} to start.",
        )
    except Exception as e:
        logger.error(f"add_description failed: {e}", exc_info=True)
        await update.message.reply_text(f"Error saving brand: {e}")
    return ConversationHandler.END


async def add_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("new_brand", None)
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# ---- /research_delete (conversation with confirmation) --------------------

async def delete_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return ConversationHandler.END

    if not context.args:
        await update.message.reply_text(
            "Usage: `/research_delete <brand_name>`", parse_mode="Markdown"
        )
        return ConversationHandler.END

    brand_name = " ".join(context.args)
    brands = load_brands().get("brands", {})

    # Case-insensitive lookup
    matched_name = None
    matched_cfg = None
    for name, cfg in brands.items():
        if name.lower() == brand_name.lower():
            matched_name = name
            matched_cfg = cfg
            break

    if not matched_cfg:
        available = ", ".join(brands.keys()) or "None"
        await update.message.reply_text(
            f"Brand '{brand_name}' not found.\n\nAvailable: {available}"
        )
        return ConversationHandler.END

    context.user_data["delete_brand"] = matched_name
    await update.message.reply_text(
        f"*Delete brand: {matched_name}*\n\n"
        f"Category: {matched_cfg.get('category', 'general')}\n"
        f"Keywords: {', '.join(matched_cfg.get('keywords', []))}\n"
        f"Product terms: {', '.join(matched_cfg.get('product_terms', []))}\n"
        f"Competitors: {', '.join(matched_cfg.get('competitors', [])) or 'None'}\n"
        f"Subreddits: {', '.join(matched_cfg.get('subreddit_hints', [])) or 'None'}\n\n"
        f"Are you sure? Reply *YES* to confirm or /cancel to abort.",
        parse_mode="Markdown",
    )
    return DELETE_CONFIRM


async def delete_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    brand_name = context.user_data.pop("delete_brand", None)

    if not brand_name:
        await update.message.reply_text("No brand selected. Use /research\\_delete <brand>.")
        return ConversationHandler.END

    if text.upper() != "YES":
        await update.message.reply_text("Deletion cancelled.")
        return ConversationHandler.END

    try:
        brands = load_brands()
        if brand_name in brands.get("brands", {}):
            del brands["brands"][brand_name]
            save_brands(brands)
            logger.info(f"Brand '{brand_name}' deleted from {BRANDS_CONFIG_PATH}")
            await update.message.reply_text(
                f"*{brand_name}* has been deleted.\n\n"
                f"Use /research\\_add to re-add it with updated details.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(f"Brand '{brand_name}' was already removed.")
    except Exception as e:
        logger.error(f"delete_confirm failed: {e}", exc_info=True)
        await update.message.reply_text(f"Error deleting brand: {e}")
    return ConversationHandler.END


async def delete_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("delete_brand", None)
    await update.message.reply_text("Deletion cancelled.")
    return ConversationHandler.END


# ---- /research_edit (conversation) ----------------------------------------

EDIT_FIELDS = {
    "1": "category",
    "2": "keywords",
    "3": "product_terms",
    "4": "competitors",
    "5": "subreddit_hints",
    "6": "description",
}


async def edit_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return ConversationHandler.END

    if not context.args:
        await update.message.reply_text(
            "Usage: `/research_edit <brand_name>`", parse_mode="Markdown"
        )
        return ConversationHandler.END

    brand_name = " ".join(context.args)
    brands = load_brands().get("brands", {})

    # Case-insensitive lookup
    matched_name = None
    matched_cfg = None
    for name, cfg in brands.items():
        if name.lower() == brand_name.lower():
            matched_name = name
            matched_cfg = cfg
            break

    if not matched_cfg:
        available = ", ".join(brands.keys()) or "None"
        await update.message.reply_text(
            f"Brand '{brand_name}' not found.\n\nAvailable: {available}"
        )
        return ConversationHandler.END

    context.user_data["edit_brand"] = matched_name

    # Show current config and field selection menu
    kws = ", ".join(matched_cfg.get("keywords", []))
    pts = ", ".join(matched_cfg.get("product_terms", []))
    comps = ", ".join(matched_cfg.get("competitors", [])) or "None"
    subs = ", ".join(matched_cfg.get("subreddit_hints", [])) or "None"
    desc = matched_cfg.get("description", "N/A")

    await update.message.reply_text(
        f"*Editing: {matched_name}*\n\n"
        f"1. Category: {matched_cfg.get('category', 'general')}\n"
        f"2. Keywords: {kws}\n"
        f"3. Product terms: {pts}\n"
        f"4. Competitors: {comps}\n"
        f"5. Subreddits: {subs}\n"
        f"6. Description: {desc}\n\n"
        f"*Reply with the number (1-6) of the field to edit,*\n"
        f"or /cancel to abort.",
        parse_mode="Markdown",
    )
    return EDIT_FIELD_SELECT


async def edit_field_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = update.message.text.strip()

    if choice not in EDIT_FIELDS:
        await update.message.reply_text(
            "Please reply with a number from 1 to 6, or /cancel."
        )
        return EDIT_FIELD_SELECT

    field = EDIT_FIELDS[choice]
    context.user_data["edit_field"] = field

    brand_name = context.user_data["edit_brand"]
    brands = load_brands().get("brands", {})
    current_val = brands.get(brand_name, {}).get(field, "")

    if isinstance(current_val, list):
        current_display = ", ".join(current_val) if current_val else "None"
    else:
        current_display = current_val or "None"

    if field == "category":
        prompt = (
            f"*Current {field}:* {current_display}\n\n"
            f"Enter new category:\n"
            f"(beauty / finance / health\\_fitness / food / footwear / tech\\_gadgets / general)\n"
            f"_Or type your own custom category_"
        )
    elif field == "description":
        prompt = f"*Current {field}:* {current_display}\n\nEnter new description:"
    else:
        prompt = (
            f"*Current {field}:* {current_display}\n\n"
            f"Enter new values (comma-separated):"
        )

    await update.message.reply_text(prompt, parse_mode="Markdown")
    return EDIT_FIELD_VALUE


async def edit_field_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    brand_name = context.user_data.pop("edit_brand", None)
    field = context.user_data.pop("edit_field", None)

    if not brand_name or not field:
        await update.message.reply_text("Something went wrong. Please try /research\\_edit again.")
        return ConversationHandler.END

    text = update.message.text.strip()

    try:
        brands = load_brands()
        brand_cfg = brands["brands"].get(brand_name)
        if not brand_cfg:
            await update.message.reply_text(f"Brand '{brand_name}' no longer exists.")
            return ConversationHandler.END

        # Parse the value: lists for most fields, plain string for category/description
        if field in ("category", "description"):
            new_val = text
        else:
            new_val = [v.strip() for v in text.split(",") if v.strip()]

        brand_cfg[field] = new_val
        save_brands(brands)
        logger.info(f"Brand '{brand_name}' field '{field}' updated")

        if isinstance(new_val, list):
            display = ", ".join(new_val)
        else:
            display = new_val

        await update.message.reply_text(
            f"*{brand_name}* — *{field}* updated to:\n{display}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error(f"edit_field_value failed: {e}", exc_info=True)
        await update.message.reply_text(f"Error saving edit: {e}")

    return ConversationHandler.END


async def edit_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("edit_brand", None)
    context.user_data.pop("edit_field", None)
    await update.message.reply_text("Edit cancelled.")
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
    app.add_handler(CommandHandler("research_quick", cmd_research_quick))
    app.add_handler(CommandHandler("research_detailed", cmd_research_detailed))
    app.add_handler(CommandHandler("research_stop", cmd_research_stop))

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

    # Conversation: /research_delete
    delete_conv = ConversationHandler(
        entry_points=[CommandHandler("research_delete", delete_start)],
        states={
            DELETE_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, delete_confirm)],
        },
        fallbacks=[
            CommandHandler("cancel", delete_cancel),
        ],
    )
    app.add_handler(delete_conv)

    # Conversation: /research_edit
    edit_conv = ConversationHandler(
        entry_points=[CommandHandler("research_edit", edit_start)],
        states={
            EDIT_FIELD_SELECT: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_field_select)],
            EDIT_FIELD_VALUE:  [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_field_value)],
        },
        fallbacks=[
            CommandHandler("cancel", edit_cancel),
            CommandHandler("research_edit", edit_start),  # restart mid-conversation
        ],
    )
    app.add_handler(edit_conv)

    # Register commands in Telegram menu
    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("research_quick", "Quick research on a brand (3 months)"),
            BotCommand("research_detailed", "Detailed research with comments & competitive intel"),
            BotCommand("research_add", "Add a new brand"),
            BotCommand("research_edit", "Edit a brand's config"),
            BotCommand("research_delete", "Delete a brand"),
            BotCommand("research_list", "List configured brands"),
            BotCommand("research_stop", "Cancel running research"),
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
