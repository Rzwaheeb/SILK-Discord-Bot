"""AI Logs — separate logging cog for AoTRGPT interactions.

This cog listens for the custom 'aotr_gpt_log' event dispatched by AoTRGPT.
It creates a Discord thread in a configured log channel and writes each log
category as separate messages.

No AI API calls are made here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands

logger = logging.getLogger(__name__)


class AILogs(commands.Cog):
    """Global AoTR AI interaction logger."""

    CONFIG_PATH = Path("data/ai_logs_config.json")
    THREAD_ARCHIVE_MINUTES = 60
    SEND_DELAY_SECONDS = 0.35
    MESSAGE_CHUNK_LIMIT = 1750
    QUEUE_MAX_SIZE = 1000

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.QUEUE_MAX_SIZE)
        self._worker: asyncio.Task | None = None

    async def cog_load(self) -> None:
        self._worker = asyncio.create_task(self._worker_loop())

    async def cog_unload(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker

    # ── Configuration Command ─────────────────────────────────────────
    @commands.command(name="setailogs")
    @commands.is_owner()
    @commands.guild_only()
    async def setailogs(self, ctx: commands.Context) -> None:
        """Set this channel as the global AoTR AI log channel."""
        channel = ctx.channel

        if isinstance(channel, discord.Thread):
            channel = channel.parent

        if not isinstance(channel, discord.TextChannel):
            await ctx.reply("❌ Please run this in a normal text channel.", mention_author=False)
            return

        await asyncio.to_thread(self._save_config, ctx.guild.id, channel.id)

        perms = channel.permissions_for(ctx.guild.me)
        missing: list[str] = []

        if not perms.send_messages:
            missing.append("Send Messages")
        if not perms.create_public_threads:
            missing.append("Create Public Threads")
        if not perms.send_messages_in_threads:
            missing.append("Send Messages in Threads")

        if missing:
            await ctx.reply(
                f"✅ Log channel saved, but I am missing permissions: {', '.join(missing)}.",
                mention_author=False,
            )
            return

        await ctx.reply(
            f"✅ AoTR AI logs will now be sent to {channel.mention}.",
            mention_author=False,
        )

    @setailogs.error
    async def setailogs_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.NotOwner):
            await ctx.reply("❌ Only the bot owner can set AI logs.", mention_author=False)
        elif isinstance(error, commands.NoPrivateMessage):
            await ctx.reply("❌ This command must be used in a server.", mention_author=False)
        else:
            logger.exception("setailogs failed.", exc_info=error)
            await ctx.reply("⚠️ Failed to set AI log channel.", mention_author=False)

    # ── Event Listener ────────────────────────────────────────────────
    @commands.Cog.listener()
    async def on_aotr_gpt_log(self, payload: dict[str, Any]) -> None:
        """Receive log payload from AoTRGPT and queue it."""
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning(
                "AI log queue full; dropping log_id=%s",
                payload.get("log_id", "unknown"),
            )

    # ── Background Worker ─────────────────────────────────────────────
    async def _worker_loop(self) -> None:
        while True:
            payload = await self._queue.get()

            try:
                await self._write_log(payload)
            except Exception:
                logger.exception("Failed to write AI log.")
            finally:
                self._queue.task_done()
                await asyncio.sleep(self.SEND_DELAY_SECONDS)

    # ── Log Writing ───────────────────────────────────────────────────
    async def _write_log(self, payload: dict[str, Any]) -> None:
        cfg = await asyncio.to_thread(self._load_config)

        if cfg is None:
            return

        channel = self.bot.get_channel(cfg["channel_id"])

        if channel is None:
            try:
                channel = await self.bot.fetch_channel(cfg["channel_id"])
            except Exception:
                logger.warning("AI log channel could not be fetched.")
                return

        if not isinstance(channel, discord.TextChannel):
            logger.warning("AI log channel is not a text channel.")
            return

        if channel.guild.id != cfg["guild_id"]:
            logger.warning("AI log channel guild mismatch.")
            return

        perms = channel.permissions_for(channel.guild.me)

        if not (
            perms.send_messages
            and perms.create_public_threads
            and perms.send_messages_in_threads
        ):
            logger.warning("Missing permissions for AI log thread creation/sending.")
            return

        thread_name = self._thread_name(payload)

        try:
            thread = await channel.create_thread(
                name=thread_name,
                auto_archive_duration=self.THREAD_ARCHIVE_MINUTES,
                reason="AoTR AI interaction log",
            )
        except discord.HTTPException:
            logger.exception("Failed to create AI log thread.")
            return

        await self._send_summary(thread, payload)
        await self._send_query(thread, payload)
        await self._send_documents(thread, payload)
        await self._send_answer(thread, payload)

        if payload.get("error"):
            await self._send_error(thread, payload)

    # ── Log Sections ──────────────────────────────────────────────────
    async def _send_summary(self, thread: discord.Thread, payload: dict[str, Any]) -> None:
        latency = payload.get("latency") or {}

        lines = [
            f"Log ID: {payload.get('log_id')}",
            f"Timestamp: {payload.get('timestamp_utc')}",
            f"Guild: {payload.get('guild_name')} ({payload.get('guild_id')})",
            f"Channel: <#{payload.get('channel_id')}> ({payload.get('channel_id')})",
            f"User: {payload.get('user_name')} ({payload.get('user_id')})",
            f"Cache Hit: {payload.get('cache_hit')}",
            f"Confidence: {payload.get('confidence')}",
            f"Top Score: {payload.get('top_score')}",
            f"Thresholds: high={payload.get('high_threshold')} low={payload.get('low_threshold')}",
        ]

        if latency:
            latency_text = ", ".join(f"{key}={value}" for key, value in latency.items())
            lines.append(f"Latency: {latency_text}")

        await self._send_section(thread, "Summary", "\n".join(lines))

    async def _send_query(self, thread: discord.Thread, payload: dict[str, Any]) -> None:
        await self._send_section(thread, "User Query", payload.get("query") or "None")

    async def _send_documents(self, thread: discord.Thread, payload: dict[str, Any]) -> None:
        documents = payload.get("documents") or []

        if not documents:
            await self._send_section(thread, "Retrieved Documents", "No retrieved documents.")
            return

        blocks: list[str] = []

        for doc in documents:
            block_lines = [
                f"#{doc.get('rank')} | score={doc.get('score')} | id={doc.get('id')} | name={doc.get('name')}"
            ]

            snippet = doc.get("snippet")
            if snippet:
                block_lines.append(snippet)

            blocks.append("\n".join(block_lines))

        await self._send_section(thread, "Retrieved Documents", "\n\n".join(blocks))

    async def _send_answer(self, thread: discord.Thread, payload: dict[str, Any]) -> None:
        await self._send_section(thread, "Final Answer", payload.get("answer") or "None")

    async def _send_error(self, thread: discord.Thread, payload: dict[str, Any]) -> None:
        await self._send_section(thread, "Error", payload.get("error") or "Unknown error")

    # ── Formatting Helpers ────────────────────────────────────────────
    async def _send_section(self, thread: discord.Thread, title: str, body: str) -> None:
        body = discord.utils.escape_mentions(str(body or ""))
        header = f"## {title}"

        if len(header) + len(body) + 1 <= self.MESSAGE_CHUNK_LIMIT:
            await thread.send(
                f"{header}\n{body}",
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return

        chunks = self._split_text(body, self.MESSAGE_CHUNK_LIMIT)

        for idx, chunk in enumerate(chunks, 1):
            part_header = f"## {title} ({idx}/{len(chunks)})"
            await thread.send(
                f"{part_header}\n{chunk}",
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await asyncio.sleep(0.15)

    def _split_text(self, text: str, limit: int) -> list[str]:
        if limit <= 0:
            limit = 1750

        if len(text) <= limit:
            return [text]

        chunks: list[str] = []

        while text:
            if len(text) <= limit:
                chunks.append(text)
                break

            window = text[:limit]
            split_pos = window.rfind("\n")

            if split_pos <= limit // 2:
                split_pos = window.rfind(" ")

            if split_pos <= 0:
                split_pos = limit

            chunks.append(text[:split_pos].rstrip())
            text = text[split_pos:].lstrip()

        return chunks

    def _thread_name(self, payload: dict[str, Any]) -> str:
        query = str(payload.get("query") or "no-query").replace("\n", " ").strip()
        query = discord.utils.escape_mentions(query)

        if len(query) > 45:
            query = query[:42] + "..."

        status = "ERR" if payload.get("error") else str(payload.get("confidence") or "LOG")
        timestamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")

        name = f"[{timestamp}] {payload.get('log_id', '?')} {status} {query}"
        return name[:100]

    # ── Config Storage ────────────────────────────────────────────────
    def _load_config(self) -> dict[str, int] | None:
        try:
            if not self.CONFIG_PATH.exists():
                return None

            data = json.loads(self.CONFIG_PATH.read_text(encoding="utf-8"))

            return {
                "guild_id": int(data["guild_id"]),
                "channel_id": int(data["channel_id"]),
            }
        except Exception:
            logger.exception("Failed to load AI log config.")
            return None

    def _save_config(self, guild_id: int, channel_id: int) -> None:
        try:
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.CONFIG_PATH.write_text(
                json.dumps(
                    {
                        "guild_id": guild_id,
                        "channel_id": channel_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to save AI log config.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AILogs(bot))