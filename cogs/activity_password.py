import discord
from discord import app_commands
from discord.ext import commands
import hashlib
import secrets
from datetime import datetime, timezone

# MUST stay identical to lib/sessions.js on the activity backend.
PBKDF2_ITERATIONS = 100_000
SALT_BYTES = 16
KEY_LEN = 32
MIN_LEN, MAX_LEN = 6, 64


def hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PBKDF2_ITERATIONS,
        dklen=KEY_LEN,
    ).hex()


class ActivityPassword(commands.Cog):
    """Password accounts for the S.I.L.K. Trade Bulletin Board (Discord activity)."""

    activity = app_commands.Group(
        name="activity",
        description="Trade Bulletin Board account management",
    )

    def __init__(self, bot):
        self.bot = bot
        self.db_client = bot.mongo_client
        if self.db_client:
            self.collection = self.db_client["silk_bot"]["activity_accounts"]
        else:
            self.collection = None
            print("Warning: MONGO_URI not found. ActivityPassword module will fail.")

    @activity.command(name="set-password", description="Create or update your Trade Bulletin Board password")
    @app_commands.describe(
        password="New password (6-64 characters)",
        confirm="Type the exact same password again",
    )
    async def set_password(self, interaction: discord.Interaction, password: str, confirm: str):
        if self.collection is None:
            return await interaction.response.send_message("❌ System configuration missing (MongoDB URI).", ephemeral=True)
        if password != confirm:
            return await interaction.response.send_message("❌ Passwords do not match. Nothing was saved.", ephemeral=True)
        if not (MIN_LEN <= len(password) <= MAX_LEN):
            return await interaction.response.send_message(f"❌ Password must be {MIN_LEN}–{MAX_LEN} characters.", ephemeral=True)

        salt_hex = secrets.token_hex(SALT_BYTES)
        hash_hex = hash_password(password, salt_hex)
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"userId": str(interaction.user.id)},
            {
                "$set": {
                    "username": interaction.user.name,
                    "usernameLower": interaction.user.name.lower(),
                    "globalName": interaction.user.global_name,
                    "avatar": str(interaction.user.avatar) if interaction.user.avatar else None,
                    "salt": salt_hex,
                    "hash": hash_hex,
                    "updatedAt": now,
                },
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )
        await interaction.response.send_message(
            "✅ Your Trade Bulletin password is set — use it in the activity's 🔑 LOGIN to post ads.\n"
            "-# Tip: never reuse a password you care about elsewhere.",
            ephemeral=True,
        )

    @activity.command(name="remove-password", description="Delete your Trade Bulletin Board account")
    async def remove_password(self, interaction: discord.Interaction):
        if self.collection is None:
            return await interaction.response.send_message("❌ System configuration missing (MongoDB URI).", ephemeral=True)
        result = await self.collection.delete_one({"userId": str(interaction.user.id)})
        if result.deleted_count:
            await interaction.response.send_message(
                "🗑️ Activity account deleted. Existing ads stay visible, but you can't manage them until you set a new password.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message("ℹ️ You had no activity account.", ephemeral=True)

    @activity.command(name="status", description="Check your Trade Bulletin Board account status")
    async def status(self, interaction: discord.Interaction):
        if self.collection is None:
            return await interaction.response.send_message("❌ System configuration missing (MongoDB URI).", ephemeral=True)
        doc = await self.collection.find_one({"userId": str(interaction.user.id)})
        if not doc:
            return await interaction.response.send_message(
                "ℹ️ No activity account yet. Use `/activity set-password` to create one.", ephemeral=True
            )
        updated = doc.get("updatedAt")
        stamp = updated.strftime("%d %b %Y") if isinstance(updated, datetime) else "unknown date"
        await interaction.response.send_message(
            f"🔑 Account active for username **{doc.get('username')}** (last updated {stamp}).", ephemeral=True
        )


async def setup(bot):
    await bot.add_cog(ActivityPassword(bot))
