import discord
from dotenv import load_dotenv
from io import BytesIO
import os
import sys

from script_ocr import (
    load_character_mapping,
    bytesio_to_cv2_image,
    script_image_to_json,
    compress_json,
)


async def process_json_request(
        interaction: discord.Interaction,
        attached_image: discord.Attachment,
        character_mapping):

    await interaction.response.defer()

    if attached_image.content_type is None or not attached_image.content_type.startswith("image/"):
        await interaction.followup.send(
            "Please upload an image.",
            ephemeral=True
        )
        return

    try:
        image = bytesio_to_cv2_image(BytesIO(await attached_image.read()))
    except Exception:
        await interaction.followup.send(
            "Something went wrong.",
            ephemeral=True
        )
        return

    try:
        (script_name, author, script_json) = script_image_to_json(character_mapping, image)

        reply_body = ""
        if script_name:
            reply_body = script_name
            if author:
                reply_body += f" by {author}"
            reply_body += "\n"

        reply_body += f"```json\n{script_json}\n```"

        url = f"https://script.bloodontheclocktower.com?script={compress_json(script_json)}"

        embed = discord.Embed(
            description=f"[Open in Script Tool]({url})"
        )

        await interaction.followup.send(
            reply_body,
            embed=embed
        )

    except Exception:
        await interaction.followup.send(
            "Something went wrong.",
            ephemeral=True
        )


def main():
    load_dotenv()

    json_bot_token = os.getenv("JSON_BOT_TOKEN")
    if json_bot_token is None:
        raise RuntimeError("JSON_BOT_TOKEN not found")

    try:
        character_mapping = load_character_mapping('characters.tsv')
    except Exception as e:
        print("Error loading character data")
        sys.exit(1)

    intents = discord.Intents.default()

    client = discord.Client(intents=intents)
    tree = discord.app_commands.CommandTree(client)

    synced = False

    @client.event
    async def on_ready():
        global synced

        if not synced:
            await tree.sync()
            synced = True

        print(f"Logged in as {client.user}")

    @tree.command(name="json", description="Convert a script image to JSON")
    async def json_command(
        interaction: discord.Interaction,
        image: discord.Attachment
    ):
        await process_json_request(interaction, image, character_mapping)

    client.run(json_bot_token)


if __name__ == "__main__":
    main()
