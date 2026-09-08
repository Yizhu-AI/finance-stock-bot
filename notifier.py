"""
Sends the digest message to Telegram via the Bot API.
Setup: message @BotFather on Telegram to create a bot and get a token,
then message your bot once and use https://api.telegram.org/bot<token>/getUpdates
to find your chat_id.
"""
import requests
import config

# Telegram hard-caps messages at 4096 characters. Leave some headroom.
_MAX_MESSAGE_LENGTH = 3800


def _split_into_chunks(text: str, max_length: int = _MAX_MESSAGE_LENGTH):
    """
    Split on ticker boundaries (blank line before each '*TICKER*' header)
    rather than mid-message, so each chunk stays readable on its own.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > max_length and current:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _send_single(text: str, parse_mode: str = "Markdown"):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    resp = requests.post(url, json=payload, timeout=10)

    if resp.status_code == 400 and parse_mode:
        # Almost always caused by unescaped Markdown special characters in
        # dynamic content (headlines, LLM summaries) breaking Telegram's
        # parser. Fall back to plain text rather than losing the message.
        print("[notifier] Markdown parse failed, retrying as plain text...")
        payload.pop("parse_mode", None)
        resp = requests.post(url, json=payload, timeout=10)

    resp.raise_for_status()


def send_telegram_message(text: str):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("[notifier] Telegram not configured — printing digest instead:\n")
        print(text)
        return

    for i, chunk in enumerate(_split_into_chunks(text), start=1):
        _send_single(chunk)
