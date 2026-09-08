"""
Tests for notifier.py's message-splitting logic (the API-calling parts
aren't tested here since they require live Telegram credentials — this
covers the pure logic that caused the real 400 error earlier in development).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import notifier


def test_short_message_not_split():
    text = "Short digest, no signals today."
    chunks = notifier._split_into_chunks(text)
    assert chunks == [text]


def test_long_message_split_into_multiple_chunks():
    blocks = ["*Stock Signal Digest*"]
    for i in range(40):
        blocks.append(f"*TICKER{i}*\n  Some signal explanation text here that takes up meaningful space " * 3)
    long_text = "\n\n".join(blocks)
    assert len(long_text) > notifier._MAX_MESSAGE_LENGTH

    chunks = notifier._split_into_chunks(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= notifier._MAX_MESSAGE_LENGTH


def test_chunks_preserve_all_content():
    blocks = [f"Block {i} with some content" for i in range(30)]
    long_text = "\n\n".join(blocks)
    chunks = notifier._split_into_chunks(long_text, max_length=100)
    rejoined = "\n\n".join(chunks)
    # Every block's content should survive the split/rejoin round trip
    for block in blocks:
        assert block in rejoined


def test_single_block_exceeding_limit_still_returned():
    # A single block longer than max_length can't be split further by this
    # function (it splits on \n\n boundaries only) — it should still be
    # returned as its own chunk rather than silently dropped.
    huge_block = "x" * 5000
    chunks = notifier._split_into_chunks(huge_block, max_length=3800)
    assert huge_block in chunks
