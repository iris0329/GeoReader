"""Shared answer extraction for standalone benchmark scoring.

Prefer explicit answer tags or introductions, then standalone option letters in short text."""

from __future__ import annotations

import re
from typing import Optional

CHAT_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>|</?s>")


def strip_chat_special_tokens(text: object) -> str:
    """Remove potentially residual chat template special tokens from model output."""

    # Remove chat-template delimiters before parsing answer values.
    return CHAT_SPECIAL_TOKEN_RE.sub("", str(text or ""))


def _normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", strip_chat_special_tokens(text).strip())


def _extract_from_plain_text(text: str, *, valid_letters: str) -> Optional[str]:
    if not text:
        return None

    # Official/local prompt common formats: A, A., (A), Answer: A, The answer is A.
    direct_patterns = [
        rf"^\s*[\(\[\{{]?([{valid_letters}])[\)\]\}}]?(?:[\.\)]|\s|$)",
        rf"\banswer\s*[:：]\s*[\(\[\{{]?([{valid_letters}])[\)\]\}}]?\b",
        rf"\b(?:the\s+)?answer\s+is\s+[\(\[\{{]?([{valid_letters}])[\)\]\}}]?\b",
        rf"\bmy\s+answer\s+is\s+[\(\[\{{]?([{valid_letters}])[\)\]\}}]?\b",
    ]
    for pattern in direct_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        # Restrict standalone-letter extraction to short lines to avoid letters in explanations.
        if len(line) <= 24:
            match = re.search(rf"\b([{valid_letters}])\b", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()

    compact = _normalize_text(text)
    if len(compact) <= 24:
        match = re.search(rf"\b([{valid_letters}])\b", compact, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def extract_answer(text: object, *, valid_letters: str = "ABCDE") -> Optional[str]:
    """Extract MCQ option letters from model output.

    valid_letters defaulting to E is for compatibility with the 5-option scenario that may appear in CV-Bench; only A-D
    The benchmark passes with default values as well, since GT will not be E.
    """
    raw = strip_chat_special_tokens(text)
    if not raw.strip():
        return None

    valid_letters = "".join(dict.fromkeys(valid_letters.upper()))
    if not valid_letters:
        raise ValueError("valid_letters must not be empty.")

    tag_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.IGNORECASE | re.DOTALL
    )
    if tag_match:
        extracted = extract_answer(tag_match.group(1), valid_letters=valid_letters)
        if extracted is not None:
            return extracted

    # MMSI official prompt requires placing the answer inside ```, and official extraction prioritizes reading backticks or braces.
    for pattern in [r"``([^`]*)``", r"`([^`]*)`", r"\{([^}]*)\}"]:
        match = re.search(pattern, raw, flags=re.DOTALL)
        if match:
            extracted = _extract_from_plain_text(
                match.group(1), valid_letters=valid_letters
            )
            if extracted is not None:
                return extracted

    return _extract_from_plain_text(raw, valid_letters=valid_letters)
