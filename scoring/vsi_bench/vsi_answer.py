"""VSI numerical answer extraction: First find clear conclusions, do not arbitrarily pick numbers from explanations."""

import re

WORDS = dict(
    zip(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
        range(20),
    )
)
WORDS.update(
    dict(
        zip(
            "twenty thirty forty fifty sixty seventy eighty ninety".split(),
            range(20, 100, 10),
        )
    )
)
WORD = r"(?:" + "|".join(WORDS) + r")"
NUMBER = r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|(?=\.\d))(?:\.\d+)?(?:e[-+]?\d+)?(?!\w|\.\d)"
VALUE = (
    rf"(?:{NUMBER}|\b{WORD}(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?\b)"
)
APPROX = r"(?:(?:approximately|about|around|roughly|estimated at|approximately equal to)\s+)*"
UNIT = r"(?:\s*(?:m|cm|meters?|metres?|centimeters?|centimetres?|square meters?|square metres?|sq\.?\s*m|m²)(?:\^?2)?)?"


def _value(text):
    """Only convert numbers or numerals that have been determined as answers; do not treat articles like a/an as 1."""
    text = text.lower().strip()
    if re.fullmatch(NUMBER, text):
        return float(text.replace(",", ""))
    parts = re.split(r"[ -]", text)
    if len(parts) == 1 and parts[0] in WORDS:
        return float(WORDS[parts[0]])
    if (
        len(parts) == 2
        and WORDS.get(parts[0], 0) >= 20
        and 1 <= WORDS.get(parts[1], 0) <= 9
    ):
        return float(WORDS[parts[0]] + WORDS[parts[1]])
    return None


def extract_number(text):
    """Prioritize independent answer lines / clear conclusions; reject guesses for intervals, multi-digit explanations without conclusions."""
    text = re.sub(r"<\|[^>]+\|>", "", str(text)).strip().lower()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    tags = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S)
    if tags:
        text = tags[-1].strip()
    text = text.replace("**", "").replace("`", "").replace("$", "")
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", text)
    candidates = []
    # Preserve line breaks so an explanation followed by a numeric area is supported; ignore intermediate calculations.
    lines = list(re.finditer(r"[^\n]+", text))
    if lines:
        last = lines[-1]
        match = re.fullmatch(rf"\s*{APPROX}({VALUE}){UNIT}\s*[.!]?\s*", last.group())
        if match:
            return _value(match.group(1))
    # Require an explicit object count rather than a number in a compound adjective.
    declaration = rf"(?:\bthere\s+(?:are|is)|\b(?:answer|estimate|size|distance|total|count|number)[^\n.!?]{{0,90}}?\b(?:is|are)|\b(?:answer|total)\s*[:=])\s*{APPROX}({VALUE})"
    for match in re.finditer(declaration, text):
        tail = text[match.end() :]
        # 2-3, 2 to 3 is not the only answer; you cannot secretly take the first endpoint.
        if re.match(rf"\s*(?:[-–—]|to|or|and|/|\+)\s*{VALUE}", tail) or re.match(
            r"\s*(?:hundred|thousand|million|and a half)\b|-[a-z]", tail
        ):
            candidates.append((match.start(), None))
        else:
            candidates.append((match.start(), _value(match.group(1))))
    if candidates:
        return candidates[-1][1]
    # Short "2 stools" can answer counting; long explanations/multidigit text without clear conclusion won't guess.
    numbers = list(re.finditer(VALUE, text))
    if len(numbers) == 1 and len(text) <= 80:
        match = numbers[0]
        if not re.search(
            r"\b(?:image|frame|figure|step)\s*$", text[: match.start()]
        ) and not re.match(
            r"\s*(?:hundred|thousand|million|and a half)\b|-[a-z]", text[match.end() :]
        ):
            return _value(match.group())
    return None
