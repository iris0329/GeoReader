"""MindCube only reads the final options explicitly provided by the model and does not guess answers using GT."""

import re


def extract_mindcube_answer(text):
    """Return the final letter and justification; options lists, negations, or multi-select ambiguities are not considered final answers."""
    # Remove special-token delimiters before extracting an option letter.
    clean = re.sub(r"<\|[^>]+\|>", "", str(text)).strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.S).strip()
    clean = clean.replace("**", "").replace("__", "").replace("`", "")
    candidates = []
    # Recognize answer letters inside supported angle-bracket formats.
    wrapped = re.fullmatch(
        r"<([A-D])(?:[.)]\s*([^<>]*)>|>\s*([^<>]*)</(?:\1|answer)>)", clean
    )
    if wrapped:
        content = wrapped.group(2) or wrapped.group(3) or ""
        if re.search(r"\b[A-D][.)]\s|\b(?:or|and)\s+[A-D]\b", content):
            return None, "ambiguous_final"
        return wrapped.group(1), "single_wrapped_option"
    # The answer tag takes precedence over option enumerations in the body; the last declaration applies.
    for m in re.finditer(
        r"<answer>\s*([A-D])(?:\s*[.)]\s*([^<]*))?\s*</answer>", clean, re.I
    ):
        # <answer>B. Window</answer>
        content = m.group(2) or ""
        ambiguous = re.search(r"\b[A-D][.)]\s", content) or re.match(
            r"\s*(?:or|and)\s+[A-D]\b", content
        )
        candidates.append(
            (
                m.start(),
                None if ambiguous else m.group(1).upper(),
                "ambiguous_final" if ambiguous else "answer_tag",
            )
        )
    declaration = re.compile(
        r"\b(?i:(?:(?:the\s+)?(?:final|correct)\s+)?(?:answer|option|choice)\s*(?:is\s*:?|would\s+be\s*:?|:|=)\s*(?:option\s+)?)\(?([A-D])\)?(?=$|[\s.,:;)])"
    )
    # "Behind you is option B" is also a clear choice; do not treat the ordinary "Option B is ..." as an enumeration of choices.
    direct_option = re.compile(
        r"\b(?i:(?:is|would\s+be)\s+option\s+)\(?([A-D])\)?(?=$|[\s.,:;)])"
    )
    for m in list(declaration.finditer(clean)) + list(direct_option.finditer(clean)):
        # "answer is A or B" is not the only option; you cannot arbitrarily choose A.
        tail = clean[m.end() :]
        if re.match(r"\s*(?:or|and|/|,)\s*[A-D]\b", tail):
            candidates.append((m.start(), None, "ambiguous_final"))
        else:
            candidates.append((m.start(), m.group(1), "explicit_final"))
    lines = [
        (m.start(), m.group().strip())
        for m in re.finditer(r"[^\n]+", clean)
        if m.group().strip()
    ]
    if lines:
        # Accept a unique standalone option after an explicit answer introduction.
        option_lines = [
            (i, re.fullmatch(r"\(?([A-D])[.)]\s+[^\n]+", line))
            for i, (_, line) in enumerate(lines)
        ]
        option_lines = [(i, m) for i, m in option_lines if m]
        if len(option_lines) == 1:
            i, option_line = option_lines[0]
            if i > 0 and re.search(
                r"(?i)\b(?:would be|is|are|answer)\s*:\s*$", lines[i - 1][1]
            ):
                if not re.search(
                    r"\b[A-D][.)]\s", lines[i][1][option_line.end(1) + 1 :]
                ):
                    candidates.append(
                        (lines[i][0], option_line.group(1), "introduced_answer_line")
                    )
        # Accept an initial answer option followed by an explanation, excluding option enumerations.
        first_start, first = lines[0]
        leading = re.fullmatch(r"\(?([A-D])\)?[.!]?", first)
        if leading is None:
            leading = re.fullmatch(r"\(?([A-D])[.)]\s+[^\n]+", first)
        later_options = any(
            re.match(r"(?:[-*]\s*)?\(?[A-D][.)]\s+", line) for _, line in lines[1:]
        )
        if (
            leading
            and not later_options
            and not re.search(r"\b[A-D][.)]\s", first[leading.end(1) + 1 :])
        ):
            candidates.append((first_start, leading.group(1), "leading_answer"))
        start, last = lines[-1]
        bare = re.fullmatch(r"\(?([A-D])\)?[.!]?", last)
        if bare:
            candidates.append((start, bare.group(1), "terminal_letter"))
        else:
            option = re.fullmatch(r"\(?([A-D])[.)]\s+[^\n]+", last)
            # Single line "B. Lectern" may be an answer; the last item in multi-line A/B/C/D lists cannot be considered an answer.
            if option and not re.search(r"\b[A-D][.)]\s", last[option.start(1) + 2 :]):
                preceding = [line for _, line in lines[:-1]]
                # Only check the immediately preceding non-empty line; earlier enumerated analyses should not obscure the standalone answer given at the end.
                previous_is_option = bool(
                    preceding
                    and re.match(r"(?:[-*]\s*)?\(?[A-D][.)]\s+", preceding[-1])
                )
                if not previous_is_option:
                    candidates.append((start, option.group(1), "terminal_option"))
    if not candidates:
        return None, "no_explicit_final"
    _, letter, reason = max(candidates, key=lambda item: item[0])
    return letter, reason
