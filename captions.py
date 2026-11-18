"""Generate local, word-timed Advanced SubStation Alpha captions."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable


HIGHLIGHT_COLORS = {
    "yellow": "&H0000D7FF",
    "cyan": "&H00FFFF00",
    "green": "&H0000E676",
    "white": "&H00FFFFFF",
}


def _ass_time(seconds: float) -> str:
    value = max(0, round(seconds * 100))
    hours, remainder = divmod(value, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{whole_seconds:02}.{centiseconds:02}"


def _safe_text(value: str) -> str:
    return re.sub(r"[{}]", "", value).replace("\\", r"\\").strip()


def _normalized_words(words: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for word in words:
        text = str(word.get("word", "")).strip()
        if not text:
            continue
        try:
            start = max(0.0, float(word["start"]))
            end = max(start + 0.01, float(word["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        result.append({"word": text, "start": start, "end": end})
    return result


def words_from_caption(text: str, duration: float) -> list[dict[str, Any]]:
    tokens = re.findall(r"\S+", text.strip())
    if not tokens:
        return []
    duration = max(0.1, duration)
    step = duration / len(tokens)
    return [
        {
            "word": token,
            "start": index * step,
            "end": min(duration, (index + 1) * step),
        }
        for index, token in enumerate(tokens)
    ]


def write_ass(
    words: Iterable[dict[str, Any]],
    output_path: str | Path,
    highlight_color: str = "yellow",
    max_words_per_line: int = 5,
) -> Path:
    """Write karaoke-style captions; input word timestamps are clip-relative."""
    normalized = _normalized_words(words)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    highlight = HIGHLIGHT_COLORS.get(highlight_color.lower(), HIGHLIGHT_COLORS["yellow"])
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Arial,86,&H00FFFFFF,{highlight},&H00111111,"
        "&H99000000,-1,0,0,0,100,100,0,0,1,7,3,2,70,70,230,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for offset in range(0, len(normalized), max(1, max_words_per_line)):
        group = normalized[offset : offset + max_words_per_line]
        if not group:
            continue
        start = group[0]["start"]
        end = max(group[-1]["end"], start + 0.05)
        karaoke: list[str] = []
        for item in group:
            centiseconds = max(1, round((item["end"] - item["start"]) * 100))
            karaoke.append(f"{{\\kf{centiseconds}}}{_safe_text(item['word']).upper()}")
        lines.append(
            "Dialogue: 0,"
            f"{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
            + " ".join(karaoke)
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
