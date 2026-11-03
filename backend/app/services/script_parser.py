"""Tagged-script parser for the Full Video assembler.

Splits a narration script into visual segments using inline square-bracket
markers, while leaving S2 voice tags untouched in the narration text. The
distinction is the colon: visual markers carry a value after a colon
([BROLL: aerial city shot]) or are exactly [ONCAMERA]; bracket tokens without
a colon ([short pause], [excited]) are voice tags and stay in the text.

Grammar:

- Plain text            -> spoken on camera (talking head).
- [BROLL: <prompt>]     -> following text is voiced over an AI video clip.
- [IMAGE: <prompt>]     -> following text is voiced over an AI still (slow zoom).
- [CLIP: <filename>]    -> following text is voiced over an uploaded clip.
- [ONCAMERA]            -> return to the talking head.

A marker's segment extends until the next visual marker. Keywords are
case-insensitive; spaces and hyphens inside them are ignored, so
[B-ROLL: ...] and [ON CAMERA] also work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.services.validation import InputValidationError


class SegmentKind(str, Enum):
    ONCAMERA = "oncamera"
    BROLL = "broll"
    IMAGE = "image"
    CLIP = "clip"


@dataclass(frozen=True)
class ScriptSegment:
    kind: SegmentKind
    text: str  # narration including voice tags; visual markers stripped
    prompt: str | None = None  # BROLL / IMAGE generation prompt
    clip_name: str | None = None  # CLIP uploaded-file name, as written


_TOKEN_RE = re.compile(r"\[\s*(?P<kw>[A-Za-z][A-Za-z -]*?)\s*(?::(?P<arg>[^\]]*))?\]")

# Keyword (uppercased, spaces/hyphens removed) -> kind, for markers that take a value.
_ARG_MARKERS = {
    "BROLL": SegmentKind.BROLL,
    "IMAGE": SegmentKind.IMAGE,
    "CLIP": SegmentKind.CLIP,
}

_SUPPORTED = "supported markers are [BROLL: ...], [IMAGE: ...], [CLIP: ...] and [ONCAMERA]"

# Marker-like text that survived tokenizing (missing colon, unclosed bracket, ...)
# would be sent to TTS and spoken aloud — catch it after parsing instead.
_LEFTOVER_RE = re.compile(r"\[\s*(?:B[ -]?ROLL|IMAGE|CLIP|ON[ -]?CAMERA)\b", re.IGNORECASE)


def _normalize_keyword(keyword: str) -> str:
    return re.sub(r"[ -]", "", keyword).upper()


def parse_full_video_script(script: str) -> list[ScriptSegment]:
    """Parse a tagged script into ordered visual segments.

    Raises InputValidationError (user-presentable message) on malformed or
    unknown visual markers, segments with no narration, and unsafe clip names.
    """
    segments: list[ScriptSegment] = []
    kind = SegmentKind.ONCAMERA
    prompt: str | None = None
    clip_name: str | None = None
    opening: str | None = None  # marker literal that opened the current segment
    pieces: list[str] = []

    def finalize() -> None:
        text = "".join(pieces).strip()
        leftover = _LEFTOVER_RE.search(text)
        if leftover:
            snippet = text[leftover.start() : leftover.start() + 40]
            raise InputValidationError(
                f"malformed visual marker near {snippet!r} — markers need a colon"
                f" and a closing bracket, e.g. [BROLL: aerial city shot]"
            )
        if not text:
            if opening is None:
                return  # script opens with a marker; nothing to drop
            raise InputValidationError(f"{opening} has no narration text after it")
        segments.append(
            ScriptSegment(kind=kind, text=text, prompt=prompt, clip_name=clip_name)
        )

    pos = 0
    for match in _TOKEN_RE.finditer(script):
        token = match.group(0)
        keyword = _normalize_keyword(match.group("kw"))
        arg = match.group("arg")

        if keyword == "ONCAMERA":
            if arg is not None:
                raise InputValidationError(
                    f"{token} — [ONCAMERA] takes no value; write it as [ONCAMERA]"
                )
            value = None
        elif keyword in _ARG_MARKERS:
            if arg is None:
                raise InputValidationError(
                    f"{token} is missing its value — write [{keyword}: ...]"
                )
            value = arg.strip()
            if not value:
                raise InputValidationError(
                    f"{token} needs a value — e.g. [{keyword}: aerial city shot]"
                )
            if keyword == "CLIP" and (
                "/" in value or "\\" in value or ".." in value
            ):
                raise InputValidationError(
                    f"invalid clip name {value!r} — use the uploaded file's name only"
                )
        elif arg is not None and keyword != "PAUSE":
            raise InputValidationError(
                f"unknown visual marker {token} — {_SUPPORTED};"
                f" voice tags must not contain a colon"
            )
        else:
            # Voice tag ([short pause], [excited], ...) or timed [pause:2]
            # (handled by the S2 pipeline): keep it in the narration.
            pieces.append(script[pos : match.end()])
            pos = match.end()
            continue

        pieces.append(script[pos : match.start()])
        pos = match.end()
        finalize()

        if keyword == "ONCAMERA":
            kind, prompt, clip_name = SegmentKind.ONCAMERA, None, None
        else:
            kind = _ARG_MARKERS[keyword]
            prompt = value if kind is not SegmentKind.CLIP else None
            clip_name = value if kind is SegmentKind.CLIP else None
        opening = token
        pieces = []

    pieces.append(script[pos:])
    finalize()
    return segments
