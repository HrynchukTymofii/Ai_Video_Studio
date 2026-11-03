"""Tagged-script grammar: visual markers vs voice tags, segment assembly, errors."""

import pytest

from app.services.script_parser import (
    ScriptSegment,
    SegmentKind,
    parse_full_video_script,
)
from app.services.validation import InputValidationError


# ---- segment assembly ----------------------------------------------------------


def test_plain_script_is_single_oncamera_segment() -> None:
    segments = parse_full_video_script("Welcome back. Today we look at solar panels.")
    assert segments == [
        ScriptSegment(
            kind=SegmentKind.ONCAMERA,
            text="Welcome back. Today we look at solar panels.",
        )
    ]


def test_full_example_script() -> None:
    script = (
        "[professional broadcast tone] Welcome back to the channel.\n"
        "\n"
        "[BROLL: aerial drone shot of a solar farm at sunset]\n"
        "Across the world, solar capacity has tripled. [short pause]\n"
        "\n"
        "[IMAGE: clean diagram of a photovoltaic cell]\n"
        "Each cell converts photons into electron flow.\n"
        "\n"
        "[CLIP: lab-tour.mp4]\n"
        "Here is our own lab.\n"
        "\n"
        "[ONCAMERA]\n"
        "And that is why the next ten years matter."
    )
    segments = parse_full_video_script(script)
    assert [s.kind for s in segments] == [
        SegmentKind.ONCAMERA,
        SegmentKind.BROLL,
        SegmentKind.IMAGE,
        SegmentKind.CLIP,
        SegmentKind.ONCAMERA,
    ]
    # Voice tags stay in the narration; visual markers are stripped.
    assert segments[0].text == "[professional broadcast tone] Welcome back to the channel."
    assert segments[1].prompt == "aerial drone shot of a solar farm at sunset"
    assert segments[1].text == "Across the world, solar capacity has tripled. [short pause]"
    assert segments[2].prompt == "clean diagram of a photovoltaic cell"
    assert segments[3].clip_name == "lab-tour.mp4"
    assert segments[3].prompt is None
    assert segments[4].text == "And that is why the next ten years matter."


def test_script_may_open_with_a_marker() -> None:
    segments = parse_full_video_script("[BROLL: city at night] Narration over footage.")
    assert len(segments) == 1
    assert segments[0].kind is SegmentKind.BROLL


def test_leading_whitespace_before_first_marker_is_dropped() -> None:
    segments = parse_full_video_script("  \n\n[IMAGE: a chart] Numbers went up.")
    assert [s.kind for s in segments] == [SegmentKind.IMAGE]


def test_voice_tags_never_start_segments() -> None:
    segments = parse_full_video_script(
        "[excited] Big news! [short pause] More text. [whisper in small voice] Quiet now."
    )
    assert len(segments) == 1
    assert segments[0].text == (
        "[excited] Big news! [short pause] More text. [whisper in small voice] Quiet now."
    )


def test_keywords_are_case_insensitive_and_whitespace_tolerant() -> None:
    segments = parse_full_video_script("Intro.\n[ broll : foggy MOUNTAINS ]\nOver footage.")
    assert segments[1].kind is SegmentKind.BROLL
    assert segments[1].prompt == "foggy MOUNTAINS"


def test_hyphen_and_space_keyword_variants() -> None:
    segments = parse_full_video_script(
        "Intro.\n[B-ROLL: waves]\nOver waves.\n[ON CAMERA]\nBack to me."
    )
    assert [s.kind for s in segments] == [
        SegmentKind.ONCAMERA,
        SegmentKind.BROLL,
        SegmentKind.ONCAMERA,
    ]


def test_lowercase_oncamera() -> None:
    segments = parse_full_video_script("Hi.\n[broll: x y z]\nOver.\n[oncamera]\nBack.")
    assert segments[2].kind is SegmentKind.ONCAMERA


def test_prompt_may_contain_colons_and_commas() -> None:
    segments = parse_full_video_script("[BROLL: scene: dramatic, 35mm] Text here.")
    assert segments[0].prompt == "scene: dramatic, 35mm"


def test_no_markers_with_only_voice_tag_text_is_valid() -> None:
    segments = parse_full_video_script("[sigh] Well.")
    assert segments[0].kind is SegmentKind.ONCAMERA


# ---- errors --------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["[BROLL: ]", "[BROLL:]", "[IMAGE:   ]", "[CLIP: ]"])
def test_empty_marker_value_is_rejected(marker: str) -> None:
    with pytest.raises(InputValidationError, match="needs a value"):
        parse_full_video_script(f"Intro. {marker} Text.")


@pytest.mark.parametrize("marker", ["[BROLL]", "[IMAGE]", "[CLIP]"])
def test_known_keyword_without_colon_is_rejected(marker: str) -> None:
    with pytest.raises(InputValidationError, match="missing its value"):
        parse_full_video_script(f"Intro. {marker} Text.")


def test_oncamera_with_value_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="takes no value"):
        parse_full_video_script("Intro. [ONCAMERA: back to host] Text.")


@pytest.mark.parametrize("marker", ["[CHART: sales over time]", "[BROL: factory floor]"])
def test_unknown_colon_marker_is_rejected(marker: str) -> None:
    """A colon means visual marker; unknown/typo keywords must not be spoken."""
    with pytest.raises(InputValidationError, match="unknown visual marker"):
        parse_full_video_script(f"Intro. {marker} Text.")


def test_back_to_back_markers_are_rejected() -> None:
    with pytest.raises(InputValidationError, match="no narration text"):
        parse_full_video_script("Intro. [BROLL: a] [IMAGE: b] Text.")


def test_trailing_marker_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="no narration text"):
        parse_full_video_script("Intro. [BROLL: sunset]")


def test_marker_without_colon_but_with_words_is_caught() -> None:
    """[BROLL cityscape] parses as a voice tag but must not reach TTS."""
    with pytest.raises(InputValidationError, match="malformed visual marker"):
        parse_full_video_script("Intro. [BROLL cityscape] Text.")


def test_unclosed_marker_is_caught() -> None:
    with pytest.raises(InputValidationError, match="malformed visual marker"):
        parse_full_video_script("Intro. [BROLL: sunset over the")


@pytest.mark.parametrize(
    "name", ["../secrets.mp4", "a/b.mp4", "a\\b.mp4", "..\\up.mp4"]
)
def test_clip_name_path_traversal_is_rejected(name: str) -> None:
    with pytest.raises(InputValidationError, match="invalid clip name"):
        parse_full_video_script(f"Intro. [CLIP: {name}] Text.")


def test_timed_pause_marker_stays_in_narration():
    segments = parse_full_video_script("Hello there. [pause:2] [BROLL: city] More text.")
    assert segments[0].text == "Hello there. [pause:2]"
    assert segments[1].text == "More text."
