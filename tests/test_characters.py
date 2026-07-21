from src.characters import (
    AUDIO_ONLY_CHARACTERS,
    CHARACTER_IMAGES,
    CHARACTERS,
    IDLE_LOOPS,
    VOICE_MAP,
)


def test_every_character_has_a_prompt():
    for name, data in CHARACTERS.items():
        assert "prompt" in data
        assert isinstance(data["prompt"], str)
        assert len(data["prompt"].strip()) > 0


def test_every_character_has_an_image():
    for name in CHARACTERS:
        assert name in CHARACTER_IMAGES
        assert CHARACTER_IMAGES[name].endswith(".jpg")


def test_every_character_has_a_voice():
    for name in CHARACTERS:
        assert name in VOICE_MAP


def test_audio_only_characters_are_real_characters():
    # every name in AUDIO_ONLY_CHARACTERS must exist in CHARACTERS,
    # so a typo here can't silently create a dead entry
    for name in AUDIO_ONLY_CHARACTERS:
        assert name in CHARACTERS


def test_idle_loops_only_cover_non_audio_only_characters():
    # Iago and The Cave of Wonders intentionally have no idle loop —
    # this test locks in that design decision so it can't regress silently
    for name in IDLE_LOOPS:
        assert name not in AUDIO_ONLY_CHARACTERS
