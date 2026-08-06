"""
Tests for vr_service.py — the headless VR API.

Covers the six cases from Part 1.5's checklist: health, characters, a full
chat flow, unknown character, empty/unintelligible audio, and the busy path.

Heavy ML libs are already faked at import time by conftest.py. Here we mock
the specific pipeline functions vr_service.py calls, patched at the point of
use (vr_service.X) rather than their definition module, since vr_service.py
imports them by name (`from src.stt import transcribe`, etc.).
"""

import wave
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import vr_service
from vr_service import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """vr_service keeps module-level session state and a GPU semaphore.
    Both persist across tests unless reset, which would make tests order-
    dependent — clear/restore them before and after every test."""
    vr_service._sessions.clear()
    yield
    vr_service._sessions.clear()
    # Defensive: if a test errors out mid-request without releasing the
    # semaphore, don't let that leak into the next test as a false "busy".
    if vr_service._gpu_slot._value == 0:
        vr_service._gpu_slot.release()


@pytest.fixture
def silent_wav_bytes():
    """A tiny valid WAV file's raw bytes, for the multipart upload — the
    endpoint writes whatever bytes it's given to disk without inspecting
    them (transcribe() is mocked and never actually reads this file)."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence
    return buf.getvalue()


@pytest.fixture
def fake_speak(tmp_path):
    """Mocks src.tts.speak (used in vr_service as `speak`) to return a path
    to a real, valid silent WAV file each call — needed because the endpoint
    genuinely concatenates these with pydub's AudioSegment.from_wav, so the
    files must actually exist and parse, not just be a mocked return value."""
    counter = {"n": 0}

    def _make_wav(text, character_name):
        counter["n"] += 1
        path = tmp_path / f"sentence_{counter['n']}.wav"
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"\x00\x00" * 800)
        return str(path)

    with patch("vr_service.speak", side_effect=_make_wav) as mock:
        yield mock


def _fake_stream(*sentences):
    """A generator standing in for stream_character_reply — yields each
    sentence as a single delta chunk followed by a space, so the endpoint's
    incremental sentence-splitting logic sees complete sentences."""

    def _gen(character_name, user_text, history):
        for s in sentences:
            yield s + " "

    return _gen


# ---------------------------------------------------------------------------
# health / characters
# ---------------------------------------------------------------------------


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "cuda_available" in body
    assert "onnxruntime_version" in body


def test_characters_endpoint():
    response = client.get("/characters")
    assert response.status_code == 200
    body = response.json()
    assert "characters" in body
    names = [c["id"] for c in body["characters"]]
    assert "Genie" in names
    genie = next(c for c in body["characters"] if c["id"] == "Genie")
    assert genie["audio_only"] is False
    iago = next(c for c in body["characters"] if c["id"] == "Iago")
    assert iago["audio_only"] is True


# ---------------------------------------------------------------------------
# unknown character
# ---------------------------------------------------------------------------


def test_vr_chat_unknown_character(silent_wav_bytes):
    response = client.post(
        "/v1/vr-chat-sync",
        data={"character_name": "NotARealCharacter"},
        files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unknown_character"


# ---------------------------------------------------------------------------
# empty / unintelligible audio
# ---------------------------------------------------------------------------


def test_vr_chat_empty_transcript(silent_wav_bytes):
    with patch("vr_service.transcribe", return_value=""):
        response = client.post(
            "/v1/vr-chat-sync",
            data={"character_name": "Genie"},
            files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
        )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "stt_failed"


# ---------------------------------------------------------------------------
# full chat flow — audio-only character (skips video, simpler happy path)
# ---------------------------------------------------------------------------


def test_vr_chat_full_flow_audio_only(silent_wav_bytes, fake_speak):
    with (
        patch("vr_service.transcribe", return_value="What is your wish?"),
        patch(
            "vr_service.stream_character_reply",
            side_effect=_fake_stream(
                "Ah, what a marvelous wish that is indeed!",
                "Bold choice, my friend, very bold indeed.",
            ),
        ),
    ):
        response = client.post(
            "/v1/vr-chat-sync",
            data={"character_name": "Iago"},
            files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["session_id"]
    assert body["user_transcript"] == "What is your wish?"
    assert "marvelous wish" in body["character_transcript"]
    assert body["voice_audio_url"].startswith("/files/")
    assert body["talking_video_url"] is None  # audio-only character
    assert fake_speak.call_count == 2  # one call per sentence, now both ≥15 chars


def test_vr_chat_full_flow_with_video(silent_wav_bytes, fake_speak):
    with (
        patch("vr_service.transcribe", return_value="Tell me a riddle."),
        patch("vr_service.stream_character_reply", side_effect=_fake_stream("Here is one.")),
        patch("vr_service.generate_talking_video", return_value="/tmp/fake.mp4") as mock_video,
    ):
        response = client.post(
            "/v1/vr-chat-sync",
            data={"character_name": "Genie"},
            files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["talking_video_url"].startswith("/files/")
    assert body["talking_video_url"].endswith(".mp4")
    mock_video.assert_called_once()


def test_vr_chat_video_failure_still_returns_audio(silent_wav_bytes, fake_speak):
    """generate_talking_video raising should not fail the whole request —
    per vr_service.py's own except-and-continue, audio should still come
    back with video_error set."""
    with (
        patch("vr_service.transcribe", return_value="Hello"),
        patch("vr_service.stream_character_reply", side_effect=_fake_stream("Greetings.")),
        patch("vr_service.generate_talking_video", side_effect=RuntimeError("GPU OOM")),
    ):
        response = client.post(
            "/v1/vr-chat-sync",
            data={"character_name": "Genie"},
            files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["talking_video_url"] is None
    assert body["video_error"] == "video_failed"
    assert body["voice_audio_url"]  # audio still present


# ---------------------------------------------------------------------------
# session continuity
# ---------------------------------------------------------------------------


def test_vr_chat_session_id_persists(silent_wav_bytes, fake_speak):
    with (
        patch("vr_service.transcribe", return_value="Hi"),
        patch("vr_service.stream_character_reply", side_effect=_fake_stream("Hello there.")),
    ):
        r1 = client.post(
            "/v1/vr-chat-sync",
            data={"character_name": "Iago"},
            files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
        )
        session_id = r1.json()["session_id"]

        r2 = client.post(
            "/v1/vr-chat-sync",
            data={"character_name": "Iago", "session_id": session_id},
            files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
        )

    assert r2.json()["session_id"] == session_id


# ---------------------------------------------------------------------------
# busy / 429 path
# ---------------------------------------------------------------------------


def test_vr_chat_returns_busy_when_semaphore_held(silent_wav_bytes):
    vr_service._gpu_slot.acquire()
    try:
        response = client.post(
            "/v1/vr-chat-sync",
            data={"character_name": "Genie"},
            files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 429
        assert response.json()["detail"]["code"] == "busy"
    finally:
        vr_service._gpu_slot.release()

def test_vr_chat_short_sentences_get_folded_together(silent_wav_bytes, fake_speak):
    """Sentences under MIN_SENTENCE_LENGTH (15 chars) get merged before
    speaking — a real behavior of split_into_sentences that's worth locking
    in at the endpoint level too, not just in test_sentence_splitter.py."""
    with (
        patch("vr_service.transcribe", return_value="Hi"),
        patch("vr_service.stream_character_reply", side_effect=_fake_stream("Ah, a wish!", "Bold choice.")),
    ):
        response = client.post(
            "/v1/vr-chat-sync",
            data={"character_name": "Iago"},
            files={"audio_file": ("test.wav", silent_wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    assert fake_speak.call_count == 1  # folded into a single spoken chunk        