"""
Headless Backend Service

REST endpoint replacing the Gradio UI for non-browser clients — per the
supervisor's request: "make a service that sends audio, gets back to us
transcript, voice, and video." Reuses the existing pipeline modules (stt,
llm, tts, lipsync) exactly as app.py does; Gradio's own UI loop can't be
driven by a non-browser client, hence a plain HTTP/JSON API instead.

Two design choices worth flagging if a specific client (Unity, a web
frontend, etc.) is decided on later:

1. Per-session conversation history, keyed by session_id (not shared
   globally per character) — required to support more than one concurrent
   user correctly; each session gets independent conversation memory.
2. Audio/video are returned as URLs to static files, not embedded as
   base64 in the JSON body. This keeps the response small and lets a
   client fetch large binary payloads (especially video) as a separate,
   simple GET rather than parsing them out of one large JSON blob — a
   reasonable default regardless of what ends up consuming this API,
   though the ideal transport could be revisited once an actual client
   is chosen.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from src.characters import AUDIO_ONLY_CHARACTERS, CHARACTER_IMAGES, CHARACTERS
from src.lipsync import generate_talking_video
from src.llm import init_conversation_histories, stream_character_reply
from src.sentence_splitter import split_into_sentences
from src.stt import transcribe
from src.tts import speak
from datetime import datetime, timezone
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Sanity check at startup: confirm GPU execution is actually available. ---
# This is here because a silent CPU fallback (wrong onnxruntime package
# installed/shadowing onnxruntime-gpu) has bitten this project before —
# fail loudly at startup instead of discovering it mid-demo.
import onnxruntime  # noqa: E402

_providers = onnxruntime.get_available_providers()
if "CUDAExecutionProvider" not in _providers:
    log.warning(
        "CUDAExecutionProvider NOT available (got: %s). "
        "Video generation will silently run on CPU and be much slower. "
        "Check: pip uninstall onnxruntime onnxruntime-gpu -y && "
        "pip install onnxruntime-gpu==1.26.0",
        _providers,
    )
else:
    log.info("CUDAExecutionProvider confirmed available.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_cleanup_loop, daemon=True)
    thread.start()
    log.info(
        "Cleanup thread started (interval=%ds, file max age=%ds, session max idle=%ds).",
        CLEANUP_INTERVAL_SECONDS,
        OUTPUT_MAX_AGE_SECONDS,
        SESSION_MAX_IDLE_SECONDS,
    )
    yield
    # Nothing needed on shutdown — the thread is a daemon, it dies with the process.


app = FastAPI(title="VR Character Animation Headless Service", lifespan=lifespan)

# Directory for generated audio/video files served back to VR clients as URLs.
OUTPUT_DIR = Path(tempfile.gettempdir()) / "idiscovr_vr_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")
app.mount("/character_images", StaticFiles(directory="character_images"), name="character_images")


@app.get("/health")
def health_endpoint():
    """Are the models loaded, is CUDA available. Used as the Docker healthcheck
    and by any client to know when the service is actually ready."""
    return {
        "status": "ok",
        "cuda_available": "CUDAExecutionProvider" in _providers,
        "onnxruntime_version": onnxruntime.__version__,
    }


@app.get("/characters")
def characters_endpoint():
    """id, display name, whether it's audio-only, and a portrait URL for each
    character — so a client doesn't have to hardcode character names."""
    return {
        "characters": [
            {
                "id": name,
                "display_name": name,
                "audio_only": name in AUDIO_ONLY_CHARACTERS,
                "portrait_url": (
                    f"/character_images/{os.path.basename(CHARACTER_IMAGES[name])}"
                    if name in CHARACTER_IMAGES
                    else None
                ),
            }
            for name in CHARACTERS
        ]
    }


# --- Per-session state ---
# session_id -> {character_name: history_list}
# Each session gets its own full set of per-character histories, created
# lazily on first use. A lock guards session creation/lookup since FastAPI
# can serve concurrent requests; per-session history mutation itself is not
# additionally locked here — concurrent requests within the SAME session_id
# for the SAME character are assumed not to overlap (one VR user, one mic
# input at a time). Flag this explicitly if that assumption stops holding.
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()

# Only one request may run the actual pipeline (Whisper, TTS, the Wav2Lip
# ONNX session) at a time. Concurrent requests would otherwise contend for
# the same GPU/VRAM — the exact effect measured during the LatentSync
# evaluation, where steps ran ~3x slower under simultaneous load. A second
# request while one is in flight gets a fast 429 "busy" response instead of
# silently queueing behind degraded work.
_gpu_slot = threading.Semaphore(1)

# --- Cleanup: prevent unbounded growth of OUTPUT_DIR and _sessions ---
OUTPUT_MAX_AGE_SECONDS = 30 * 60      # delete generated files older than this
SESSION_MAX_IDLE_SECONDS = 30 * 60    # evict sessions untouched for this long
CLEANUP_INTERVAL_SECONDS = 5 * 60     # how often the background sweep runs

# Tracks last-touched time per session_id, kept separate from _sessions
# itself so get_or_create_session's existing return shape (session_id,
# histories_dict) doesn't change and nothing else has to know about this.
_session_last_used: dict[str, float] = {}


def cleanup_once(now: float | None = None) -> dict:
    """
    One sweep: delete files in OUTPUT_DIR older than OUTPUT_MAX_AGE_SECONDS,
    evict _sessions entries idle longer than SESSION_MAX_IDLE_SECONDS.

    Takes `now` as an optional injectable timestamp so tests can simulate
    age without sleeping for real minutes. Returns counts for logging/tests.
    """
    now = now if now is not None else time.time()

    deleted_files = 0
    for path in OUTPUT_DIR.iterdir():
        if not path.is_file():
            continue
        try:
            age = now - path.stat().st_mtime
        except FileNotFoundError:
            continue  # deleted concurrently, fine
        if age > OUTPUT_MAX_AGE_SECONDS:
            try:
                path.unlink()
                deleted_files += 1
            except FileNotFoundError:
                pass

    evicted_sessions = 0
    with _sessions_lock:
        stale_ids = [
            sid
            for sid, last_used in _session_last_used.items()
            if now - last_used > SESSION_MAX_IDLE_SECONDS
        ]
        for sid in stale_ids:
            _sessions.pop(sid, None)
            _session_last_used.pop(sid, None)
            evicted_sessions += 1

    if deleted_files or evicted_sessions:
        log.info(
            "Cleanup sweep: deleted %d output file(s), evicted %d idle session(s).",
            deleted_files,
            evicted_sessions,
        )
    return {"deleted_files": deleted_files, "evicted_sessions": evicted_sessions}


def _cleanup_loop():
    """Runs forever in a background thread, sweeping every CLEANUP_INTERVAL_SECONDS."""
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            cleanup_once()
        except Exception:
            log.error("Cleanup sweep failed", exc_info=True)

def get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    """Return (session_id, histories_dict). Creates a new session if the
    given session_id is missing or unknown."""
    with _sessions_lock:
        if session_id and session_id in _sessions:
            _session_last_used[session_id] = time.time()
            return session_id, _sessions[session_id]
        new_id = session_id or str(uuid.uuid4())
        _sessions[new_id] = init_conversation_histories()
        _session_last_used[new_id] = time.time()
        return new_id, _sessions[new_id]


@app.post("/v1/vr-chat-sync")
def vr_chat_sync_endpoint(
    character_name: str = Form(...),
    audio_file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
):
    """
    Headless endpoint for VR headsets.

    Accepts mic audio + optional session_id (omit on first call). Returns
    JSON with the session_id to reuse on subsequent calls (so conversation
    memory persists per-user), plus URLs to the generated voice audio and
    lip-synced video — not the raw bytes.
    """
    if character_name not in CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail={"code": "unknown_character", "message": f"Character '{character_name}' not found."},
        )

    effective_session_id, histories = get_or_create_session(session_id)
    history = histories[character_name]

    if not _gpu_slot.acquire(timeout=0.1):
        raise HTTPException(status_code=429, detail={"code": "busy"})

    temp_mic_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    request_id = uuid.uuid4().hex
    combined_wav_path = str(OUTPUT_DIR / f"{request_id}.wav")
    sentence_audio_paths = []

    try:
        with open(temp_mic_path, "wb") as f:
            f.write(audio_file.file.read())

        user_text = transcribe(temp_mic_path)
        if not user_text:
            raise HTTPException(
                status_code=400,
                detail={"code": "stt_failed", "message": "Could not understand incoming VR mic data."},
            )

        full_assistant_reply = ""
        buffer = ""
        for delta in stream_character_reply(character_name, user_text, history):
            buffer += delta
            sentences, buffer = split_into_sentences(buffer)
            for sentence in sentences:
                full_assistant_reply += sentence + " "
                sentence_audio_paths.append(speak(sentence, character_name))

        if buffer.strip():
            full_assistant_reply += buffer.strip()
            sentence_audio_paths.append(speak(buffer.strip(), character_name))

        if not sentence_audio_paths:
            raise HTTPException(
                status_code=500,
                detail={"code": "llm_unavailable", "message": "Failed to synthesize vocal replies."},
            )

        combined = AudioSegment.empty()
        for path in sentence_audio_paths:
            combined += AudioSegment.from_wav(path)
        combined.export(combined_wav_path, format="wav")

        for path in sentence_audio_paths:
            if os.path.exists(path):
                os.remove(path)

        video_url = None
        video_error = None
        if character_name not in AUDIO_ONLY_CHARACTERS:
            try:
                video_filename = f"{request_id}.mp4"
                video_output_path = str(OUTPUT_DIR / video_filename)
                generate_talking_video(character_name, combined_wav_path, output_path=video_output_path)
                video_url = f"/files/{video_filename}"
            except Exception as e:
                log.error("Video generation failed: %s", e, exc_info=True)
                video_error = "video_failed"

        return JSONResponse(
            content={
                "status": "success",
                "session_id": effective_session_id,
                "character": character_name,
                "user_transcript": user_text,
                "character_transcript": full_assistant_reply.strip(),
                "voice_audio_url": f"/files/{request_id}.wav",
                "talking_video_url": video_url,
                "video_error": video_error,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("VR pipeline error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"code": "pipeline_failed", "message": "Internal pipeline error."},
        )

    finally:
        _gpu_slot.release()
        if os.path.exists(temp_mic_path):
            os.remove(temp_mic_path)


def _format_sse(event: str, data: dict) -> str:
    """One Server-Sent Event: an event name line plus a JSON data line, blank-line terminated."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/vr-chat-stream")
def vr_chat_stream_endpoint(
    character_name: str = Form(...),
    audio_file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
):
    """
    Same pipeline as /v1/vr-chat-sync, but sends one SSE event per stage as
    it becomes ready instead of one JSON blob at the end.
    """
    if character_name not in CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail={"code": "unknown_character", "message": f"Character '{character_name}' not found."},
        )

    effective_session_id, histories = get_or_create_session(session_id)
    history = histories[character_name]

    if not _gpu_slot.acquire(timeout=0.1):
        raise HTTPException(status_code=429, detail={"code": "busy"})

    temp_mic_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with open(temp_mic_path, "wb") as f:
        f.write(audio_file.file.read())

    def event_stream():
        sentence_files = []
        t_start = time.time()
        try:
            user_text = transcribe(temp_mic_path)
            t_stt = time.time()
            log.info("TIMING stt=%.2fs", t_stt - t_start)
            if not user_text:
                yield _format_sse("error", {"code": "stt_failed", "message": "Could not understand incoming audio."})
                return

            yield _format_sse("transcript", {"session_id": effective_session_id, "user_transcript": user_text})

            full_assistant_reply = ""
            buffer = ""

            def handle_sentence(sentence_text: str) -> str:
                nonlocal full_assistant_reply
                full_assistant_reply += sentence_text + " "
                raw_path = speak(sentence_text, character_name)
                sentence_files.append(raw_path)
                request_id = uuid.uuid4().hex
                public_path = OUTPUT_DIR / f"{request_id}.wav"
                shutil.copy(raw_path, public_path)
                return request_id

            t_first_token = None
            t_first_sentence = None
            for delta in stream_character_reply(character_name, user_text, history):
                if t_first_token is None:
                    t_first_token = time.time()
                    log.info("TIMING first_llm_token=%.2fs (since request start)", t_first_token - t_start)
                buffer += delta
                sentences, buffer = split_into_sentences(buffer)
                for sentence in sentences:
                    request_id = handle_sentence(sentence)
                    if t_first_sentence is None:
                        t_first_sentence = time.time()
                        log.info("TIMING first_sentence_ready=%.2fs (since request start)", t_first_sentence - t_start)
                    yield _format_sse("sentence_audio", {"text": sentence, "audio_url": f"/files/{request_id}.wav"})

            if buffer.strip():
                request_id = handle_sentence(buffer.strip())
                yield _format_sse("sentence_audio", {"text": buffer.strip(), "audio_url": f"/files/{request_id}.wav"})

            yield _format_sse("reply_text", {"character_transcript": full_assistant_reply.strip()})

            if not sentence_files:
                yield _format_sse(
                    "error", {"code": "llm_unavailable", "message": "Failed to synthesize vocal replies."}
                )
                return

            combined = AudioSegment.empty()
            for path in sentence_files:
                combined += AudioSegment.from_wav(path)
            combined_request_id = uuid.uuid4().hex
            combined_wav_path = str(OUTPUT_DIR / f"{combined_request_id}.wav")
            combined.export(combined_wav_path, format="wav")

            video_url = None
            if character_name not in AUDIO_ONLY_CHARACTERS:
                try:
                    video_filename = f"{combined_request_id}.mp4"
                    video_output_path = str(OUTPUT_DIR / video_filename)
                    generate_talking_video(character_name, combined_wav_path, output_path=video_output_path)
                    video_url = f"/files/{video_filename}"
                except Exception as e:
                    log.error("Video generation failed: %s", e, exc_info=True)
                    yield _format_sse("error", {"code": "video_failed", "message": str(e)})

            yield _format_sse("video", {"video_url": video_url})
            yield _format_sse("done", {})

        except Exception as e:
            log.error("VR stream pipeline error: %s", e, exc_info=True)
            yield _format_sse("error", {"code": "pipeline_failed", "message": str(e)})

        finally:
            _gpu_slot.release()
            if os.path.exists(temp_mic_path):
                os.remove(temp_mic_path)
            for path in sentence_files:
                if os.path.exists(path):
                    os.remove(path)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
