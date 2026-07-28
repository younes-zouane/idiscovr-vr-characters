"""
VR Backend Integration Service

Headless REST endpoint for VR clients (Unity/Unreal), replacing the Gradio UI
for that use case. Reuses the existing pipeline modules (stt, llm, tts, lipsync)
exactly as the main app.py does.

Two things fixed vs. the first draft:
1. Per-session conversation history, keyed by session_id (not shared globally
   per character) — required for more than one concurrent VR user.
2. Audio/video returned as URLs to static files, not embedded base64 in the
   JSON body — base64-in-JSON is a poor fit for a VR client parsing this on
   a standalone headset (Unity's JsonUtility on multi-MB strings is slow and
   memory-heavy, and base64 inflates binary size ~33% on top of that).
"""

import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from src.characters import AUDIO_ONLY_CHARACTERS, CHARACTERS
from src.lipsync import generate_talking_video
from src.llm import init_conversation_histories, stream_character_reply
from src.sentence_splitter import split_into_sentences
from src.stt import transcribe
from src.tts import speak

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

app = FastAPI(title="VR Character Animation Headless Service")

# Directory for generated audio/video files served back to VR clients as URLs.
OUTPUT_DIR = Path(tempfile.gettempdir()) / "idiscovr_vr_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

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


def get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    """Return (session_id, histories_dict). Creates a new session if the
    given session_id is missing or unknown."""
    with _sessions_lock:
        if session_id and session_id in _sessions:
            return session_id, _sessions[session_id]
        new_id = session_id or str(uuid.uuid4())
        _sessions[new_id] = init_conversation_histories()
        return new_id, _sessions[new_id]


@app.post("/v1/vr-chat")
async def vr_chat_endpoint(
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
        raise HTTPException(status_code=400, detail=f"Character '{character_name}' not found.")

    effective_session_id, histories = get_or_create_session(session_id)
    history = histories[character_name]

    temp_mic_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    request_id = uuid.uuid4().hex
    combined_wav_path = str(OUTPUT_DIR / f"{request_id}.wav")
    sentence_audio_paths = []

    try:
        with open(temp_mic_path, "wb") as f:
            f.write(await audio_file.read())

        user_text = transcribe(temp_mic_path)
        if not user_text:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not understand incoming VR mic data."},
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
            raise HTTPException(status_code=500, detail="Failed to synthesize vocal replies.")

        combined = AudioSegment.empty()
        for path in sentence_audio_paths:
            combined += AudioSegment.from_wav(path)
        combined.export(combined_wav_path, format="wav")

        for path in sentence_audio_paths:
            if os.path.exists(path):
                os.remove(path)

        video_url = None
        if character_name not in AUDIO_ONLY_CHARACTERS:
            video_filename = f"{request_id}.mp4"
            video_output_path = str(OUTPUT_DIR / video_filename)
            generate_talking_video(character_name, combined_wav_path, output_path=video_output_path)
            video_url = f"/files/{video_filename}"

        return JSONResponse(
            content={
                "status": "success",
                "session_id": effective_session_id,
                "character": character_name,
                "user_transcript": user_text,
                "character_transcript": full_assistant_reply.strip(),
                "voice_audio_url": f"/files/{request_id}.wav",
                "talking_video_url": video_url,
            }
        )

    except Exception as e:
        log.error("VR pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_mic_path):
            os.remove(temp_mic_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
