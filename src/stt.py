from faster_whisper import WhisperModel

from . import config

stt_model = WhisperModel(
    config.WHISPER_MODEL_SIZE,
    device=config.WHISPER_DEVICE,
    compute_type=config.WHISPER_COMPUTE_TYPE,
)


def transcribe(filepath):
    segments, info = stt_model.transcribe(filepath, beam_size=5)
    return " ".join(segment.text for segment in segments).strip()
