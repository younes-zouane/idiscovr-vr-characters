import sys
from unittest.mock import MagicMock

# ── Fake out heavy ML libraries BEFORE any src.* module can import them ──
# This runs once, when pytest loads conftest.py, ahead of any test collection.

fake_faster_whisper = MagicMock()
fake_faster_whisper.WhisperModel = MagicMock(return_value=MagicMock())
sys.modules.setdefault("faster_whisper", fake_faster_whisper)

fake_kokoro = MagicMock()
fake_kokoro.KPipeline = MagicMock(return_value=MagicMock())
sys.modules.setdefault("kokoro", fake_kokoro)

fake_lipsync_local = MagicMock()
fake_lipsync_local.LocalLipSync = MagicMock(return_value=MagicMock())
sys.modules.setdefault("lipsync_local", fake_lipsync_local)