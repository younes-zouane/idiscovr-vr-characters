import sys
import tempfile

from . import config

sys.path.insert(0, config.WAV2LIP_DIR)
from lipsync_local import LocalLipSync

from .characters import CHARACTER_IMAGES

lip_sync = LocalLipSync(checkpoint_path=config.WAV2LIP_CHECKPOINT, device="cuda")


def generate_talking_video(character_name, audio_path, output_path=None):
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    image_path = CHARACTER_IMAGES[character_name]
    return lip_sync.generate(image_path, audio_path, output_path)
