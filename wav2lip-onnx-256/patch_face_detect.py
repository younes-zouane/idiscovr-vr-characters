"""
One-off patch script to fix a numpy 2.x incompatibility in
insightface_func/face_detect_crop_single.py (wav2lip-onnx-256 repo).

Run this from inside your wav2lip-onnx-256 folder:
    python patch_face_detect.py
"""

import pathlib

target = pathlib.Path("insightface_func/face_detect_crop_single.py")

if not target.exists():
    raise SystemExit(f"Could not find {target}. Run this script from inside the wav2lip-onnx-256 folder.")

text = target.read_text(encoding="utf-8")

old = """        x1 = int(bboxes[0, 0:1])
        y1 = int(bboxes[0, 1:2])
        x2 = int(bboxes[0, 2:3])
        y2 = int(bboxes[0, 3:4])"""

new = """        x1 = int(bboxes[0, 0])
        y1 = int(bboxes[0, 1])
        x2 = int(bboxes[0, 2])
        y2 = int(bboxes[0, 3])"""

if old not in text:
    raise SystemExit("Could not find the expected block to patch. The file may already be patched, or differs from what was expected.")

# backup original
backup = target.with_suffix(target.suffix + ".bak")
backup.write_text(text, encoding="utf-8")

text = text.replace(old, new)
target.write_text(text, encoding="utf-8")

print(f"Patched {target}")
print(f"Original backed up to {backup}")