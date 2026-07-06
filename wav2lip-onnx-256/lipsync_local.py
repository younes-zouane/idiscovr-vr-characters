import os
import sys
import subprocess
import numpy as np
import cv2
import sysconfig

import onnxruntime

_site_packages = sysconfig.get_paths()["purelib"]
_cuda_dir = os.path.join(_site_packages, "nvidia", "cu13", "bin", "x86_64")
_cudnn_dir = os.path.join(_site_packages, "nvidia", "cudnn", "bin")

if os.path.isdir(_cuda_dir):
    onnxruntime.preload_dlls(cuda=True, cudnn=False, directory=_cuda_dir)
# cudnn preload removed — let torch's own bundled cuDNN be the only one in the process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio  # the repo's own audio.py
from insightface_func.face_detect_crop_single import Face_detect_crop

MEL_STEP_SIZE = 16
IMG_SIZE = 256  # matches wav2lip_256.onnx


class LocalLipSync:
    def __init__(self, checkpoint_path, device="cuda", pads=(0, 10, 0, 0)):
        self.pads = pads
        base_dir = os.path.dirname(os.path.abspath(__file__))
        from gfpgan import GFPGANer

        self.face_enhancer = GFPGANer(
            model_path='GFPGANv1.4.pth',  # auto-downloads on first run if not present
            upscale=1,  # we're already resizing to the target crop size; don't double-upscale
            arch='clean',
            channel_multiplier=2,
            device=device
        )
        # 1) Load the ONNX model ONCE
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(checkpoint_path, sess_options=session_options, providers=providers)

        # 2) Load the face detector ONCE
        self.detector = Face_detect_crop(name="antelope", root=os.path.join(base_dir, "insightface_func", "models"))
        self.detector.prepare(ctx_id=0, det_thresh=0.3, det_size=(320, 320), mode="none")

        # 3) Per-character face crop cache — detection runs once per image ever, not per reply
        self._face_cache = {}

    def _get_face(self, image_path):
        if image_path in self._face_cache:
            return self._face_cache[image_path]

        image = cv2.imread(image_path)
        bbox = self.detector.getBox(image)
        if bbox is None:
            raise ValueError(f"No face detected in {image_path}")

        pady1, pady2, padx1, padx2 = self.pads
        y1 = max(0, bbox[1] - pady1)
        y2 = min(image.shape[0], bbox[3] + pady2)
        x1 = max(0, bbox[0] - padx1)
        x2 = min(image.shape[1], bbox[2] + padx2)

        # --- force a square crop, centered on the current box, before resizing ---
        box_h = y2 - y1
        box_w = x2 - x1
        side = max(box_h, box_w)

        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2

        y1 = max(0, cy - side // 2)
        y2 = min(image.shape[0], y1 + side)
        x1 = max(0, cx - side // 2)
        x2 = min(image.shape[1], x1 + side)
        # re-clamp in case we hit an image edge
        y1 = max(0, y2 - side)
        x1 = max(0, x2 - side)

        print("crop box size:", (y2 - y1), "x", (x2 - x1))
        face = cv2.resize(image[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
        cv2.imwrite("debug_face_crop.jpg", face)

        img_masked = face.copy()
        img_masked[IMG_SIZE // 2:, :] = 0
        combined = np.concatenate((img_masked, face), axis=2).astype(np.float32) / 255.0
        img_batch = combined[np.newaxis].transpose(0, 3, 1, 2)

        result = {"full_frame": image, "coords": (y1, y2, x1, x2), "img_batch": img_batch}
        self._face_cache[image_path] = result
        return result

    def generate(self, image_path, audio_path, output_path="reply_video.mp4", fps=25.0):
        face_data = self._get_face(image_path)
        full_frame = face_data["full_frame"]
        y1, y2, x1, x2 = face_data["coords"]
        img_batch = face_data["img_batch"]

        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel contains NaN — check the input wav")

        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + MEL_STEP_SIZE > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + MEL_STEP_SIZE])
            i += 1

        frame_h, frame_w = full_frame.shape[:2]
        temp_avi = output_path.replace(".mp4", "_temp.avi")
        out = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*"DIVX"), fps, (frame_w, frame_h))

        # --- build a feathered blend mask ONCE (same size as the crop) ---
        patch_h, patch_w = y2 - y1, x2 - x1
        mask = np.ones((patch_h, patch_w), dtype=np.float32)
        feather = max(4, int(min(patch_h, patch_w) * 0.08))  # ~8% of patch size
        mask = cv2.rectangle(mask, (0, 0), (patch_w - 1, patch_h - 1), 0, feather * 2)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
        mask_3ch = mask[:, :, np.newaxis]

        for m in mel_chunks:
            mel_batch = m.reshape(1, m.shape[0], m.shape[1], 1).transpose(0, 3, 1, 2).astype(np.float32)
            pred = self.session.run(None, {"mel_spectrogram": mel_batch, "video_frames": img_batch})[0][0]
            pred = (pred.transpose(1, 2, 0) * 255).astype(np.uint8)
            pred = cv2.resize(pred, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC)

            frame = full_frame.copy()
            original_patch = frame[y1:y2, x1:x2].astype(np.float32)
            pred_f = pred.astype(np.float32)

            blended = pred_f * mask_3ch + original_patch * (1 - mask_3ch)
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)

            out.write(frame)

        out.release()

        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-i", temp_avi, "-strict", "-2", "-q:v", "1", output_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.remove(temp_avi)
        return output_path