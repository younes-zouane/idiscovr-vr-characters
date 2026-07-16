# Third-Party Notices

This project uses two third-party components whose licenses restrict usage
to non-commercial, personal, or research purposes. This is a hard
constraint on the whole project, not just those components — see
"What this means for iDISCOVR" below.

## Wav2Lip (`wav2lip-onnx-256/`)

- **Source:** https://github.com/Rudrabha/Wav2Lip
- **License:** Personal / research / non-commercial use only.
- **Why:** Per the original authors, the Wav2Lip models are trained on the
  LRS2 dataset, which itself prohibits commercial use. Any use of the
  models — including this project's ONNX-converted version — inherits
  that restriction. Quoting the original repository directly:

  > This repository can only be used for personal/research/non-commercial
  > purposes. However, for commercial requests, please contact us directly
  > at radrabha.m@research.iiit.ac.in or prajwal.k@research.iiit.ac.in.

- **What this project uses:** the ONNX-converted `wav2lip_256.onnx`
  checkpoint (not committed to this repo — see `.gitignore`; obtained and
  placed manually per the README setup instructions) and the inference code
  adapted into `wav2lip-onnx-256/lipsync_local.py` and related files.

## insightface (`wav2lip-onnx-256/insightface_func/`)

- **Source:** https://github.com/deepinsight/insightface
- **License:** Split license — read carefully, the code and the models are
  licensed differently:
  - **Code:** MIT License. No restriction on commercial or academic use.
  - **Pre-trained models** (including the `antelope`/SCRFD face detection
    model this project uses, `scrfd_2.5g_bnkps.onnx`): non-commercial
    research purposes only. Quoting the official repository directly:

    > The code of InsightFace is released under the MIT License. There is
    > no limitation for both academic and commercial usage. The training
    > data containing the annotation (and the models trained with these
    > data) are available for non-commercial research purposes only.

- **What this project uses:** the face-detection/cropping code in
  `wav2lip-onnx-256/insightface_func/`, and the pre-trained
  `scrfd_2.5g_bnkps.onnx` detection model (committed to this repo, small
  file) used to locate and crop character faces before lip-sync inference.

## What this means for iDISCOVR

Because this project's lip-sync pipeline depends on both a Wav2Lip model
(research-only by license) and an insightface pre-trained model
(research-only by license), **the pipeline as a whole cannot be used
commercially** without separately licensing both components from their
respective authors. This project is built and used for an academic
internship / research context, which is consistent with both licenses —
but this restriction should be kept in mind before any commercial or
production deployment, and should not be removed or worked around without
first securing appropriate licensing from Rudrabha/Wav2Lip and
deepinsight/insightface.

All other dependencies of this project (Gradio, faster-whisper, Kokoro,
Ollama/llama3.1, onnxruntime, etc.) are used under their own respective
open-source licenses, which permit broader use — this notice specifically
covers the two components above because their restrictions are the ones
that actually constrain what this project as a whole can be used for.
