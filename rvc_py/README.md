# rvc_py

Minimalist RVC (Retrieval-based Voice Conversion) implementation for Python 3.11, adapted from <https://github.com/daswer123/rvc-python>.

- Compatible with the main VibeVoice project.
- No third-party UI required.
- For integration, use the rvc_infer function from main.py.

## Inference Features (adapted from RVC WebUI)

- **Sample rate support:** 32kHz, 40kHz, 48kHz (determined by model, sr parameter).
- **Inference parameters:** pitch shift, index rate (retrieval blending), f0 method (rmvpe/torchcrepe/parselmouth/pm/dio), hop size, block size, auto predict f0, fp16/half-precision.
- **Vector search:** Faiss support for retrieval-based voice conversion (accelerates and improves quality).
- **Flexible audio processing:** automatic sample rate, hop size, block size detection.
- **Architecture:** modular — separate modules for preprocessing, feature extraction, inference, and search.
- **Dependencies:** torch, faiss, librosa, soundfile, numpy, torchcrepe, parselmouth, onnxruntime, pyworld, scipy.

## Installing Dependencies

```shell
pip install torch numpy librosa soundfile scipy torchcrepe faiss-cpu parselmouth onnxruntime pyworld
```

## Usage

Import rvc_infer from rvc_py/rvc_infer.py and call it as a post-process.

---

Original repository: <https://github.com/daswer123/rvc-python>
WebUI: <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>
Adaptation: Python 3.11, no UI, inference only.
