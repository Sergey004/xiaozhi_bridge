# Xiaozhi Server — a voice backend for xiaozhi clients

[Русский](README_ru.md) | English

`xiaozhi_server.py` is an all-in-one server for the xiaozhi protocol:
**OTA + WebSocket bridge + OpenAI-compatible STT/TTS/LLM + Dashboard**.
It is tuned for the client running on a Yandex Station
([Sergey004/xiaozhi_yandex_station](https://github.com/Sergey004/xiaozhi_yandex_station)),
but works with any xiaozhi device, including ESP32.

## Features

- **WebSocket bridge** (`/ws`, Bearer token, per-MAC device whitelist):
  server-side VAD with automatic noise calibration, two-way audio
  streaming, per-device downlink (HD clients 48 kHz, ESP32 16 kHz),
  abort/cancel, ping/pong, per-device dialogue context and memory.
- **Voice: Chatterbox TTS → RVC.** Voice cloning from a reference file
  (`AUDIO_PROMPT_PATH`), batch `rvc_infer` (HuBERT/ContentVec, rmvpe,
  Faiss index), **automatic pitch alignment**: the median f0 of the
  reference vs the generated audio → an exact semitone shift, so the
  output always sits in the target voice's register (measured: 139 Hz
  against a 142 Hz target). Downlink is normalized to `DOWNLINK_SR`
  (48 kHz by default).
- **STT**: Whisper (`/v1/audio/transcriptions`, `/stt`), configurable language.
- **LLM**: NVIDIA NIM / Ollama (`/v1/chat/completions`), per-device memory.
- **OTA** (`/ota`, `/xiaozhi/ota`): activation, config, firmware.
- **Dashboard + API**: `/` — status and devices, `/health`, `/devices`,
  `/logs/stream`, `/tools`, `/memory/{device_id}`.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp env.example .env          # tokens, RVC model, voice reference, LLM
./start_server.sh            # OTA+API on :8000, WS bridge on :8765
```

Client (`xiaozhi_config.json`):

```json
{ "ws_url": "ws://SERVER:8765/ws",
  "ota_url": "http://SERVER:8000/xiaozhi/ota/",
  "ws_token": "BRIDGE_TOKEN" }
```

The device MAC goes into `ALLOWED_DEVICES`; HD downlink is enabled by the
`features.downlink_hd` flag in the client hello (ESP32 devices without the
flag receive 16 kHz).

## Configuration

Everything lives in `.env` (template: [env.example](env.example)): the TTS
engine (`turbo|multilingual|dramabox`), RVC model/index/reference paths,
`DOWNLINK_SR`, `TTS_STREAMING` (`off|sentence|chunk` — batch / per-sentence
/ streaming), VAD parameters, the LLM provider.

## Architecture

```
client (WS 8765) ──▶ server-side VAD ──▶ Whisper STT ──▶ LLM ──▶ Chatterbox
                                                                     │
client ◀── opus 48k/16k paced send ◀── AudioRateController ◀───── RVC
```

Conversion tests: `test_tts_rvc.py` (batch, HuBERT vs ContentVec),
`test_tts_stream.py` (A/B batch vs realtime).

## Status

A working voice loop with a Yandex Station Lite 2 (vqe_0 → STT → LLM → RVC
→ speaker). Known limitations: batch RVC adds per-phrase latency
(`TTS_STREAMING=sentence` softens it); the streaming path has a slightly
different timbre.

## License

See [LICENSE](LICENSE). RVC models and voice references belong to their
owners — only use voices you have the rights to.
