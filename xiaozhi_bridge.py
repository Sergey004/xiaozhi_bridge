"""
xiaozhi_bridge.py — WebSocket сервер для XiaoZhi ESP32 устройств
Использует voice_api.py как бэкенд для STT/TTS, NVIDIA NIM для LLM.

Направь девайс на: ws://your-host:8765
"""

import asyncio
import json
import logging
import os
import re

import httpx
import opuslib
import numpy as np
import rich.logging
import websockets

# ─── Логгер ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("xiaozhi_bridge")

# ─── Конфиг ──────────────────────────────────────────────────────────────────
VOICE_API_URL = os.getenv("VOICE_API_URL", "http://localhost:8000")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash")
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8765"))

# ─── Opus декодер ─────────────────────────────────────────────────────────────
try:
    _opus_decoder = opuslib.Decoder(16000, 1)
    OPUS_AVAILABLE = True
except Exception:
    OPUS_AVAILABLE = False
    logger.warning("⚠️ opuslib not available, assuming raw PCM input")


def decode_audio_frame(data: bytes) -> bytes:
    """Декодирует Opus фрейм → PCM16 @ 16kHz, или возвращает как есть (PCM)."""
    if OPUS_AVAILABLE:
        try:
            return _opus_decoder.decode(data, 960)
        except Exception:
            pass
    return data


# ─── STT через voice_api ──────────────────────────────────────────────────────
async def stt(pcm_bytes: bytes) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{VOICE_API_URL}/v1/audio/transcriptions",
            files={"file": ("audio.wav", _pcm_to_wav(pcm_bytes, 16000), "audio/wav")},
            data={"language": "ru"},
        )
        r.raise_for_status()
        return r.json().get("text", "")


# ─── TTS через voice_api ──────────────────────────────────────────────────────
async def tts(text: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{VOICE_API_URL}/tts",
            json={"text": text},
        )
        r.raise_for_status()
        return r.content  # WAV bytes


# ─── LLM через NVIDIA NIM напрямую (стриминг) ────────────────────────────────
async def llm_stream(text: str, history: list):
    """Стриминговый LLM — возвращает предложения по мере генерации."""
    history.append({"role": "user", "content": text})

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": NVIDIA_MODEL,
                "messages": history,
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": True,
            },
        ) as resp:
            buf = ""
            full_reply = ""

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    token = data["choices"][0]["delta"].get("content", "")
                    buf += token
                    full_reply += token

                    # Отдаём по предложениям
                    if re.search(r"[.!?…]\s*$", buf) or buf.endswith("\n"):
                        yield buf.strip()
                        buf = ""
                except Exception:
                    continue

            # Остаток
            if buf.strip():
                yield buf.strip()

            history.append({"role": "assistant", "content": full_reply})


# ─── Утилита: PCM → WAV ───────────────────────────────────────────────────────
def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


# ─── WAV → Opus (для отправки на XiaoZhi) ────────────────────────────────────
def wav_to_opus(wav_bytes: bytes) -> bytes:
    """Конвертирует WAV в Opus @ 16kHz через ffmpeg."""
    import subprocess
    proc = subprocess.run(
        [
            "ffmpeg", "-i", "pipe:0",
            "-ar", "16000", "-ac", "1",
            "-c:a", "libopus", "-b:a", "32k",
            "-f", "ogg", "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode == 0:
        return proc.stdout
    # Fallback: отдать WAV как есть
    return wav_bytes


# ─── Основной обработчик соединения ──────────────────────────────────────────
async def handle_connection(ws):
    logger.info(f"📱 XiaoZhi connected: {ws.remote_address}")
    audio_buffer = bytearray()
    chat_history = []

    try:
        async for message in ws:
            # ── Бинарный фрейм = аудиоданные ──────────────────────────────
            if isinstance(message, bytes):
                pcm = decode_audio_frame(message)
                audio_buffer.extend(pcm)

            # ── Текстовый фрейм = управляющее сообщение ───────────────────
            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                state = data.get("state")

                # Начало записи
                if msg_type == "listen" and state == "start":
                    audio_buffer.clear()
                    logger.info("🎤 Recording started")

                # Конец записи → pipeline
                elif msg_type == "listen" and state == "stop":
                    if not audio_buffer:
                        continue

                    logger.info(f"🎤 Audio captured: {len(audio_buffer)} bytes")

                    # 1. STT
                    try:
                        text = await stt(bytes(audio_buffer))
                        audio_buffer.clear()
                        logger.info(f"📝 STT: {text}")
                    except Exception as e:
                        logger.error(f"STT error: {e}")
                        audio_buffer.clear()
                        continue

                    if not text.strip():
                        continue

                    # Показать что услышали
                    await ws.send(json.dumps({
                        "type": "stt", "text": text
                    }))

                    # 2. LLM (стриминг) + TTS по предложениям
                    await ws.send(json.dumps({
                        "type": "tts", "state": "start"
                    }))

                    async for sentence in llm_stream(text, chat_history):
                        if not sentence:
                            continue

                        logger.info(f"🤖 LLM sentence: {sentence}")

                        # Показать текст на экране
                        await ws.send(json.dumps({
                            "type": "tts",
                            "state": "sentence_start",
                            "text": sentence,
                        }))

                        # TTS для предложения
                        try:
                            wav = await tts(sentence)
                            audio_out = wav_to_opus(wav)
                            await ws.send(audio_out)
                        except Exception as e:
                            logger.error(f"TTS error: {e}")

                    await ws.send(json.dumps({
                        "type": "tts", "state": "stop"
                    }))

                # Пинг / другие сообщения
                elif msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"📱 XiaoZhi disconnected: {ws.remote_address}")
    except Exception as e:
        logger.error(f"Connection error: {e}", exc_info=True)


# ─── Запуск ──────────────────────────────────────────────────────────────────
async def main():
    logger.info(f"🌐 XiaoZhi Bridge listening on ws://{BRIDGE_HOST}:{BRIDGE_PORT}")
    logger.info(f"🔗 Voice API: {VOICE_API_URL}")
    logger.info(f"🤖 LLM: {NVIDIA_MODEL}")

    async with websockets.serve(handle_connection, BRIDGE_HOST, BRIDGE_PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
