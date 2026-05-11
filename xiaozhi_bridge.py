"""
xiaozhi_bridge.py — WebSocket сервер для XiaoZhi ESP32 устройств
Использует voice_api.py как бэкенд для STT/TTS, NVIDIA NIM для LLM.

Аутентификация:
  - Токен в URL: ws://host:8765?token=YOUR_TOKEN
  - Опциональный вайтлист MAC-адресов девайсов

Направь девайс на: ws://your-host:8765?token=YOUR_TOKEN
"""

import asyncio
import json
import logging
import os
import re
import secrets
from urllib.parse import parse_qs, urlparse

import httpx
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
BRIDGE_HOST   = os.getenv("BRIDGE_HOST",   "0.0.0.0")
BRIDGE_PORT   = int(os.getenv("BRIDGE_PORT", "8765"))
# LLM и промпт живут в voice_api.py → ai_service.py → ai_config.py
# Менять персонажа/промпт нужно там, а не здесь

# Токен доступа (обязателен).
# Если не задан в .env — генерируем рандомный и выводим в лог.
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "")
if not BRIDGE_TOKEN:
    BRIDGE_TOKEN = secrets.token_urlsafe(32)
    logger.warning(f"⚠️  BRIDGE_TOKEN not set! Using random token: {BRIDGE_TOKEN}")
    logger.warning("    Set BRIDGE_TOKEN in .env to make it permanent.")

# Вайтлист MAC-адресов (опционально).
# Формат в .env: ALLOWED_DEVICES=aa:bb:cc:dd:ee:ff,11:22:33:44:55:66
# Если пустой — пускаем любой девайс с правильным токеном.
_raw_devices = os.getenv("ALLOWED_DEVICES", "")
ALLOWED_DEVICES: set = {
    d.strip().lower() for d in _raw_devices.split(",") if d.strip()
}



# ─── Opus декодер ─────────────────────────────────────────────────────────────
try:
    import opuslib
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


# ─── Аутентификация ───────────────────────────────────────────────────────────

def _extract_token(ws) -> str:
    """Извлекает токен из URL: ws://host:port/path?token=xxx"""
    try:
        path = getattr(ws, "request", None)
        if path:
            path = path.path
        else:
            path = getattr(ws, "path", "/")
        parsed = urlparse(path)
        params = parse_qs(parsed.query)
        return params.get("token", [""])[0]
    except Exception:
        return ""


def _check_token(token: str) -> bool:
    """Безопасное сравнение токенов (защита от timing attack)."""
    return secrets.compare_digest(token, BRIDGE_TOKEN)


def _check_device(mac: str) -> bool:
    """Проверяет MAC девайса против вайтлиста. Если вайтлист пуст — всё OK."""
    if not ALLOWED_DEVICES:
        return True
    return mac.lower() in ALLOWED_DEVICES


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


# ─── LLM через voice_api → phoneguy_reply (с тегами для Chatterbox) ──────────
#
# Почему НЕ напрямую в NIM:
#   phoneguy_reply() в ai_service.py:
#     1. Применяет Phone Guy system prompt + RAG контекст
#     2. Чистит ответ через clean_response_text()
#     3. СОХРАНЯЕТ теги [sigh][chuckle][clear throat] и т.д.
#   Chatterbox TTS эти теги понимает как просодию — голос звучит живо.
#   Прямой NIM → теги теряются или генерируются неправильные.
#
async def llm_via_phoneguy(text: str) -> str:
    """
    Вызывает phoneguy_reply через voice_api /v1/chat/completions.
    Возвращает полный ответ с тегами [sigh][chuckle] и т.д.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{VOICE_API_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 512,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _split_by_sentences(text: str) -> list:
    """
    Разбивает текст на предложения, сохраняя теги вместе с ближайшим предложением.
    Пример: "Uh, hello? [clear throat] Nice to meet you." →
            ["Uh, hello?", "[clear throat] Nice to meet you."]
    """
    # Сначала разбиваем по знакам конца предложения
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    result = []
    pending_tags = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Если часть — только тег, прикрепим к следующему предложению
        if re.fullmatch(r"(\[[^\]]+\]\s*)+", part):
            pending_tags += part + " "
        else:
            result.append((pending_tags + part).strip())
            pending_tags = ""

    # Остаток тегов без предложения
    if pending_tags.strip():
        if result:
            result[-1] += " " + pending_tags.strip()
        else:
            result.append(pending_tags.strip())

    return result if result else [text]


# ─── Утилиты ──────────────────────────────────────────────────────────────────
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


def wav_to_opus(wav_bytes: bytes) -> bytes:
    """Конвертирует WAV в Opus @ 16kHz через ffmpeg."""
    import subprocess
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
         "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "pipe:1"],
        input=wav_bytes, capture_output=True, timeout=30,
    )
    return proc.stdout if proc.returncode == 0 else wav_bytes


# ─── Hello handshake ──────────────────────────────────────────────────────────
async def handle_hello(ws, data: dict) -> tuple:
    """
    Обрабатывает hello сообщение от девайса.
    Возвращает (ok: bool, device_id: str).
    """
    device_id    = data.get("device_id", data.get("mac", "unknown"))
    version      = data.get("version", 1)
    transport    = data.get("transport", "websocket")
    audio_params = data.get("audio_params", {})

    logger.info(
        f"👋 Hello: device_id={device_id} version={version} "
        f"transport={transport} audio={audio_params}"
    )

    # Проверить MAC вайтлист
    if not _check_device(device_id):
        logger.warning(f"❌ Device {device_id} not in whitelist")
        await ws.send(json.dumps({"type": "hello", "error": "device_not_allowed"}))
        return False, device_id

    # Согласовать аудио параметры с девайсом
    await ws.send(json.dumps({
        "type": "hello",
        "transport": "websocket",
        "audio_params": {
            "format": "opus",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration": 60,
        },
    }))
    logger.info(f"✅ Hello OK: {device_id}")
    return True, device_id


# ─── Основной обработчик соединения ──────────────────────────────────────────
async def handle_connection(ws):
    remote = ws.remote_address

    # 1. Проверка токена в URL
    token = _extract_token(ws)
    if not _check_token(token):
        logger.warning(f"❌ Unauthorized from {remote} (bad token)")
        await ws.close(1008, "Unauthorized: invalid token")
        return

    logger.info(f"📱 Connected: {remote}")

    audio_buffer = bytearray()
    device_id    = "unknown"
    hello_done   = False  # девайс должен прислать hello перед остальным

    try:
        async for message in ws:

            # ── Бинарный фрейм = аудио ─────────────────────────────────────
            if isinstance(message, bytes):
                if not hello_done:
                    continue  # игнорируем аудио до hello
                pcm = decode_audio_frame(message)
                audio_buffer.extend(pcm)
                continue

            # ── Текстовый фрейм = управляющее сообщение ───────────────────
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")
            state    = data.get("state")

            # Hello handshake — обязателен первым
            if msg_type == "hello":
                ok, device_id = await handle_hello(ws, data)
                if not ok:
                    await ws.close(1008, "Device not allowed")
                    return
                hello_done = True
                continue

            # Всё остальное — только после hello
            if not hello_done:
                logger.warning(f"⚠️ Ignoring '{msg_type}' before hello from {remote}")
                continue

            # Начало записи
            if msg_type == "listen" and state == "start":
                audio_buffer.clear()
                logger.info(f"🎤 [{device_id}] Recording started")

            # Конец записи → STT → LLM → TTS
            elif msg_type == "listen" and state == "stop":
                if not audio_buffer:
                    continue

                logger.info(f"🎤 [{device_id}] Audio: {len(audio_buffer)} bytes")

                # STT
                try:
                    text = await stt(bytes(audio_buffer))
                    audio_buffer.clear()
                    logger.info(f"📝 [{device_id}] STT: {text}")
                except Exception as e:
                    logger.error(f"STT error: {e}")
                    audio_buffer.clear()
                    continue

                if not text.strip():
                    continue

                await ws.send(json.dumps({"type": "stt", "text": text}))

                # LLM → phoneguy_reply (теги сохраняются) → TTS по предложениям
                await ws.send(json.dumps({"type": "tts", "state": "start"}))

                try:
                    full_reply = await llm_via_phoneguy(text)
                    logger.info(f"🤖 [{device_id}] LLM: {full_reply}")
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    await ws.send(json.dumps({"type": "tts", "state": "stop"}))
                    continue

                # Разбиваем на предложения — теги остаются при своих предложениях
                # "Uh, hello? [clear throat] Nice to meet you." →
                # ["Uh, hello?", "[clear throat] Nice to meet you."]
                sentences = _split_by_sentences(full_reply)

                for sentence in sentences:
                    if not sentence:
                        continue

                    # На экран — только читаемый текст (без тегов)
                    display_text = re.sub(r"\[[^\]]+\]", "", sentence).strip()
                    if display_text:
                        await ws.send(json.dumps({
                            "type": "tts",
                            "state": "sentence_start",
                            "text": display_text,
                        }))

                    # В Chatterbox — с тегами (просодия)
                    try:
                        wav = await tts(sentence)
                        await ws.send(wav_to_opus(wav))
                    except Exception as e:
                        logger.error(f"TTS error: {e}")

                await ws.send(json.dumps({"type": "tts", "state": "stop"}))

            # Пинг
            elif msg_type == "ping":
                await ws.send(json.dumps({"type": "pong"}))

            # Аборт (девайс прервал воспроизведение / barge-in)
            elif msg_type == "abort":
                logger.info(f"⏹️ [{device_id}] Abort")
                audio_buffer.clear()

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"📱 Disconnected: {remote} (code={e.code})")
    except Exception as e:
        logger.error(f"Connection error [{remote}]: {e}", exc_info=True)


# ─── Запуск ──────────────────────────────────────────────────────────────────
async def main():
    logger.info(f"🌐 XiaoZhi Bridge: ws://{BRIDGE_HOST}:{BRIDGE_PORT}")
    logger.info(f"🔑 Token: {BRIDGE_TOKEN[:8]}...")
    logger.info(f"📋 Whitelist: {ALLOWED_DEVICES or 'disabled (all devices allowed)'}")
    logger.info(f"🔗 Voice API: {VOICE_API_URL}")
    logger.info(f"🤖 LLM: via voice_api → ai_service (Phone Guy)")

    async with websockets.serve(handle_connection, BRIDGE_HOST, BRIDGE_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
