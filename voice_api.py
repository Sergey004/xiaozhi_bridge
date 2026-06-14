"""
voice_api.py — OpenAI-совместимый микросервис для STT / TTS / LLM
Оборачивает stt_adapter.py, tts_adapter.py, ai_service.py без изменений.

Эндпоинты:
  POST /v1/audio/transcriptions   — STT (Whisper)
  POST /v1/audio/speech           — TTS (Chatterbox + RVC)
  POST /v1/chat/completions       — LLM (NVIDIA NIM / Ollama)
  GET  /health                    — статус всех компонентов

Запуск:
  uvicorn voice_api:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import io
import logging
import os
import struct
import time
import wave
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import rich.logging
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from scipy import signal as scipy_signal
import dotenv
dotenv.load_dotenv()

# ─── Логгер ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("voice_api")

# ─── Конфигурация (через .env или env vars) ─────────────────────────────────
STT_CONFIG = {
    "stt": {
        "model": os.getenv("WHISPER_MODEL", "turbo"),
        "device": os.getenv("WHISPER_DEVICE", "cuda"),
        "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        "language": os.getenv("WHISPER_LANGUAGE", "ru"),
        "beam_size": int(os.getenv("WHISPER_BEAM_SIZE", "5")),
        "target_sample_rate": 16000,
        "energy_threshold": 300,
        "min_chunk_ms": 1000,
        "silence_end_ms": 500,
    }
}

TTS_CONFIG = {
    "tts": {
        "engine": os.getenv("TTS_ENGINE", "turbo"),           # turbo | multilingual | base
        "device": os.getenv("TTS_DEVICE", "cuda"),
        "audio_prompt_path": os.getenv("TTS_VOICE_SAMPLE"),   # None = дефолтный голос
        "rvc_enabled": os.getenv("RVC_ENABLED", "false").lower() == "true",
        "rvc_model_path": os.getenv("RVC_MODEL_PATH"),
        "rvc_index_path": os.getenv("RVC_INDEX_PATH"),
        "rvc_f0_method": os.getenv("RVC_F0_METHOD", "rmvpe"),
        "rvc_pitch_shift": int(os.getenv("RVC_PITCH_SHIFT", "0")),
        "rvc_index_rate": float(os.getenv("RVC_INDEX_RATE", "0.5")),
    }
}

# ─── LLM конфигурация (прямой httpx клиент, без langchain) ───────────────
# Использует ТЕ ЖЕ переменные из .env что и ai_service.py:
#   LLM_PROVIDER=nvidia|ollama
#   NVIDIA_API_KEY=nvapi-...
#   NVIDIA_MODEL=meta/llama3-70b-instruct
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=llama3.1
#
# Если задан LLM_BASE_URL напрямую — используется он (override для других API).

LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "nvidia").lower()
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "meta/llama3-70b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1")

# Override: если задан LLM_BASE_URL напрямую — используем его
LLM_BASE_URL_OVERRIDE = os.getenv("LLM_BASE_URL", "")
if LLM_BASE_URL_OVERRIDE:
    _direct_base_url = LLM_BASE_URL_OVERRIDE
    _direct_api_key  = os.getenv("LLM_API_KEY", "")
    _direct_model    = os.getenv("LLM_MODEL", "")
    logger.info(f"🤖 LLM override: {_direct_base_url} model={_direct_model}")
elif LLM_PROVIDER == "ollama":
    _direct_base_url = OLLAMA_BASE_URL.rstrip("/") + "/v1"
    _direct_api_key  = "ollama"
    _direct_model    = OLLAMA_MODEL
    logger.info(f"🤖 LLM via Ollama: {_direct_base_url} model={OLLAMA_MODEL}")
else:
    # nvidia (default)
    _direct_base_url = "https://integrate.api.nvidia.com/v1"
    _direct_api_key  = NVIDIA_API_KEY
    _direct_model    = NVIDIA_MODEL
    logger.info(f"🤖 LLM via NVIDIA: model={NVIDIA_MODEL}")

LLM_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TIMEOUT    = int(os.getenv("LLM_TIMEOUT", "30"))

# LLM ready?
_llm_ready = bool(_direct_api_key and _direct_model)
if _llm_ready:
    logger.info(f"✅ LLM ready: {_direct_base_url} model={_direct_model}")
else:
    logger.warning("⚠️  LLM NOT configured!")
    if LLM_PROVIDER == "nvidia":
        logger.warning("   Set NVIDIA_API_KEY in .env")
    else:
        logger.warning("   Make sure Ollama is running: ollama list")

# История диалога для LLM (простая in-memory)
_llm_history: list[dict] = []
LLM_HISTORY_MAX = 20  # макс. пар в истории


def _llm_history_reset():
    """Сбрасывает историю диалога."""
    _llm_history.clear()


def _llm_history_add(role: str, content: str):
    """Добавляет сообщение в историю."""
    _llm_history.append({"role": role, "content": content})
    # Обрезаем старые сообщения
    while len(_llm_history) > LLM_HISTORY_MAX:
        _llm_history.pop(0)


async def _call_llm_direct(user_message: str) -> str | None:
    """
    Прямой вызов LLM через httpx (OpenAI-compatible /chat/completions).
    НЕ требует langchain, openai SDK, или чего-либо кроме httpx.
    Возвращает текст ответа или None при ошибке.
    """
    if not _llm_ready:
        return None

    # Строим сообщения
    messages = []
    if LLM_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": LLM_SYSTEM_PROMPT})
    messages.extend(_llm_history)
    messages.append({"role": "user", "content": user_message})

    url = f"{_direct_base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if _direct_api_key:
        headers["Authorization"] = f"Bearer {_direct_api_key}"

    payload = {
        "model": _direct_model,
        "messages": messages,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            reply = data["choices"][0]["message"]["content"].strip()

            # Сохраняем в историю
            _llm_history_add("user", user_message)
            _llm_history_add("assistant", reply)

            return reply
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except httpx.TimeoutException:
        logger.error(f"LLM timeout after {LLM_TIMEOUT}s")
        return None
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


# ─── Глобальные экземпляры адаптеров ────────────────────────────────────────
stt_adapter = None
tts_adapter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте, очистка при остановке."""
    global stt_adapter, tts_adapter

    logger.info("🚀 Starting Voice API...")

    # STT
    try:
        from stt_adapter import STTAdapter
        stt_adapter = STTAdapter(STT_CONFIG, logger)
        logger.info("✅ STT (Whisper) ready")
    except Exception as e:
        logger.error(f"❌ STT init failed: {e}")

    # TTS
    try:
        from tts_adapter import create_tts_adapter
        tts_adapter = create_tts_adapter(TTS_CONFIG, logger)
        ok = await tts_adapter.check_health()
        engine_name = TTS_CONFIG.get("tts", {}).get("engine", "turbo")
        if ok:
            logger.info(f"✅ TTS ({engine_name}) ready")
        else:
            logger.warning(f"⚠️ TTS ({engine_name}) health check failed, continuing anyway")
    except Exception as e:
        logger.error(f"❌ TTS init failed: {e}")

    logger.info("🎙️ Voice API online at :8000")
    yield

    # Cleanup
    if stt_adapter:
        stt_adapter.close()
    logger.info("👋 Voice API shutdown")


app = FastAPI(
    title="Voice API",
    description="OpenAI-compatible STT/TTS microservice (Whisper + Chatterbox + RVC)",
    version="1.0.0",
    lifespan=lifespan,
)

# Хранилище кодов активации (в памяти, для домашнего сервера достаточно)
_activation_codes: dict = {}


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def _load_audio_to_numpy(file_bytes: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """
    Загружает аудио из байтов в numpy float32.
    Поддерживает WAV, PCM (raw), OGG через ffmpeg-fallback.
    Возвращает (audio_np, sample_rate).
    """
    # WAV — читаем напрямую
    try:
        buf = io.BytesIO(file_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            # Стерео → моно
            if wf.getnchannels() == 2:
                arr = arr.reshape(-1, 2).mean(axis=1)
            return arr, sr
    except Exception:
        pass

    # Opus/OGG/MP3 — через ffmpeg
    try:
        import subprocess
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
            input=file_bytes,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            arr = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return arr, 16000
    except Exception:
        pass

    # Fallback: сырой PCM int16 @ 16000Hz (а не 8000Hz как было раньше)
    # XiaoZhi шлёт именно 16kHz PCM
    arr = np.frombuffer(file_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return arr, 16000


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Конвертирует сырой PCM int16 в WAV-байты."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Ресемплинг через scipy."""
    if src_sr == dst_sr:
        return audio
    n_out = int(round(len(audio) * dst_sr / src_sr))
    return scipy_signal.resample(audio, n_out).astype(np.float32)


def _get_tts_attr(adapter, attr: str, default=None):
    """Безопасно читает атрибут TTS адаптера (может не существовать)."""
    try:
        return getattr(adapter, attr, default)
    except Exception:
        return default


def _set_tts_attr(adapter, attr: str, value) -> bool:
    """Безопасно устанавливает атрибут TTS адаптера."""
    try:
        setattr(adapter, attr, value)
        return True
    except Exception:
        return False


# ─── Эндпоинты STT ───────────────────────────────────────────────────────────

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    """
    OpenAI-совместимый STT.
    Принимает: WAV, PCM, Opus, OGG, MP3
    Возвращает: {"text": "..."}
    """
    if stt_adapter is None:
        raise HTTPException(503, "STT adapter not initialized")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Empty audio file")

    # Загрузить аудио
    audio_np, src_sr = _load_audio_to_numpy(file_bytes, file.filename or "")

    # Ресемплинг до 16kHz (Whisper)
    if src_sr != 16000:
        audio_np = _resample(audio_np, src_sr, 16000)

    # Транскрипция через STT адаптер (напрямую, минуя VAD-очередь)
    lang = language or _get_tts_attr(stt_adapter, "language", "ru")

    # Временно перекрываем language для этого запроса
    orig_lang = _get_tts_attr(stt_adapter, "language", lang)
    if lang:
        _set_tts_attr(stt_adapter, "language", lang)

    text = await stt_adapter.transcribe_numpy(audio_np, 16000)

    _set_tts_attr(stt_adapter, "language", orig_lang)  # восстанавливаем

    if response_format == "text":
        return text

    return {"text": text}


# Алиас без /v1/ префикса
@app.post("/stt")
async def stt_simple(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    return await transcriptions(file=file, language=language)


# ─── Эндпоинты TTS ───────────────────────────────────────────────────────────

@app.post("/v1/audio/speech")
async def speech(
    model: str = Form("tts-1"),
    input: str = Form(...),
    voice: str = Form("default"),
    response_format: str = Form("wav"),
    speed: float = Form(1.0),
):
    """
    OpenAI-совместимый TTS.
    Принимает: text
    Возвращает: WAV аудио
    """
    if tts_adapter is None:
        raise HTTPException(503, "TTS adapter not initialized")
    if not input.strip():
        raise HTTPException(400, "Empty text")

    # Голосовой семпл (voice как путь к файлу или preset)
    orig_prompt = _get_tts_attr(tts_adapter, "audio_prompt_path")
    if voice != "default" and os.path.exists(voice):
        _set_tts_attr(tts_adapter, "audio_prompt_path", voice)

    pcm_bytes, sample_rate = await tts_adapter.synthesize(input)

    _set_tts_attr(tts_adapter, "audio_prompt_path", orig_prompt)  # восстанавливаем

    if not pcm_bytes:
        raise HTTPException(500, "TTS synthesis failed")

    # PCM → WAV
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sample_rate)},
    )


# Алиас для JSON-body запроса (удобнее из кода)
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    response_format: Optional[str] = "wav"


@app.post("/tts")
async def tts_simple(req: TTSRequest):
    """
    Упрощённый TTS эндпоинт (JSON body).
    POST /tts
    {"text": "Привет!", "voice": "default"}
    """
    if tts_adapter is None:
        raise HTTPException(503, "TTS adapter not initialized")

    orig_prompt = _get_tts_attr(tts_adapter, "audio_prompt_path")
    if req.voice and req.voice != "default" and os.path.exists(req.voice):
        _set_tts_attr(tts_adapter, "audio_prompt_path", req.voice)

    t0 = time.time()
    pcm_bytes, sample_rate = await tts_adapter.synthesize(req.text)
    _set_tts_attr(tts_adapter, "audio_prompt_path", orig_prompt)

    if not pcm_bytes:
        raise HTTPException(500, "TTS synthesis failed")

    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)
    elapsed = time.time() - t0
    duration = len(pcm_bytes) / (2 * sample_rate) if sample_rate else 0
    logger.info(
        f"🔊 TTS: '{req.text[:40]}...' → {len(wav_bytes)} bytes, "
        f"{duration:.1f}s audio, {elapsed:.2f}s gen"
    )

    # Response вместо StreamingResponse — httpx надёжно читает весь ответ
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sample_rate)},
    )


# ─── TTS Стриминг (по предложениям, для XiaoZhi) ────────────────────────────

@app.post("/v1/audio/speech/stream")
async def speech_stream(req: TTSRequest):
    """
    Стриминговый TTS — синтезирует и отдаёт WAV частями по предложениям.
    Полезно для XiaoZhi WebSocket сервера чтобы не ждать весь ответ.
    """
    if tts_adapter is None:
        raise HTTPException(503, "TTS adapter not initialized")

    import re
    sentences = re.split(r'(?<=[.!?…])\s+', req.text.strip())
    sentences = [s for s in sentences if s.strip()]

    async def generate():
        for sentence in sentences:
            pcm_bytes, sr = await tts_adapter.synthesize(sentence)
            if pcm_bytes:
                wav = _pcm_to_wav(pcm_bytes, sr)
                # WAV заголовок + данные как один чанк
                yield wav

    return StreamingResponse(generate(), media_type="audio/wav")


# ─── LLM прокси (опционально, если не хочется менять base_url в ai_service) ─

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """
    LLM endpoint. Совместим с OpenAI chat/completions форматом.

    Приоритет:
      1. ai_service.py (если langchain + NVIDIA/Ollama настроены)
      2. Прямой httpx вызов (LLM_BASE_URL + LLM_MODEL в .env)
      3. 503 если ничего не настроено
    """
    # Берём последнее user-сообщение
    user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), None
    )
    if not user_msg:
        raise HTTPException(400, "No user message found")

    # Способ 1: через ai_service.py (langchain)
    try:
        from ai_service import phoneguy_reply, AI_ENABLED
        if AI_ENABLED:
            reply = await asyncio.to_thread(phoneguy_reply, user_msg)
            return {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "model": req.model or "phoneguy",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
    except ImportError:
        pass

    # Способ 2: прямой httpx вызов OpenAI-compatible API
    if _llm_ready:
        reply = await _call_llm_direct(user_msg)
        if reply:
            return {
                "id": "chatcmpl-direct",
                "object": "chat.completion",
                "model": req.model or _direct_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        # LLM вызов прошёл, но вернул None (ошибка) — пробрасываем 503
        raise HTTPException(503, "LLM call failed")

    # Способ 3: ничего не настроено
    raise HTTPException(503, "LLM not configured. Set LLM_BASE_URL and LLM_MODEL in .env")


# ─── OTA конфиг для XiaoZhi V2 ───────────────────────────────────────────────
# Девайс запрашивает этот эндпоинт при старте и получает WebSocket URL.
# В веб-конфиге (192.168.4.1) вписываешь: http://твой-ip:8000/ota

MY_HOST       = os.getenv("MY_HOST", "192.168.1.182")
BRIDGE_PORT   = int(os.getenv("BRIDGE_PORT", "8765"))
BRIDGE_TOKEN  = os.getenv("BRIDGE_TOKEN", "RnOHhY0tsWFcXbtE1zdTWuCvdCmEE-orUFDNFSjwyWQ")
OTA_VERSION   = os.getenv("OTA_VERSION", "2.0.0")
OTA_FIRMWARE  = os.getenv("OTA_FIRMWARE_URL", "")
OTA_TIMEZONE  = os.getenv("OTA_TIMEZONE", "Europe/Samara")


def _get_timezone_offset(tz_name: str) -> int:
    """
    Вычисляет текущий UTC offset в минутах для указанной таймзоны.
    Работает даже без pytz — через datetime + time.
    """
    try:
        import datetime
        # Создаём datetime с нужной таймзоной
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
        # UTC offset в минутах
        offset = now.utcoffset()
        if offset is None:
            return 0
        return int(offset.total_seconds() / 60)
    except Exception:
        # Fallback: пробуем через time
        try:
            if time.daylight and time.localtime().tm_isdst:
                return -time.altzone // 60
            return -time.timezone // 60
        except Exception:
            return 180  # Samara по умолчанию


class OTARequest(BaseModel):
    application: Optional[dict] = None
    board: Optional[dict] = None


@app.post("/ota")
@app.get("/ota")
async def ota_config(request: Request, body: Optional[OTARequest] = None):
    """
    XiaoZhi V2 selfhosted OTA.
    Полностью отключаем cloud activation.
    """
    device_id = request.headers.get("Device-Id", "unknown")
    client_id = request.headers.get("Client-Id", "unknown")

    mac = device_id
    if body and body.board:
        mac = body.board.get("mac", device_id)

    logger.info(
        f"📡 OTA request: "
        f"mac={mac} "
        f"device_id={device_id} "
        f"client_id={client_id}"
    )

    ws_url = f"ws://{MY_HOST}:{BRIDGE_PORT}/xiaozhi/v1?token={BRIDGE_TOKEN}"

    logger.info(f"🔗 WS URL => {ws_url}")

    # Динамически вычисляем timezone offset (учитывает летнее/зимнее время)
    tz_offset = _get_timezone_offset(OTA_TIMEZONE)

    return {
        "server_time": {
            "timestamp": int(time.time() * 1000),
            "timezone": OTA_TIMEZONE,
            "timezone_offset": tz_offset,
        },

        # КРИТИЧНО ДЛЯ V2
        "activation": {
            "activated": True
        },

        "firmware": {
            "version": OTA_VERSION,
            "url": OTA_FIRMWARE or "",
        },

        "websocket": {
            "url": ws_url,
            "token": BRIDGE_TOKEN,
        },

        "mqtt": None,
    }


# Stub endpoint чтобы firmware не падал если всё же вызовет activate
@app.post("/ota/activate")
async def ota_activate(request: Request):
    device_id = request.headers.get("Device-Id", "unknown")

    try:
        body = await request.json()
    except Exception:
        body = {}

    logger.info(
        f"📡 Activate bypass: "
        f"device_id={device_id} body={body}"
    )

    ws_url = f"ws://{MY_HOST}:{BRIDGE_PORT}/xiaozhi/v1?token={BRIDGE_TOKEN}"

    return {
        "success": True,

        "activation": {
            "activated": True
        },

        "websocket": {
            "url": ws_url,
            "token": BRIDGE_TOKEN,
        }
    }


# ─── LLM управление (сброс истории и т.д.) ─────────────────────────────────

@app.post("/v1/llm/reset")
async def llm_reset():
    """Сбрасывает историю диалога LLM."""
    _llm_history_reset()
    try:
        from ai_service import reset_conversation_history
        reset_conversation_history()
    except ImportError:
        pass
    return {"status": "ok", "message": "LLM history cleared"}

@app.post("/ota/upload")
async def ota_upload(file: UploadFile = File(...)):
    """Загрузить новую прошивку на сервер для OTA обновления."""
    os.makedirs("firmware", exist_ok=True)
    content = await file.read()
    with open("firmware/xiaozhi.bin", "wb") as f:
        f.write(content)
    logger.info(f"✅ OTA firmware uploaded: {len(content)} bytes")
    return {"status": "ok", "size": len(content), "version": OTA_VERSION}


@app.get("/ota/firmware")
async def ota_firmware():
    """Отдаёт бинарник прошивки для OTA."""
    from fastapi.responses import FileResponse
    path = "firmware/xiaozhi.bin"
    if not os.path.exists(path):
        raise HTTPException(404, "No firmware available")
    return FileResponse(path, media_type="application/octet-stream",
                        filename="xiaozhi.bin")


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "components": {
            "stt": "ready" if stt_adapter else "unavailable",
            "tts": "ready" if tts_adapter else "unavailable",
        },
        "config": {
            "whisper_model": STT_CONFIG["stt"]["model"],
            "whisper_device": STT_CONFIG["stt"]["device"],
            "tts_engine": TTS_CONFIG["tts"]["engine"],
            "tts_device": TTS_CONFIG["tts"]["device"],
            "rvc_enabled": TTS_CONFIG["tts"]["rvc_enabled"],
        },
        "llm": {
            "ready": _llm_ready,
            "provider": LLM_PROVIDER,
            "base_url": _direct_base_url,
            "model": _direct_model,
            "has_key": bool(_direct_api_key),
        },
    }


@app.get("/")
async def root():
    return {
        "service": "Voice API",
        "endpoints": {
            "ota":        "GET  /ota",
            "stt":        "POST /v1/audio/transcriptions",
            "tts":        "POST /v1/audio/speech",
            "tts_simple": "POST /tts",
            "tts_stream": "POST /v1/audio/speech/stream",
            "chat":       "POST /v1/chat/completions",
            "health":     "GET  /health",
        },
    }
