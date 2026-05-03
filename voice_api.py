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
import wave
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import rich.logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy import signal as scipy_signal

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
        from tts_adapter import TTSAdapter
        tts_adapter = TTSAdapter(TTS_CONFIG, logger)
        ok = await tts_adapter.check_health()
        if ok:
            logger.info("✅ TTS (Chatterbox) ready")
        else:
            logger.warning("⚠️ TTS health check failed, continuing anyway")
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

    # Fallback: сырой PCM int16 @ 8000Hz (для фреймов с XiaoZhi)
    arr = np.frombuffer(file_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return arr, 8000


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
    lang = language or stt_adapter.language
    
    # Временно перекрываем language для этого запроса
    orig_lang = stt_adapter.language
    if lang:
        stt_adapter.language = lang

    text = await stt_adapter.transcribe_numpy(audio_np, 16000)

    stt_adapter.language = orig_lang  # восстанавливаем

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
    orig_prompt = tts_adapter.audio_prompt_path
    if voice != "default" and os.path.exists(voice):
        tts_adapter.audio_prompt_path = voice

    pcm_bytes, sample_rate = await tts_adapter.synthesize(input)

    tts_adapter.audio_prompt_path = orig_prompt  # восстанавливаем

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

    orig_prompt = tts_adapter.audio_prompt_path
    if req.voice and req.voice != "default" and os.path.exists(req.voice):
        tts_adapter.audio_prompt_path = req.voice

    pcm_bytes, sample_rate = await tts_adapter.synthesize(req.text)
    tts_adapter.audio_prompt_path = orig_prompt

    if not pcm_bytes:
        raise HTTPException(500, "TTS synthesis failed")

    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
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
    Тонкая обёртка над ai_service.phoneguy_reply / custom_reply.
    Совместима с OpenAI chat/completions форматом.
    """
    try:
        from ai_service import phoneguy_reply, AI_ENABLED
    except ImportError:
        raise HTTPException(503, "ai_service not available")

    if not AI_ENABLED:
        raise HTTPException(503, "LLM not initialized")

    # Берём последнее user-сообщение
    user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), None
    )
    if not user_msg:
        raise HTTPException(400, "No user message found")

    reply = await asyncio.to_thread(phoneguy_reply, user_msg)

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": req.model or "local",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


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
    }


@app.get("/")
async def root():
    return {
        "service": "Voice API",
        "endpoints": {
            "stt": "POST /v1/audio/transcriptions",
            "tts": "POST /v1/audio/speech",
            "tts_simple": "POST /tts",
            "tts_stream": "POST /v1/audio/speech/stream",
            "chat": "POST /v1/chat/completions",
            "health": "GET /health",
        },
    }
