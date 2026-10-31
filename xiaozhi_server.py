"""
xiaozhi_server.py — Всё в одном файле (v2).

HTTP :8000 — OTA сервер + OpenAI-совместимый STT/TTS/LLM API + Dashboard
WS :8765 — XiaoZhi WebSocket бридж (STT→LLM→TTS)

Запуск: python xiaozhi_server.py
OTA URL: http://твой-ip:8000/ota
Dashboard: http://твой-ip:8000/
"""

import array
import asyncio
import hashlib
import hmac
import io
import json
import logging
import math
import os
import random
import re
import secrets
import struct
import subprocess
import sys
import time
import wave
from collections import deque
from contextlib import asynccontextmanager
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import rich.logging
import uvicorn
import websockets
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy import signal as scipy_signal

import dotenv
dotenv.load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("xiaozhi")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MY_HOST = os.getenv("MY_HOST", "192.168.1.100")
HTTP_PORT = int(os.getenv("HTTP_PORT", "8000"))
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8765"))
OTA_VERSION = os.getenv("OTA_VERSION", "1.0.0")
OTA_FIRMWARE = os.getenv("OTA_FIRMWARE_URL", "")
OTA_TIMEZONE = os.getenv("OTA_TIMEZONE", "Europe/Samara")

BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "")
if not BRIDGE_TOKEN:
    BRIDGE_TOKEN = secrets.token_urlsafe(32)
    logger.warning(f"⚠️ BRIDGE_TOKEN not set! Using random: {BRIDGE_TOKEN}")

# HMAC token settings
HMAC_SECRET = os.getenv("HMAC_SECRET", BRIDGE_TOKEN)
TOKEN_EXPIRE_DAYS = int(os.getenv("TOKEN_EXPIRE_DAYS", "30"))

_raw_devices = os.getenv("ALLOWED_DEVICES", "")
ALLOWED_DEVICES: set = {d.strip().lower() for d in _raw_devices.split(",") if d.strip()}

TTS_ENGINE = os.getenv("TTS_ENGINE", "turbo").lower()
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", "auto").lower()

# Exit commands
EXIT_COMMANDS = [c.strip().lower() for c in os.getenv("EXIT_COMMANDS", "выход,выключись,exit,quit").split(",") if c.strip()]

# End-of-conversation
END_PROMPT_ENABLED = os.getenv("END_PROMPT_ENABLED", "true").lower() == "true"
END_PROMPT_TEXT = os.getenv("END_PROMPT_TEXT", "До свидания! Хорошего дня!")
END_PROMPT_TIMEOUT = int(os.getenv("END_PROMPT_TIMEOUT", "30"))

# Memory
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_MAX_ENTRIES = int(os.getenv("MEMORY_MAX_ENTRIES", "20"))
MEMORY_DIR = os.getenv("MEMORY_DIR", "data/memory")

# Function calling
TOOLS_ENABLED = os.getenv("TOOLS_ENABLED", "true").lower() == "true"
TOOL_CALL_MODE = os.getenv("TOOL_CALL_MODE", "function_call")  # function_call | intent_llm | nointent

# Listen mode
DEFAULT_LISTEN_MODE = os.getenv("LISTEN_MODE", "auto")  # auto | manual

# TTS streaming: true/sentence — по предложениям (батч TTS),
# chunk — стриминг TTS+RVC блоками через RVCStreamer, false/off — чистый батч
_raw_tts_streaming = os.getenv("TTS_STREAMING", "true").strip().lower()
if _raw_tts_streaming in ("chunk", "rvc", "block"):
    TTS_STREAMING_MODE = "chunk"
elif _raw_tts_streaming in ("false", "off", "0", "no", "batch"):
    TTS_STREAMING_MODE = "batch"
else:  # true / sentence / 1 / yes
    TTS_STREAMING_MODE = "sentence"
TTS_STREAMING = TTS_STREAMING_MODE != "batch"  # совместимость со старыми проверками

# Параметры стримингового RVC (rvc_py.realtime.RVCStreamer)
RVC_STREAM_BLOCK_MS = int(os.getenv("RVC_STREAM_BLOCK_MS", "300"))
RVC_STREAM_OVERLAP = int(os.getenv("RVC_STREAM_OVERLAP", "1024"))
RVC_STREAM_CROSSFADE = int(os.getenv("RVC_STREAM_CROSSFADE", "512"))
_raw_vad_db = os.getenv("RVC_STREAM_VAD_DB", "-45").strip().lower()
RVC_STREAM_VAD_DB = None if _raw_vad_db in ("", "none", "off") else float(_raw_vad_db)
# Контекст слева для HuBERT/F0 (сэмплы @16кГц). 0 = как в либе по умолчанию,
# 40000 (2.5s) — рекомендация автора rvc_py для стабильного голоса на фразах
RVC_STREAM_EXTRA_FRAME = int(os.getenv("RVC_STREAM_EXTRA_FRAME", "0"))

# Порция подачи PCM в RVCStreamer (~100мс @16кГц): push() за вызов
# обрабатывает один блок, поэтому батчевый чанк от TTS нужно резать
RVC_STREAM_SLICE = 1600

# EdgeTTS voice
EDGETTS_VOICE = os.getenv("EDGETTS_VOICE", "ru-RU-SvetlanaNeural")

STT_CONFIG = {
    "stt": {
        "model": os.getenv("WHISPER_MODEL", "turbo"),
        "device": os.getenv("WHISPER_DEVICE", "cuda"),
        "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        "language": os.getenv("WHISPER_LANGUAGE", "ru"),
        "beam_size": int(os.getenv("WHISPER_BEAM_SIZE", "3")),
        "target_sample_rate": 16000,
        "energy_threshold": 300,
        "min_chunk_ms": 1000,
        "silence_end_ms": 500,
    }
}

TTS_CONFIG = {
    "tts": {
        "engine": os.getenv("TTS_ENGINE", "turbo"),
        "device": os.getenv("TTS_DEVICE", "cuda"),
        # .env использует AUDIO_PROMPT_PATH (TTS_VOICE_SAMPLE — старое имя)
        "audio_prompt_path": os.getenv("TTS_VOICE_SAMPLE") or os.getenv("AUDIO_PROMPT_PATH"),
        "rvc_enabled": os.getenv("RVC_ENABLED", "false").lower() == "true",
        "rvc_model_path": os.getenv("RVC_MODEL_PATH"),
        "rvc_index_path": os.getenv("RVC_INDEX_PATH"),
        "rvc_f0_method": os.getenv("RVC_F0_METHOD", "rmvpe"),
        "rvc_pitch_shift": int(os.getenv("RVC_PITCH_SHIFT", "0")),
        "rvc_index_rate": float(os.getenv("RVC_INDEX_RATE", "0.5")),
    }
}

VAD_SILENCE_FRAMES = int(os.getenv("VAD_SILENCE_FRAMES", "25"))
VAD_MIN_RECORD_MS = int(os.getenv("VAD_MIN_RECORD_MS", "600"))
VAD_MAX_RECORD_MS = int(os.getenv("VAD_MAX_RECORD_MS", "10000"))
VAD_SPEECH_BOOST = float(os.getenv("VAD_SPEECH_BOOST", "1.5"))

# Даунлинк (TTS → девайс): полное качество выхода RVC.
# Opus поддерживает только 8000/12000/16000/24000/48000 — 40k выход RVC
# нормализуем до 48kHz (совпадает с dmix-слейвом станции — без ресемпла в maind).
DOWNLINK_SR = int(os.getenv("DOWNLINK_SR", "48000"))
FRAME_SAMPLES = DOWNLINK_SR * 60 // 1000   # 2880 @ 48k
FRAME_BYTES = FRAME_SAMPLES * 2

LLM_FALLBACK_REPLIES = [
    "У меня сейчас проблемы со связью. Попробуй позже.",
    "Сигнал пропал. Перезвони, пожалуйста.",
    "Кажется, мой мозг завис. Подожди немного.",
]

# System prompt template (Jinja2)
SYSTEM_PROMPT_TEMPLATE = os.getenv("SYSTEM_PROMPT_TEMPLATE", """You are a helpful voice assistant.
Current time: {{ current_time }}
Date: {{ current_date }}
Day of week: {{ day_of_week }}
Device ID: {{ device_id }}
Language: {{ language }}

{% if memory_context %}
Previous conversation context:
{{ memory_context }}
{% endif %}

Respond concisely and naturally. You can use emotion tags like [laugh], [sigh], [gasp] etc.
When asked to do something, use the available tools.""")

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════

_stt_adapter = None
_tts_adapter = None
_phoneguy_reply = None

_activation_codes: dict = {}
_activated_devices: set = set()

# Dashboard state
_devices: dict = {}
_log_buffer: list = []
_log_subscribers: list = []
_LOG_BUFFER_MAX = 200
_SSE_MAX_QUEUE = 50


class _DashboardLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            return
        _log_buffer.append(msg)
        if len(_log_buffer) > _LOG_BUFFER_MAX:
            _log_buffer.pop(0)
        dead = []
        for q in _log_subscribers:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _log_subscribers.remove(q)
            except ValueError:
                pass


_dash_handler = _DashboardLogHandler()
_dash_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%d/%m/%y %H:%M:%S"))
_dash_handler.setLevel(logging.INFO)
logging.getLogger("xiaozhi").addHandler(_dash_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# PLUGIN REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class ToolType(Enum):
    NONE = "none"
    WAIT = "wait"
    CHANGE_SYS_PROMPT = "change_sys_prompt"
    SYSTEM_CTL = "system_ctl"
    IOT_CTL = "iot_ctl"


class ActionResponse(Enum):
    RESPONSE = "response"
    REQLLM = "reqllm"
    RECORD = "record"
    NOTFOUND = "notfound"
    ERROR = "error"


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, dict] = {}

    def register(self, name: str, description: str, parameters: dict = None,
                 tool_type: ToolType = ToolType.NONE):
        def decorator(func):
            self._tools[name] = {
                "name": name,
                "description": description,
                "parameters": parameters or {},
                "tool_type": tool_type,
                "handler": func,
            }
            return func
        return decorator

    def get_tool_definitions(self) -> list:
        defs = []
        for tool in self._tools.values():
            defs.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            })
        return defs

    def get_schemas_for_prompt(self) -> str:
        lines = ["Available tools:"]
        for tool in self._tools.values():
            params_str = json.dumps(tool["parameters"]) if tool["parameters"] else "none"
            lines.append(f"- {tool['name']}: {tool['description']} (params: {params_str})")
        return "\n".join(lines)

    async def execute(self, name: str, arguments: dict, device_id: str = "") -> Tuple[ActionResponse, str]:
        tool = self._tools.get(name)
        if not tool:
            return ActionResponse.NOTFOUND, f"Tool '{name}' not found"
        try:
            result = await tool["handler"](arguments, device_id=device_id)
            if isinstance(result, tuple):
                return result
            return ActionResponse.RESPONSE, result
        except Exception as e:
            logger.error(f"Tool '{name}' error: {e}")
            return ActionResponse.ERROR, str(e)

    @property
    def tools(self):
        return self._tools


tool_registry = ToolRegistry()


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY SYSTEM (local short-term)
# ═══════════════════════════════════════════════════════════════════════════════

class LocalMemory:
    def __init__(self):
        self._memories: Dict[str, deque] = {}  # device_id -> deque of (role, text, ts)
        os.makedirs(MEMORY_DIR, exist_ok=True)

    def _path(self, device_id: str) -> str:
        safe = re.sub(r'[^\w\-]', '_', device_id)
        return os.path.join(MEMORY_DIR, f"{safe}.json")

    def load(self, device_id: str):
        path = self._path(device_id)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self._memories[device_id] = deque(data[-MEMORY_MAX_ENTRIES:], maxlen=MEMORY_MAX_ENTRIES)
            except Exception:
                self._memories[device_id] = deque(maxlen=MEMORY_MAX_ENTRIES)
        else:
            self._memories[device_id] = deque(maxlen=MEMORY_MAX_ENTRIES)

    def save(self, device_id: str):
        if device_id not in self._memories:
            return
        path = self._path(device_id)
        try:
            with open(path, "w") as f:
                json.dump(list(self._memories[device_id]), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Memory save error: {e}")

    def add(self, device_id: str, role: str, text: str):
        if not MEMORY_ENABLED:
            return
        if device_id not in self._memories:
            self.load(device_id)
        self._memories[device_id].append((role, text[:500], time.time()))
        self.save(device_id)

    def get_context(self, device_id: str, max_entries: int = 10) -> str:
        if not MEMORY_ENABLED:
            return ""
        if device_id not in self._memories:
            self.load(device_id)
        entries = list(self._memories[device_id])[-max_entries:]
        if not entries:
            return ""
        lines = []
        for role, text, ts in entries:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    def clear(self, device_id: str):
        self._memories[device_id] = deque(maxlen=MEMORY_MAX_ENTRIES)
        path = self._path(device_id)
        if os.path.exists(path):
            os.remove(path)


memory = LocalMemory()


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC PROMPT (Jinja2)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_system_prompt(device_id: str = "unknown") -> str:
    try:
        from jinja2 import Template
        now = time.time()
        try:
            import datetime, zoneinfo
            tz = zoneinfo.ZoneInfo(OTA_TIMEZONE)
            local_now = datetime.datetime.now(tz)
        except Exception:
            local_now = datetime.datetime.now()

        weekdays_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        template = Template(SYSTEM_PROMPT_TEMPLATE)
        return template.render(
            current_time=local_now.strftime("%H:%M"),
            current_date=local_now.strftime("%d.%m.%Y"),
            day_of_week=weekdays_ru[local_now.weekday()],
            device_id=device_id,
            language="ru",
            memory_context=memory.get_context(device_id),
        )
    except Exception as e:
        logger.warning(f"Prompt render error: {e}")
        return f"You are a helpful voice assistant. Device: {device_id}. Time: {time.strftime('%H:%M')}"


# ═══════════════════════════════════════════════════════════════════════════════
# HMAC TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_hmac_token(device_id: str) -> str:
    expire_ts = int(time.time()) + TOKEN_EXPIRE_DAYS * 86400
    payload = f"{device_id}:{expire_ts}"
    sig = hmac.new(HMAC_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{device_id}:{expire_ts}:{sig}"


def _verify_hmac_token(token: str) -> bool:
    if not token:
        return False
    parts = token.split(":")
    if len(parts) != 3:
        return secrets.compare_digest(token, BRIDGE_TOKEN)
    device_id, expire_ts_str, sig = parts
    try:
        expire_ts = int(expire_ts_str)
    except ValueError:
        return False
    if time.time() > expire_ts:
        return False
    payload = f"{device_id}:{expire_ts_str}"
    expected = hmac.new(HMAC_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO RATE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class AudioRateController:
    def __init__(self, frame_duration_ms: int = 60):
        self.frame_duration_ms = frame_duration_ms
        self._queue: asyncio.Queue = asyncio.Queue()
        self._sending = False
        self._send_task = None
        self._pcm_remainder = b""  # хвост < кадра между enqueue_pcm (стриминг)

    async def start(self, ws, opus_encoder=None):
        self._ws = ws
        self._opus_encoder = opus_encoder
        self._sending = True
        self._send_task = asyncio.create_task(self._send_loop())

    async def stop(self):
        self._sending = False
        if self._send_task:
            try:
                await asyncio.wait_for(self._send_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    async def enqueue(self, wav_bytes: bytes):
        """Батч-режим: WAV-байты → normalize → кадры. Хвост дополняется нулями."""
        await self._queue_pcm(_normalize_wav_to_16k(wav_bytes))
        await self.flush_pcm()

    async def enqueue_pcm(self, pcm_bytes: bytes):
        """Стриминг: PCM 16kHz mono int16 напрямую (уже прошёл RVC/ресэмплинг).
        Неполный кадр держится в буфере до flush_pcm(), чтобы не вставлять
        тишину в середину потока."""
        await self._queue_pcm(pcm_bytes)

    async def flush_pcm(self):
        """Дополнить остаток до полного кадра нулями и поставить в очередь."""
        if not self._pcm_remainder:
            return
        tail = self._pcm_remainder
        self._pcm_remainder = b""
        if len(tail) < FRAME_BYTES:
            tail = tail + b'\x00' * (FRAME_BYTES - len(tail))
        if self._opus_encoder:
            try:
                await self._queue.put(self._opus_encoder.encode(tail, FRAME_SAMPLES))
            except Exception:
                pass
        else:
            await self._queue.put(tail)

    async def _queue_pcm(self, pcm_data: bytes):
        if self._pcm_remainder:
            pcm_data = self._pcm_remainder + pcm_data
            self._pcm_remainder = b""
        complete = len(pcm_data) - (len(pcm_data) % FRAME_BYTES)
        if complete <= 0:
            self._pcm_remainder = pcm_data
            return
        frames = []
        if self._opus_encoder:
            for i in range(0, complete, FRAME_BYTES):
                try:
                    frames.append(self._opus_encoder.encode(pcm_data[i:i + FRAME_BYTES], FRAME_SAMPLES))
                except Exception:
                    break
        else:
            for i in range(0, complete, FRAME_BYTES):
                frames.append(pcm_data[i:i + FRAME_BYTES])
        self._pcm_remainder = pcm_data[complete:]
        for frame in frames:
            await self._queue.put(frame)

    async def _send_loop(self):
        pre_buffer = 5
        sent = 0
        start_time = None
        while self._sending or not self._queue.empty():
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if sent < pre_buffer:
                pass  # send immediately
            else:
                if start_time is None:
                    start_time = time.monotonic()
                target = start_time + (sent * self.frame_duration_ms / 1000)
                wait = target - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            try:
                await self._ws.send(frame)
            except Exception:
                break
            sent += 1
        if sent > 0:
            elapsed = time.monotonic() - start_time if start_time else 0
            logger.info(f"📤 AudioRate: {sent} frames sent in {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="get_current_time",
    description="Get the current date and time",
    parameters={"type": "object", "properties": {}},
    tool_type=ToolType.NONE,
)
async def _tool_get_time(args: dict, device_id: str = "") -> str:
    try:
        import datetime, zoneinfo
        tz = zoneinfo.ZoneInfo(OTA_TIMEZONE)
        now = datetime.datetime.now(tz)
    except Exception:
        now = datetime.datetime.now()
    return now.strftime("%H:%M %d.%m.%Y (%A)")


@tool_registry.register(
    name="exit_conversation",
    description="End the conversation gracefully. Say goodbye.",
    parameters={"type": "object", "properties": {}},
    tool_type=ToolType.SYSTEM_CTL,
)
async def _tool_exit(args: dict, device_id: str = "") -> Tuple[ActionResponse, str]:
    return ActionResponse.RESPONSE, END_PROMPT_TEXT


@tool_registry.register(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    },
)
async def _tool_weather(args: dict, device_id: str = "") -> str:
    location = args.get("location", "unknown")
    try:
        import urllib.request
        url = f"https://wttr.in/{location}?format=%C+%t+%w&lang=ru"
        req = urllib.request.Request(url, headers={"User-Agent": "XiaoZhi/1.0"})
        resp = urllib.request.urlopen(req, timeout=5)
        data = resp.read().decode().strip()
        return f"Погода в {location}: {data}"
    except Exception as e:
        return f"Не удалось получить погоду для {location}: {e}"


@tool_registry.register(
    name="web_search",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
)
async def _tool_web_search(args: dict, device_id: str = "") -> str:
    query = args.get("query", "")
    if not query:
        return "No search query provided"
    try:
        import urllib.request, urllib.parse
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={"User-Agent": "XiaoZhi/1.0"})
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])
        return "\n".join(results) if results else f"Ничего не найдено по запросу: {query}"
    except Exception as e:
        return f"Ошибка поиска: {e}"


@tool_registry.register(
    name="play_music",
    description="Play music (tells device to play music)",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Song or artist name"},
        },
        "required": ["query"],
    },
    tool_type=ToolType.SYSTEM_CTL,
)
async def _tool_play_music(args: dict, device_id: str = "") -> Tuple[ActionResponse, str]:
    query = args.get("query", "")
    return ActionResponse.RESPONSE, f"Включаю музыку: {query}"


@tool_registry.register(
    name="set_reminder",
    description="Set a reminder for a specific time",
    parameters={
        "type": "object",
        "properties": {
            "time": {"type": "string", "description": "Time in HH:MM format"},
            "message": {"type": "string", "description": "Reminder message"},
        },
        "required": ["time", "message"],
    },
)
async def _tool_reminder(args: dict, device_id: str = "") -> str:
    t = args.get("time", "??:??")
    msg = args.get("message", "")
    return f"Напоминание установлено на {t}: {msg}"


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTIONS
# ═══════════════════════════════════════════════════════════════════════════════

EMOTION_TAG_TO_TTS = {
    "[laugh]": "*laugh* ", "[laughs]": "*laugh* ",
    "[chuckle]": "*chuckle* ", "[chuckles]": "*chuckle* ",
    "[gasp]": "*gasp* ", "[gasps]": "*gasp* ",
    "[sigh]": "*sigh* ", "[sighs]": "*sigh* ",
    "[cough]": "*cough* ", "[coughs]": "*cough* ",
    "[groan]": "*groan* ", "[groans]": "*groan* ",
    "[sniff]": "*sniff* ", "[sniffs]": "*sniff* ",
    "[clear throat]": "*clears throat* ",
    "[shush]": "*shush* ",
    "[yawn]": "*yawn* ", "[yawns]": "*yawn* ",
    "[hum]": "*hum* ", "[hums]": "*hum* ",
    "[stutter]": "U-uh... ", "[stammers]": "U-uh... ",
}

# Chatterbox Turbo понимает [tag] нативно — конвертация не нужна,
# достаточно убрать неизвестные теги и нормализовать известные.
EMOTION_TAGS_TURBO = {
    "[laugh]", "[laughs]", "[chuckle]", "[chuckles]",
    "[gasp]", "[gasps]", "[sigh]", "[sighs]",
    "[cough]", "[coughs]", "[groan]", "[groans]",
    "[sniff]", "[sniffs]", "[clear throat]", "[shush]",
    "[yawn]", "[yawns]", "[hum]", "[hums]",
}

# Нормализация LLM-вариантов → канонический тег для Turbo
EMOTION_TAG_NORMALIZE = {
    "[laughs]": "[laugh]", "[chuckles]": "[chuckle]",
    "[gasps]": "[gasp]", "[sighs]": "[sigh]",
    "[coughs]": "[cough]", "[groans]": "[groan]",
    "[sniffs]": "[sniff]", "[yawns]": "[yawn]",
    "[hums]": "[hum]", "[clears throat]": "[clear throat]",
    "[stammers]": "[sigh]",
}

# Пол персонажа для DramaBox (She/He)
# Управляется через: DRAMABOX_GENDER=male или DRAMABOX_GENDER=female (по умолчанию)
DRAMABOX_GENDER = os.getenv("DRAMABOX_GENDER", "female").lower().strip()

EMOTION_TAG_TO_DRAMABOX_FEMALE = {
    "[laugh]": ("She laughs,", '"Hahaha,"'),
    "[laughs]": ("She laughs,", '"Hahaha,"'),
    "[chuckle]": ("She chuckles,", '"Hehe,"'),
    "[chuckles]": ("She chuckles,", '"Hehe,"'),
    "[gasp]": ("She gasps,", ""), "[gasps]": ("She gasps,", ""),
    "[sigh]": ("She sighs deeply,", ""), "[sighs]": ("She sighs,", ""),
    "[cough]": ("She clears her throat,", ""), "[coughs]": ("She coughs,", ""),
    "[groan]": ("She groans,", '"Ugh,"'), "[groans]": ("She groans,", '"Ugh,"'),
    "[sniff]": ("She sniffs,", ""), "[sniffs]": ("She sniffs,", ""),
    "[clear throat]": ("She clears her throat,", ""),
    "[shush]": ("She shushes,", '"Shh,"'),
    "[yawn]": ("She yawns,", '"Mmmm,"'), "[yawns]": ("She yawns,", ""),
    "[hum]": ("She hums softly,", '"Mmmm,"'), "[hums]": ("She hums,", '"Mmmm,"'),
    "[stutter]": ("She stammers,", '"I... I..."'),
    "[stammers]": ("She stammers,", '"I... I..."'),
}

EMOTION_TAG_TO_DRAMABOX_MALE = {
    "[laugh]": ("He laughs,", '"Hahaha,"'),
    "[laughs]": ("He laughs,", '"Hahaha,"'),
    "[chuckle]": ("He chuckles,", '"Hehe,"'),
    "[chuckles]": ("He chuckles,", '"Hehe,"'),
    "[gasp]": ("He gasps,", ""), "[gasps]": ("He gasps,", ""),
    "[sigh]": ("He sighs deeply,", ""), "[sighs]": ("He sighs,", ""),
    "[cough]": ("He clears his throat,", ""), "[coughs]": ("He coughs,", ""),
    "[groan]": ("He groans,", '"Ugh,"'), "[groans]": ("He groans,", '"Ugh,"'),
    "[sniff]": ("He sniffs,", ""), "[sniffs]": ("He sniffs,", ""),
    "[clear throat]": ("He clears his throat,", ""),
    "[shush]": ("He shushes,", '"Shh,"'),
    "[yawn]": ("He yawns,", '"Mmmm,"'), "[yawns]": ("He yawns,", ""),
    "[hum]": ("He hums softly,", '"Mmmm,"'), "[hums]": ("He hums,", '"Mmmm,"'),
    "[stutter]": ("He stammers,", '"I... I..."'),
    "[stammers]": ("He stammers,", '"I... I..."'),
}

# Активная таблица — выбирается один раз при старте
EMOTION_TAG_TO_DRAMABOX = (
    EMOTION_TAG_TO_DRAMABOX_MALE
    if DRAMABOX_GENDER == "male"
    else EMOTION_TAG_TO_DRAMABOX_FEMALE
)

XIAOZHI_EMOTIONS = {
    "neutral": "😶", "happy": "🙂", "laughing": "😆", "funny": "😂",
    "sad": "😔", "angry": "😠", "crying": "😭", "loving": "😍",
    "embarrassed": "😳", "surprised": "😯", "shocked": "😱",
    "thinking": "🤔", "winking": "😉", "cool": "😎", "relaxed": "😌",
    "delicious": "🤤", "kissy": "😘", "confident": "😏",
    "sleepy": "😴", "silly": "😜", "confused": "🙄",
}


def _convert_emotion_tags_for_tts(text: str, engine: str = None) -> str:
    if engine is None:
        engine = TTS_ENGINE
    if engine == "turbo":
        # Turbo понимает [tag] нативно — нормализуем варианты LLM
        # и убираем что не понимает Turbo (чтобы не произносилось вслух)
        def _replace_turbo(m):
            tag = m.group(0).strip().lower()
            tag = EMOTION_TAG_NORMALIZE.get(tag, tag)
            return tag + " " if tag in EMOTION_TAGS_TURBO else " "
        return re.sub(r"\[[^\]]+\]", _replace_turbo, text).strip()
    # base/multilingual/edge/openai — конвертация в *tag* формат
    return re.sub(r"\[[^\]]+\]", lambda m: EMOTION_TAG_TO_TTS.get(m.group(0).strip(), " "), text).strip()


def _convert_emotion_tags_for_dramabox(text: str) -> str:
    def _r(m):
        entry = EMOTION_TAG_TO_DRAMABOX.get(m.group(0).strip())
        if not entry:
            return " "
        d, s = entry
        return f"{d} {s} " if s else f"{d} "
    return re.sub(r"\[[^\]]+\]", _r, text).strip()


def _clean_emotion_tags(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text).strip()


def _detect_emotion(text: str) -> str:
    text_lower = text.lower()
    hints = {
        "laughing": ["[laugh]"], "funny": ["[laugh]", "haha", "lol"],
        "sad": ["[sigh]", "[groan]", "sorry", "sad", "unfortunately"],
        "thinking": ["[clear throat]", "hmm", "let me think", "well", "um,"],
        "surprised": ["[gasp]", "oh!", "wow", "really?"],
        "happy": ["[chuckle]", "great", "awesome", "love", "happy"],
        "embarrassed": ["[sniff]", "[cough]", "oops"],
        "angry": ["angry", "mad", "furious"],
        "loving": ["love you", "my dear", "superstar"],
        "cool": ["cool", "rock star", "awesome"],
        "confident": ["of course", "absolutely", "i'm sure", "i know"],
        "silly": ["silly", "crazy", "ridiculous"],
        "confused": ["[shush]", "what do you mean", "i don't understand"],
        "relaxed": ["relax", "calm down", "don't worry"],
        "sleepy": ["tired", "sleepy", "yawn"],
    }
    for emotion, keywords in hints.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return emotion
    return "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# OPUS
# ═══════════════════════════════════════════════════════════════════════════════

OPUS_AVAILABLE = False
try:
    import opuslib
    _test_enc = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
    _test_dec = opuslib.Decoder(16000, 1)
    del _test_enc, _test_dec
    OPUS_AVAILABLE = True
    logger.info("✅ opuslib available")
except Exception as e:
    logger.warning(f"⚠️ opuslib NOT available: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE VAD
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveVAD:
    def __init__(self):
        self.threshold = None
        self.silence_frames_limit = VAD_SILENCE_FRAMES
        self.min_frames = max(VAD_MIN_RECORD_MS // 60, 3)
        self.max_frames = max(VAD_MAX_RECORD_MS // 60, VAD_SILENCE_FRAMES + 5)
        self.speech_boost = VAD_SPEECH_BOOST
        self.frame_count = 0
        self.silence_count = 0
        self.speech_started = False
        self._calibration_values = []
        self._calibration_frames = 25
        self._calibration_warmup = 10
        self._calibrated = False
        self.last_rms = 0

    def reset(self):
        self.frame_count = 0
        self.silence_count = 0
        self.speech_started = False
        self._calibration_values.clear()
        self._calibrated = False

    def _compute_rms(self, pcm_data: bytes) -> int:
        if len(pcm_data) < 2:
            return 0
        samples = array.array('h', pcm_data[:len(pcm_data) & ~1])
        if not samples:
            return 0
        return int(math.sqrt(sum(s * s for s in samples) / len(samples)) * 100)

    def _calibrate(self):
        vals = self._calibration_values[self._calibration_warmup:]
        if len(vals) < 3:
            vals = self._calibration_values
        quiet = [v for v in vals if v < 150000]
        if len(quiet) < 3:
            if len(vals) >= 3:
                median = sorted(vals)[0]
                self.threshold = max(int(median * self.speech_boost), 15000)
            else:
                self.threshold = 15000
        else:
            sorted_quiet = sorted(quiet)
            median = sorted_quiet[len(sorted_quiet) // 2]
            self.threshold = max(int(median * self.speech_boost), 8000)
            logger.info(f"VAD calibrated: median={median:.0f} threshold={self.threshold}")
        self._calibrated = True

    def process_frame(self, pcm_data: bytes) -> bool:
        self.frame_count += 1
        rms = self._compute_rms(pcm_data)
        self.last_rms = rms
        if not self._calibrated:
            self._calibration_values.append(rms)
            if len(self._calibration_values) >= self._calibration_frames:
                self._calibrate()
            return False
        is_speech = rms > self.threshold
        if is_speech:
            self.silence_count = 0
            self.speech_started = True
        elif self.speech_started:
            self.silence_count += 1
        if self.frame_count >= self.max_frames:
            return True
        if self.speech_started and self.silence_count >= self.silence_frames_limit:
            return True
        return False

    def get_stats(self) -> dict:
        return {"rms": self.last_rms, "threshold": self.threshold, "calibrated": self._calibrated,
                "silence": self.silence_count, "speech": self.speech_started, "frames": self.frame_count}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    # ВАЖНО: seek ПОСЛЕ закрытия wave файла,
    # иначе data_size в заголовке остаётся 0 и _parse_wav читает 0 фреймов
    buf.seek(0)
    return buf.read()


def _load_audio_to_numpy(file_bytes: bytes, filename: str = "") -> tuple:
    try:
        buf = io.BytesIO(file_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if wf.getnchannels() == 2:
                arr = arr.reshape(-1, 2).mean(axis=1)
            return arr, sr
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
            input=file_bytes, capture_output=True, timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0, 16000
    except Exception:
        pass
    return np.frombuffer(file_bytes, dtype=np.int16).astype(np.float32) / 32768.0, 8000


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    n_out = int(round(len(audio) * dst_sr / src_sr))
    return scipy_signal.resample(audio, n_out).astype(np.float32)


def _split_by_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    result = []
    pending_tags = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(r"(\[[^\]]+\]\s*)+", part):
            pending_tags += part + " "
        else:
            result.append((pending_tags + part).strip())
            pending_tags = ""
    if pending_tags.strip():
        if result:
            result[-1] += " " + pending_tags.strip()
        else:
            result.append(pending_tags.strip())
    return result if result else [text]


def _ws_url() -> str:
    url = f"ws://{MY_HOST}:{BRIDGE_PORT}/xiaozhi/v1/"
    if BRIDGE_TOKEN:
        url += f"?token={BRIDGE_TOKEN}"
    return url


def _parse_wav(wav_bytes: bytes) -> tuple:
    if len(wav_bytes) < 44 or wav_bytes[:4] != b'RIFF':
        return wav_bytes, 0, 1
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, 'rb') as wf:
            return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()
    except Exception:
        pass
    pos = 12
    while pos < len(wav_bytes) - 8:
        chunk_id = wav_bytes[pos:pos + 4]
        chunk_size = struct.unpack_from('<I', wav_bytes, pos + 4)[0]
        if chunk_id == b'data':
            return wav_bytes[pos + 8:pos + 8 + chunk_size], struct.unpack_from('<I', wav_bytes, 24)[0], 1
        pos += 8 + chunk_size + (chunk_size % 2)
    return wav_bytes[44:], 0, 1


def _resample_pcm(pcm_bytes: bytes, src_sr: int, dst_sr: int) -> bytes:
    if src_sr == dst_sr or src_sr == 0:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes[:len(pcm_bytes) & ~1], dtype=np.int16).astype(np.float64)
    if len(samples) == 0:
        return pcm_bytes
    frac = Fraction(dst_sr, src_sr)
    resampled = scipy_signal.resample_poly(samples, frac.numerator, frac.denominator, window=('kaiser', 14.76))
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def _stereo_to_mono(pcm_bytes: bytes) -> bytes:
    if len(pcm_bytes) < 4:
        return pcm_bytes
    samples = array.array('h', pcm_bytes[:len(pcm_bytes) & ~1])
    n = len(samples) // 2
    mono = array.array('h', [(samples[i * 2] + samples[i * 2 + 1]) // 2 for i in range(n)])
    return mono.tobytes()


def _normalize_wav_to_16k(wav_bytes: bytes) -> bytes:
    pcm_data, sr, n_ch = _parse_wav(wav_bytes)
    if n_ch == 2:
        pcm_data = _stereo_to_mono(pcm_data)
    if not pcm_data:
        logger.warning(f"_normalize_wav_to_16k: empty PCM after parse (sr={sr})")
        return b''
    if sr == 0:
        # sr=0 — fallback парсер не смог определить SR.
        # Без ресэмплинга raw PCM пошёл бы в Opus как DOWNLINK_SR →
        # девайс играл бы 24к-источник в 2x быстрее (октава вверх).
        # Fallback: считаем источник Chatterbox (24000Hz) — и ресэмпл до даунлинка.
        logger.warning(f"_normalize_wav_to_16k: sr=0, WAV заголовок нечитаем, считаем 24000Hz → {DOWNLINK_SR}Hz")
        pcm_data = _resample_pcm(pcm_data, 24000, DOWNLINK_SR)
        return pcm_data
    if sr != 0 and sr != DOWNLINK_SR:
        logger.debug(f"_normalize_wav_to_16k: resampling {sr}Hz → {DOWNLINK_SR}Hz ({len(pcm_data)} bytes)")
        pcm_data = _resample_pcm(pcm_data, sr, DOWNLINK_SR)
    return pcm_data


# ═══════════════════════════════════════════════════════════════════════════════
# TTS ENGINES
# ═══════════════════════════════════════════════════════════════════════════════

async def _tts_edge(text: str) -> Tuple[bytes, int]:
    """EdgeTTS — free, no API key."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, EDGETTS_VOICE)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        if not audio_data:
            return b"", 0
        # edge-tts outputs MP3, need to convert via ffmpeg
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate(input=audio_data)
        if stdout:
            return stdout, 16000
        return b"", 0
    except Exception as e:
        logger.error(f"EdgeTTS error: {e}")
        return b"", 0


async def _tts_openai(text: str) -> Tuple[bytes, int]:
    """OpenAI-compatible TTS API."""
    try:
        import httpx
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
        if not api_key:
            return b"", 0
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url}/audio/speech",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "tts-1", "input": text, "voice": voice, "response_format": "wav"},
            )
            if resp.status_code == 200:
                pcm, sr, _ = _parse_wav(resp.content)
                return pcm, sr
        return b"", 0
    except Exception as e:
        logger.error(f"OpenAI TTS error: {e}")
        return b"", 0


async def run_tts(text: str) -> Tuple[bytes, int]:
    """Route TTS to the configured engine."""
    engine = TTS_ENGINE
    if engine == "edge":
        return await _tts_edge(text)
    elif engine == "openai":
        return await _tts_openai(text)
    else:
        # Chatterbox / local adapter
        if _tts_adapter is None:
            raise RuntimeError("TTS not initialized")
        return await _tts_adapter.synthesize(text)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

async def run_stt(pcm_bytes: bytes) -> str:
    if _stt_adapter is None:
        raise RuntimeError("STT not initialized")
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return await _stt_adapter.transcribe_numpy(arr, 16000)


async def run_llm(text: str, device_id: str = "unknown") -> str:
    if _phoneguy_reply is None:
        raise RuntimeError("LLM not initialized")
    # Add memory context
    ctx = memory.get_context(device_id)
    if ctx:
        text = f"[Context]:\n{ctx}\n\n[User]: {text}"
    return await asyncio.to_thread(_phoneguy_reply, text)


def _check_exit_command(text: str) -> bool:
    t = text.strip().lower()
    return any(cmd in t for cmd in EXIT_COMMANDS)


async def _process_audio_pipeline(ws, audio_buffer: bytearray, device_id: str,
                                  opus_encoder, is_aborted, session_id: str = "",
                                  listen_mode: str = "auto"):
    if not audio_buffer:
        return

    buf_seconds = len(audio_buffer) / (16000 * 2)
    logger.info(f"🎤 [{device_id}] Processing {len(audio_buffer)} bytes ({buf_seconds:.1f}s)")

    try:
        await _run_pipeline(ws, audio_buffer, device_id, opus_encoder, is_aborted, session_id, listen_mode)
    except asyncio.CancelledError:
        logger.info(f"⏹️ [{device_id}] Pipeline canceled")
        # Send tts/stop so device exits Speaking state
        try:
            await ws.send(json.dumps({"type": "tts", "state": "stop", "session_id": session_id}))
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)


async def _run_pipeline(ws, audio_buffer: bytearray, device_id: str,
                        opus_encoder, is_aborted, session_id: str = "",
                        listen_mode: str = "auto"):
    # STT
    try:
        text = await run_stt(bytes(audio_buffer))
        audio_buffer.clear()
        logger.info(f"📝 [{device_id}] STT: '{text}'")
    except Exception as e:
        logger.error(f"STT error: {e}", exc_info=True)
        audio_buffer.clear()
        return

    if not text.strip():
        return

    await ws.send(json.dumps({"type": "stt", "text": text, "session_id": session_id}))

    # Save to memory
    memory.add(device_id, "user", text)

    # Check exit command
    if _check_exit_command(text):
        logger.info(f"🚪 [{device_id}] Exit command detected: '{text}'")
        farewell = END_PROMPT_TEXT
        await ws.send(json.dumps({"type": "tts", "state": "start", "session_id": session_id}))
        await ws.send(json.dumps({"type": "tts", "state": "sentence_start", "text": farewell, "session_id": session_id}))
        try:
            pcm_bytes, sr = await run_tts(farewell)
            if pcm_bytes:
                wav_bytes = _pcm_to_wav(pcm_bytes, sr)
                rate_ctrl = AudioRateController()
                await rate_ctrl.start(ws, opus_encoder)
                await rate_ctrl.enqueue(wav_bytes)
                await rate_ctrl.stop()
        except Exception as e:
            logger.error(f"TTS error on exit: {e}")
        await ws.send(json.dumps({"type": "tts", "state": "stop", "session_id": session_id}))
        await ws.close(1000, "Exit command")
        return

    # Send tts/start IMMEDIATELY after STT (like official server)
    await ws.send(json.dumps({"type": "tts", "state": "start", "session_id": session_id}))

    # Function calling / tool use
    if TOOLS_ENABLED and TOOL_CALL_MODE == "function_call":
        text, tool_result = await _try_function_call(text, device_id)
        if tool_result:
            memory.add(device_id, "assistant", tool_result)

    # LLM
    full_reply = None
    try:
        full_reply = await run_llm(text, device_id)
    except Exception as e:
        logger.error(f"LLM error: {e}")

    if not full_reply or not full_reply.strip():
        full_reply = random.choice(LLM_FALLBACK_REPLIES)

    logger.info(f"🤖 [{device_id}] Reply: '{full_reply}'")
    memory.add(device_id, "assistant", full_reply)

    # TTS streaming: chunk (RVCStreamer блоками) / sentence (по предложениям) / batch
    if _use_chunk_streaming():
        await _tts_stream_rvc(ws, full_reply, opus_encoder, session_id, device_id)
    elif TTS_STREAMING:
        await _tts_stream_sentences(ws, full_reply, opus_encoder, session_id, device_id)
    else:
        if TTS_ENGINE == "dramabox":
            tts_text = _convert_emotion_tags_for_dramabox(full_reply)
        else:
            tts_text = _convert_emotion_tags_for_tts(full_reply)
        display_text = _clean_emotion_tags(full_reply)
        if display_text:
            await ws.send(json.dumps({"type": "tts", "state": "sentence_start", "text": display_text, "session_id": session_id}))
        try:
            pcm_bytes, sr = await run_tts(tts_text)
            if pcm_bytes:
                wav_bytes = _pcm_to_wav(pcm_bytes, sr)
                rate_ctrl = AudioRateController()
                await rate_ctrl.start(ws, opus_encoder)
                await rate_ctrl.enqueue(wav_bytes)
                await rate_ctrl.stop()
        except Exception as e:
            logger.error(f"TTS error: {e}")

    await ws.send(json.dumps({"type": "tts", "state": "stop", "session_id": session_id}))
    logger.info(f"✅ [{device_id}] Response complete")


async def _tts_stream_sentences(ws, full_reply: str, opus_encoder, session_id: str, device_id: str):
    """Sentence-by-sentence TTS with overlapped generation and playback."""
    sentences = _split_by_sentences(full_reply)
    if not sentences:
        return

    logger.info(f"🗣️ [{device_id}] Streaming {len(sentences)} sentences")

    rate_ctrl = AudioRateController()
    await rate_ctrl.start(ws, opus_encoder)

    for i, sentence in enumerate(sentences):
        if TTS_ENGINE == "dramabox":
            tts_text = _convert_emotion_tags_for_dramabox(sentence)
        else:
            tts_text = _convert_emotion_tags_for_tts(sentence)

        try:
            pcm_bytes, sr = await run_tts(tts_text)
            if pcm_bytes:
                wav_bytes = _pcm_to_wav(pcm_bytes, sr)
                # Send sentence_start right before first audio packet (like official server)
                display_text = _clean_emotion_tags(sentence)
                if display_text:
                    await ws.send(json.dumps({
                        "type": "tts",
                        "state": "sentence_start",
                        "text": display_text,
                        "session_id": session_id,
                    }))
                await rate_ctrl.enqueue(wav_bytes)
                logger.info(f"🗣️ [{device_id}] Sentence {i + 1}/{len(sentences)} ({len(tts_text)} chars)")
        except Exception as e:
            logger.error(f"TTS sentence error: {e}")

    await rate_ctrl.stop()


def _use_chunk_streaming() -> bool:
    """chunk-режим: локальный TTS-адаптер + включённый RVC + стример доступен."""
    if TTS_STREAMING_MODE != "chunk" or TTS_ENGINE in ("edge", "openai", "dramabox"):
        return False
    tts_cfg = TTS_CONFIG["tts"]
    if _tts_adapter is None or not tts_cfg.get("rvc_enabled") or not tts_cfg.get("rvc_model_path"):
        return False
    try:
        import rvc_py.realtime  # noqa: F401
        return True
    except Exception as e:
        logger.warning(f"⚠️ RVC streaming unavailable, falling back to sentence mode: {e}")
        return False


def _make_rvc_streamer(input_sr: int = 16000):
    """RVCStreamer на utterance. Модели берутся из кеша rvc_infer._rvc_cache —
    VRAM не дублируется с батч-путём. Питается PCM 16кГц (TTS-чанк ресэмплится
    целиком ДО нарезки)."""
    from rvc_py.realtime import RVCStreamer

    tts_cfg = TTS_CONFIG["tts"]
    return RVCStreamer(
        rvc_model_path=tts_cfg["rvc_model_path"],
        device=tts_cfg.get("device", "cuda"),
        f0_up_key=tts_cfg.get("rvc_pitch_shift", 0),
        f0_method=tts_cfg.get("rvc_f0_method", "rmvpe"),
        index_path=tts_cfg.get("rvc_index_path"),
        index_rate=tts_cfg.get("rvc_index_rate", 0.5),
        input_sr=input_sr,
        output_sr=DOWNLINK_SR,
        block_size=max(1, int(16000 * RVC_STREAM_BLOCK_MS / 1000)),
        overlap=RVC_STREAM_OVERLAP,
        crossfade=RVC_STREAM_CROSSFADE,
        vad_threshold_db=RVC_STREAM_VAD_DB,
        extra_frame=RVC_STREAM_EXTRA_FRAME,
    )


def _resample_f32(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Ресэмпл float32 gcd-методом — та же математика, что в rvc_py.realtime."""
    if src == dst:
        return audio
    g = math.gcd(src, dst)
    return scipy_signal.resample_poly(audio, dst // g, src // g).astype(np.float32)


def _float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


async def _warmup_rvc_stream(input_sr: int = 16000):
    """Прогрев стримера: модели уже в кеше после батч-warmup check_health,
    здесь прогревается инференс-путь и ловятся ошибки конфига до первого клиента."""
    streamer = _make_rvc_streamer(input_sr)
    t = np.arange(input_sr // 10, dtype=np.float32) / float(input_sr)
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    await asyncio.to_thread(streamer.push, tone)
    await asyncio.to_thread(streamer.flush)
    streamer.reset()


async def _tts_stream_rvc(ws, full_reply: str, opus_encoder, session_id: str, device_id: str):
    """Chunk-стриминг: TTS-чанки → RVCStreamer.push() → инкрементальная отправка.

    RVC-стадия начинает отдавать звук после накопления первого блока
    (~RVC_STREAM_BLOCK_MS), не дожидаясь конвертации всего предложения.
    Блоки конвертируются в asyncio.to_thread — inference синхронный,
    отправка и paced-плейер продолжают жить в event loop (abort/ping работают).
    """
    sentences = _split_by_sentences(full_reply)
    if not sentences:
        return

    logger.info(f"🗣️ [{device_id}] Streaming {len(sentences)} sentences (RVC chunk mode)")

    rate_ctrl = AudioRateController()
    await rate_ctrl.start(ws, opus_encoder)
    streamer = None

    try:
        for i, sentence in enumerate(sentences):
            if TTS_ENGINE == "dramabox":
                tts_text = _convert_emotion_tags_for_dramabox(sentence)
            else:
                tts_text = _convert_emotion_tags_for_tts(sentence)
            display_text = _clean_emotion_tags(sentence)

            async def _send_sentence_start():
                if display_text:
                    await ws.send(json.dumps({
                        "type": "tts", "state": "sentence_start",
                        "text": display_text, "session_id": session_id,
                    }))

            try:
                sent_start = False

                async for chunk_np, chunk_sr in _tts_adapter.synthesize_stream(tts_text):
                    # Ресэмпл чанка ЦЕЛИКОМ: resample_poly на мелких кусочках
                    # даёт щелчки на краях. Стример питается PCM 16кГц.
                    if chunk_sr != 16000:
                        chunk_np = await asyncio.to_thread(
                            _resample_f32, chunk_np, chunk_sr, 16000
                        )
                    if streamer is None:
                        streamer = _make_rvc_streamer()
                    for off in range(0, len(chunk_np), RVC_STREAM_SLICE):
                        converted = await asyncio.to_thread(
                            streamer.push, chunk_np[off:off + RVC_STREAM_SLICE]
                        )
                        if converted is not None and len(converted) > 0:
                            if not sent_start:
                                await _send_sentence_start()
                                sent_start = True
                            await rate_ctrl.enqueue_pcm(_float_to_pcm16_bytes(converted))

                if streamer is not None:
                    tail = await asyncio.to_thread(streamer.flush)
                    if tail is not None and len(tail) > 0:
                        if not sent_start:
                            await _send_sentence_start()
                            sent_start = True
                        await rate_ctrl.enqueue_pcm(_float_to_pcm16_bytes(tail))
                    streamer.reset()
                await rate_ctrl.flush_pcm()

                if not sent_start:
                    await _send_sentence_start()

                logger.info(f"🗣️ [{device_id}] Sentence {i + 1}/{len(sentences)} ({len(tts_text)} chars)")

            except Exception as e:
                logger.error(f"TTS chunk-stream error (sentence {i + 1}): {e}", exc_info=True)
                if streamer is not None:
                    try:
                        streamer.reset()
                    except Exception:
                        pass
                # Fallback: это предложение батчевым путём, чтобы звук не прерывался
                try:
                    pcm_bytes, sr = await run_tts(tts_text)
                    if pcm_bytes:
                        await _send_sentence_start()
                        await rate_ctrl.enqueue(_pcm_to_wav(pcm_bytes, sr))
                except Exception as e2:
                    logger.error(f"TTS batch fallback error (sentence {i + 1}): {e2}")
    finally:
        await rate_ctrl.stop()


async def _try_function_call(text: str, device_id: str) -> Tuple[str, str]:
    """Try to extract and execute a function call from LLM response."""
    try:
        # Ask LLM to use tools
        system_prompt = _render_system_prompt(device_id)
        tools_json = json.dumps(tool_registry.get_tool_definitions(), ensure_ascii=False)

        tool_prompt = f"""{system_prompt}

{tool_registry.get_schemas_for_prompt()}

User said: "{text}"

If the user wants to use a tool, respond ONLY with a JSON object:
{{"tool_call": true, "name": "tool_name", "arguments": {{...}}}}

If no tool needed, respond normally (not JSON)."""

        reply = await asyncio.to_thread(_phoneguy_reply, tool_prompt)

        # Try to parse tool call
        reply = reply.strip()
        if reply.startswith("{"):
            try:
                data = json.loads(reply)
                if data.get("tool_call") and data.get("name"):
                    result_resp, result_text = await tool_registry.execute(
                        data["name"], data.get("arguments", {}), device_id
                    )
                    logger.info(f"🔧 [{device_id}] Tool '{data['name']}' → {result_text[:100]}")
                    if result_resp == ActionResponse.RESPONSE:
                        return text, result_text
                    elif result_resp == ActionResponse.REQLLM:
                        return f"{text}\n\nTool result: {result_text}", ""
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.error(f"Function call error: {e}")
    return text, ""


# ═══════════════════════════════════════════════════════════════════════════════
# HELLO HANDSHAKE
# ═══════════════════════════════════════════════════════════════════════════════

async def handle_hello(ws, data: dict) -> tuple:
    device_id = data.get("device_id") or data.get("mac") or "unknown"
    if device_id == "unknown":
        try:
            header_mac = ws.request.headers.get("device-id", "")
            if header_mac:
                device_id = header_mac
        except Exception:
            pass

    audio_params = data.get("audio_params", {})
    logger.info(f"👋 Hello: device={device_id} audio={audio_params}")

    if device_id != "unknown" and not _check_device(device_id):
        logger.warning(f"❌ Device {device_id} not allowed")
        await ws.send(json.dumps({"type": "hello", "error": "device_not_allowed"}))
        return False, device_id, "pcm"

    device_format = (audio_params.get("format") or "pcm").lower()
    if AUDIO_FORMAT == "auto":
        negotiated = device_format if device_format in ("opus", "pcm") else "pcm"
        if negotiated == "opus" and not OPUS_AVAILABLE:
            negotiated = "pcm"
    else:
        negotiated = AUDIO_FORMAT
        if AUDIO_FORMAT == "opus" and not OPUS_AVAILABLE:
            negotiated = "pcm"

    response = {
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "session_id": secrets.token_hex(16),
        "audio_params": {
            "format": negotiated,
            "sample_rate": DOWNLINK_SR,
            "channels": 1,
            "frame_duration": 60,
        },
    }
    await ws.send(json.dumps(response))
    logger.info(f"✅ Hello OK: {device_id} | format={negotiated}")
    return True, device_id, negotiated


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stt_adapter, _tts_adapter, _phoneguy_reply
    logger.info("🚀 Loading AI adapters...")

    try:
        from stt_adapter import STTAdapter
        _stt_adapter = STTAdapter(STT_CONFIG, logger)
        logger.info("✅ STT (Whisper) ready")
    except Exception as e:
        logger.error(f"❌ STT: {e}")

    # TTS: skip loading adapter if using edge or openai engine
    if TTS_ENGINE not in ("edge", "openai"):
        try:
            from tts_adapter import ChatterboxAdapter as TTSAdapter
            _tts_adapter = TTSAdapter(TTS_CONFIG, logger)
            ok = await _tts_adapter.check_health()
            logger.info("✅ TTS (Chatterbox) ready" if ok else "⚠️ TTS health warn")
        except Exception as e:
            logger.error(f"❌ TTS: {e}")
    else:
        logger.info(f"✅ TTS engine: {TTS_ENGINE}")

    # Прогрев RVC-стримера (после батч-warmup модели уже в кеше rvc_infer)
    if _tts_adapter and _use_chunk_streaming():
        try:
            await _warmup_rvc_stream()
            logger.info("✅ RVC streaming warm")
        except Exception as e:
            logger.warning(f"⚠️ RVC streaming warmup failed: {e}")

    try:
        from ai_service import phoneguy_reply
        _phoneguy_reply = phoneguy_reply
        logger.info("✅ LLM (ai_service) ready")
    except Exception as e:
        logger.warning(f"⚠️ LLM: {e}")

    logger.info(f"✅ Plugins: {len(tool_registry.tools)} tools registered")
    logger.info(f"✅ Memory: {'enabled' if MEMORY_ENABLED else 'disabled'}")
    logger.info(f"✅ Function calling: {'enabled' if TOOLS_ENABLED else 'disabled'} ({TOOL_CALL_MODE})")
    logger.info(f"✅ TTS streaming: {TTS_STREAMING_MODE}")
    logger.info(f"✅ Exit commands: {EXIT_COMMANDS}")
    logger.info(f"✅ End prompt: {'enabled' if END_PROMPT_ENABLED else 'disabled'}")

    yield

    if _stt_adapter:
        _stt_adapter.close()
    logger.info("👋 Shutdown")


app = FastAPI(
    title="XiaoZhi Server",
    description="OTA + OpenAI-compatible STT/TTS/LLM + WebSocket bridge + Dashboard",
    version="2.0.0",
    lifespan=lifespan,
)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ─── OTA ─────────────────────────────────────────────────────────────────────

class OTARequest(BaseModel):
    application: Optional[dict] = None
    board: Optional[dict] = None


@app.post("/xiaozhi/ota/")
@app.get("/xiaozhi/ota/")
@app.post("/xiaozhi/ota")
@app.get("/xiaozhi/ota")
@app.post("/ota")
@app.get("/ota")
async def ota_config(request: Request, body: Optional[OTARequest] = None):
    device_id = request.headers.get("device-id", "") or request.headers.get("Device-Id", "unknown")
    client_id = request.headers.get("client-id", "") or request.headers.get("Client-Id", "unknown")
    mac = body.board.get("mac", device_id) if body and body.board else device_id

    ws_url = _ws_url()
    tz_offset = _get_timezone_offset(OTA_TIMEZONE)

    # Generate HMAC token for device
    hmac_token = _generate_hmac_token(mac) if HMAC_SECRET != BRIDGE_TOKEN else BRIDGE_TOKEN

    response = {
        "server_time": {
            "timestamp": int(time.time() * 1000),
            "timezone_offset": tz_offset,
        },
        "firmware": {"version": OTA_VERSION, "url": OTA_FIRMWARE or ""},
        "websocket": {"url": ws_url, "token": hmac_token},
    }

    logger.info(f"📡 OTA: mac={mac} client={client_id}")
    return response


@app.post("/xiaozhi/ota/activate")
@app.post("/ota/activate")
async def ota_activate(request: Request):
    device_id = request.headers.get("device-id", "") or request.headers.get("Device-Id", "unknown")
    try:
        body = await request.json()
    except Exception:
        body = {}

    logger.info(f"📡 Activate: device={device_id}")
    ws = _ws_url()
    _activated_devices.add(device_id)
    return {"success": True, "websocket": {"url": ws, "token": BRIDGE_TOKEN}}


@app.post("/ota/upload")
async def ota_upload(file: UploadFile = File(...)):
    os.makedirs("firmware", exist_ok=True)
    content = await file.read()
    with open("firmware/xiaozhi.bin", "wb") as f:
        f.write(content)
    logger.info(f"✅ Firmware uploaded: {len(content)} bytes")
    return {"status": "ok", "size": len(content), "version": OTA_VERSION}


@app.get("/ota/firmware")
async def ota_firmware():
    path = "firmware/xiaozhi.bin"
    if not os.path.exists(path):
        raise HTTPException(404, "No firmware available")
    return FileResponse(path, media_type="application/octet-stream", filename="xiaozhi.bin")


# ─── STT ─────────────────────────────────────────────────────────────────────

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    if _stt_adapter is None:
        raise HTTPException(503, "STT not initialized")
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Empty audio file")
    audio_np, src_sr = _load_audio_to_numpy(file_bytes, file.filename or "")
    if src_sr != 16000:
        audio_np = _resample(audio_np, src_sr, 16000)
    orig_lang = _stt_adapter.language
    if language:
        _stt_adapter.language = language
    text = await _stt_adapter.transcribe_numpy(audio_np, 16000)
    _stt_adapter.language = orig_lang
    return text if response_format == "text" else {"text": text}


@app.post("/stt")
async def stt_simple(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    return await transcriptions(file=file, language=language)


# ─── TTS ─────────────────────────────────────────────────────────────────────

@app.post("/v1/audio/speech")
async def speech(
    model: str = Form("tts-1"),
    input: str = Form(...),
    voice: str = Form("default"),
    response_format: str = Form("wav"),
    speed: float = Form(1.0),
):
    if not input.strip():
        raise HTTPException(400, "Empty text")
    pcm_bytes, sr = await run_tts(input)
    if not pcm_bytes:
        raise HTTPException(500, "TTS synthesis failed")
    return StreamingResponse(
        io.BytesIO(_pcm_to_wav(pcm_bytes, sr)),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    response_format: Optional[str] = "wav"


@app.post("/tts")
async def tts_simple(req: TTSRequest):
    pcm_bytes, sr = await run_tts(req.text)
    if not pcm_bytes:
        raise HTTPException(500, "TTS synthesis failed")
    return Response(content=_pcm_to_wav(pcm_bytes, sr), media_type="audio/wav",
                    headers={"X-Sample-Rate": str(sr)})


@app.post("/v1/audio/speech/stream")
async def speech_stream(req: TTSRequest):
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", req.text.strip()) if s.strip()]

    async def generate():
        for sentence in sentences:
            pcm_bytes, sr = await run_tts(sentence)
            if pcm_bytes:
                yield _pcm_to_wav(pcm_bytes, sr)

    return StreamingResponse(generate(), media_type="audio/wav")


# ─── LLM ─────────────────────────────────────────────────────────────────────

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
    if _phoneguy_reply is None:
        raise HTTPException(503, "LLM not initialized")
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_msg:
        raise HTTPException(400, "No user message found")
    reply = await asyncio.to_thread(_phoneguy_reply, user_msg)
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": req.model or "local",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ─── Dashboard endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    llm_info = "unavailable"
    llm_provider = ""
    llm_model = ""
    if _phoneguy_reply:
        llm_info = "ready"
        try:
            import ai_service
            llm_provider = getattr(ai_service, "PROVIDER", "")
            llm_model = getattr(ai_service, "MODEL", "")
        except Exception:
            pass
    return {
        "status": "ok",
        "version": "2.0.0",
        "connected_devices": len([d for d in _devices.values() if d.get("connected")]),
        "components": {
            "stt": "ready" if _stt_adapter else "unavailable",
            "tts": "ready" if (_tts_adapter or TTS_ENGINE in ("edge", "openai")) else "unavailable",
            "llm": llm_info,
        },
        "llm": {"provider": llm_provider, "model": llm_model},
        "features": {
            "tools_enabled": TOOLS_ENABLED,
            "tool_call_mode": TOOL_CALL_MODE,
            "memory_enabled": MEMORY_ENABLED,
            "tts_streaming": TTS_STREAMING,
            "tts_streaming_mode": TTS_STREAMING_MODE,
            "end_prompt": END_PROMPT_ENABLED,
            "tts_engine": TTS_ENGINE,
        },
        "config": {
            "whisper_model": STT_CONFIG["stt"]["model"],
            "whisper_device": STT_CONFIG["stt"]["device"],
            "tts_engine": TTS_CONFIG["tts"]["engine"],
            "tts_device": TTS_CONFIG["tts"]["device"],
            "rvc_enabled": TTS_CONFIG["tts"]["rvc_enabled"],
        },
        "ws_url": _ws_url(),
        "ota_url": f"http://{MY_HOST}:{HTTP_PORT}/ota",
    }


@app.get("/devices")
async def list_devices():
    now = time.time()
    return {"devices": [
        {"device_id": did, "client_id": info.get("client_id", ""), "connected": info.get("connected", False),
         "format": info.get("format", "pcm"), "last_seen_ago": int(now - info.get("last_seen", now))}
        for did, info in _devices.items()
    ], "total": len(_devices)}


@app.get("/config")
async def get_config():
    return {
        "MY_HOST": MY_HOST, "HTTP_PORT": HTTP_PORT, "BRIDGE_PORT": BRIDGE_PORT,
        "OTA_VERSION": OTA_VERSION, "OTA_TIMEZONE": OTA_TIMEZONE,
        "TTS_ENGINE": TTS_ENGINE, "AUDIO_FORMAT": AUDIO_FORMAT,
        "WHISPER_MODEL": STT_CONFIG["stt"]["model"], "WHISPER_DEVICE": STT_CONFIG["stt"]["device"],
        "WHISPER_LANGUAGE": STT_CONFIG["stt"]["language"],
        "TTS_DEVICE": TTS_CONFIG["tts"]["device"], "RVC_ENABLED": TTS_CONFIG["tts"]["rvc_enabled"],
        "VAD_SILENCE_FRAMES": VAD_SILENCE_FRAMES, "VAD_MIN_RECORD_MS": VAD_MIN_RECORD_MS,
        "VAD_MAX_RECORD_MS": VAD_MAX_RECORD_MS, "VAD_SPEECH_BOOST": VAD_SPEECH_BOOST,
        "TOOLS_ENABLED": TOOLS_ENABLED, "TOOL_CALL_MODE": TOOL_CALL_MODE,
        "MEMORY_ENABLED": MEMORY_ENABLED, "MEMORY_MAX_ENTRIES": MEMORY_MAX_ENTRIES,
        "TTS_STREAMING": TTS_STREAMING_MODE, "END_PROMPT_ENABLED": END_PROMPT_ENABLED,
        "RVC_STREAM_BLOCK_MS": RVC_STREAM_BLOCK_MS, "RVC_STREAM_OVERLAP": RVC_STREAM_OVERLAP,
        "RVC_STREAM_CROSSFADE": RVC_STREAM_CROSSFADE, "RVC_STREAM_VAD_DB": RVC_STREAM_VAD_DB,
        "END_PROMPT_TIMEOUT": END_PROMPT_TIMEOUT, "EXIT_COMMANDS": EXIT_COMMANDS,
        "LISTEN_MODE": DEFAULT_LISTEN_MODE, "EDGETTS_VOICE": EDGETTS_VOICE,
        "BRIDGE_TOKEN": BRIDGE_TOKEN[:8] + "...",
        "ALLOWED_DEVICES": list(ALLOWED_DEVICES) if ALLOWED_DEVICES else "(all allowed)",
        "TOOLS_REGISTERED": list(tool_registry.tools.keys()),
    }


@app.get("/logs")
async def get_logs():
    return {"logs": _log_buffer[-100:]}


@app.get("/logs/stream")
async def log_stream():
    import asyncio as _asyncio
    import queue as _queue

    q = _queue.Queue(maxsize=_SSE_MAX_QUEUE)
    _log_subscribers.append(q)

    async def event_generator():
        try:
            for line in _log_buffer[-50:]:
                yield f"data: {line}\n\n"
            while True:
                try:
                    msg = q.get_nowait()
                    yield f"data: {msg}\n\n"
                except _queue.Empty:
                    await _asyncio.sleep(0.5)
        except _asyncio.CancelledError:
            pass
        finally:
            try:
                _log_subscribers.remove(q)
            except ValueError:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/memory/{device_id}/clear")
async def clear_memory(device_id: str):
    memory.clear(device_id)
    return {"status": "ok", "message": f"Memory cleared for {device_id}"}


@app.get("/memory/{device_id}")
async def get_memory(device_id: str):
    return {"device_id": device_id, "context": memory.get_context(device_id)}


@app.get("/tools")
async def list_tools():
    return {"tools": tool_registry.get_tool_definitions(), "count": len(tool_registry.tools)}


@app.post("/server/restart")
async def server_restart():
    logger.info("🔄 Server restart requested via API")
    os.execv(sys.executable, [sys.executable] + sys.argv)


@app.get("/")
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return HTMLResponse(content=open(index_path).read())
    return {"service": "XiaoZhi Server", "version": "2.0.0",
            "dashboard": f"http://{MY_HOST}:{HTTP_PORT}/"}


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_token(ws) -> str:
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
    if secrets.compare_digest(token, BRIDGE_TOKEN):
        return True
    return _verify_hmac_token(token)


def _check_device(mac: str) -> bool:
    return not ALLOWED_DEVICES or mac.lower() in ALLOWED_DEVICES


def _get_timezone_offset(tz_name: str) -> int:
    try:
        import datetime, zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
        offset = now.utcoffset()
        if offset is None:
            return 0
        return int(offset.total_seconds() / 60)
    except Exception:
        try:
            if time.daylight and time.localtime().tm_isdst:
                return -time.altzone // 60
            return -time.timezone // 60
        except Exception:
            return 180


async def handle_connection(ws):
    remote = ws.remote_address

    # Auth
    token = _extract_token(ws)
    if not token:
        try:
            token = ws.request.headers.get("Authorization", "").replace("Bearer ", "")
        except Exception:
            pass

    if not _check_token(token):
        logger.warning(f"❌ Unauthorized from {remote}")
        await ws.close(1008, "Unauthorized")
        return

    logger.info(f"🌐 Path: {ws.request.path}")

    client_id = ""
    try:
        client_id = ws.request.headers.get("client-id", "") or ws.request.headers.get("Client-Id", "")
    except Exception:
        pass

    # State
    audio_buffer = bytearray()
    device_id = "unknown"
    session_id = secrets.token_hex(16)
    hello_done = False
    listening = False
    aborted = False
    processing = False
    tts_active = False
    pending_listen = False
    negotiated_fmt = "pcm"
    listen_mode = DEFAULT_LISTEN_MODE
    _dead_air = 0
    DEAD_AIR_FRAMES = 50
    opus_decoder = None
    opus_encoder = None
    vad = AdaptiveVAD()
    _frame_count = 0
    _vad_log_timer = time.time()
    _last_voice_time = time.time()
    _end_prompt_sent = False
    pipeline_task = None

    try:
        async for message in ws:

            # ── Check if background pipeline completed ──
            if pipeline_task and pipeline_task.done():
                try:
                    pipeline_task.result()
                except Exception as e:
                    logger.error(f"Pipeline error: {e}")
                pipeline_task = None
                processing = False
                tts_active = False
                vad.reset()
                _frame_count = 0
                _vad_log_timer = time.time()
                _dead_air = DEAD_AIR_FRAMES
                if pending_listen:
                    pending_listen = False
                    audio_buffer.clear()
                    listening = True
                    aborted = False
                    vad.reset()
                    _frame_count = 0
                    _vad_log_timer = time.time()
                    _dead_air = DEAD_AIR_FRAMES
                    logger.info(f"🎤 [{device_id}] Deferred listen started")

            # ── Binary frame ──
            if isinstance(message, bytes):
                if not hello_done or not listening or processing:
                    continue

                _frame_count += 1

                if negotiated_fmt == "opus" and opus_decoder:
                    try:
                        pcm = opus_decoder.decode(message, FRAME_SAMPLES)
                    except Exception:
                        pcm = b''
                elif negotiated_fmt == "opus":
                    pcm = b''
                else:
                    pcm = message

                if not pcm:
                    continue

                audio_buffer.extend(pcm)
                _last_voice_time = time.time()
                _end_prompt_sent = False

                if _dead_air > 0:
                    _dead_air -= 1
                    if _dead_air == 0:
                        logger.info(f"🔇 [{device_id}] Dead air done, VAD starting")
                    continue

                should_stop = vad.process_frame(pcm)

                now = time.time()
                if now - _vad_log_timer >= 2.0:
                    stats = vad.get_stats()
                    buf_s = len(audio_buffer) / (16000 * 2)
                    thr_str = f"{stats['threshold']}" if stats['threshold'] else "calibrating..."
                    logger.info(
                        f"📊 [{device_id}] VAD: rms={stats['rms']:.0f} thr={thr_str} "
                        f"silence={stats['silence']}/{VAD_SILENCE_FRAMES} "
                        f"speech={stats['speech']} buf={buf_s:.1f}s"
                    )
                    _vad_log_timer = now

                if should_stop:
                    stats = vad.get_stats()
                    buf_s = len(audio_buffer) / (16000 * 2)

                    if vad.frame_count < vad.min_frames:
                        vad.silence_count = 0
                        continue

                    logger.info(f"🛑 [{device_id}] End of speech! ({buf_s:.1f}s)")

                    listening = False
                    processing = True
                    tts_active = True

                    await ws.send(json.dumps({"type": "listen", "state": "stop", "session_id": session_id}))

                    # Run pipeline in background — read loop stays free for ping/abort
                    pipeline_task = asyncio.create_task(
                        _process_audio_pipeline(
                            ws, audio_buffer, device_id,
                            opus_encoder, lambda: aborted, session_id, listen_mode
                        )
                    )

                    if pending_listen:
                        pending_listen = False
                        audio_buffer.clear()
                        listening = True
                        aborted = False
                        vad.reset()
                        _frame_count = 0
                        _vad_log_timer = time.time()
                        _dead_air = DEAD_AIR_FRAMES
                        logger.info(f"🎤 [{device_id}] Deferred listen started")

                continue

            # ── JSON text frame ──
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                continue

            if not isinstance(data, dict):
                continue

            msg_type = data.get("type")
            state = data.get("state")

            # Hello
            if msg_type == "hello":
                ok, device_id, negotiated_fmt = await handle_hello(ws, data)
                if not ok:
                    await ws.close(1008, "Device not allowed")
                    return
                hello_done = True
                _devices[device_id] = {
                    "client_id": client_id, "connected": True,
                    "last_seen": time.time(), "format": negotiated_fmt,
                }
                memory.load(device_id)

                if negotiated_fmt == "opus" and OPUS_AVAILABLE:
                    try:
                        # Аплинк — 16к (формат девайса/ASR), даунлинк — DOWNLINK_SR
                        opus_decoder = opuslib.Decoder(16000, 1)
                        opus_encoder = opuslib.Encoder(DOWNLINK_SR, 1, opuslib.APPLICATION_AUDIO)
                    except Exception as e:
                        logger.error(f"Opus codec error: {e}")
                        negotiated_fmt = "pcm"
                continue

            if not hello_done:
                continue

            # listen/start
            if msg_type == "listen" and state == "start":
                listen_mode = data.get("mode", DEFAULT_LISTEN_MODE)
                if processing or tts_active:
                    pending_listen = True
                    logger.info(f"⏳ [{device_id}] listen/start while busy — queued (mode={listen_mode})")
                    await ws.send(json.dumps({"type": "listen", "state": "stop", "session_id": session_id}))
                    continue

                audio_buffer.clear()
                listening = True
                aborted = False
                processing = False
                vad.reset()
                _frame_count = 0
                _vad_log_timer = time.time()
                _dead_air = DEAD_AIR_FRAMES
                _last_voice_time = time.time()
                _end_prompt_sent = False
                logger.info(f"🎤 [{device_id}] Recording started (format={negotiated_fmt}, mode={listen_mode})")

            # listen/detect (device-side ASR result)
            elif msg_type == "listen" and state == "detect":
                detected_text = data.get("text", "")
                if detected_text:
                    logger.info(f"📝 [{device_id}] Detected text: '{detected_text}'")
                    listening = False
                    processing = True
                    tts_active = True
                    await ws.send(json.dumps({"type": "listen", "state": "stop", "session_id": session_id}))

                    memory.add(device_id, "user", detected_text)
                    await ws.send(json.dumps({"type": "stt", "text": detected_text, "session_id": session_id}))

                    # Send tts/start immediately (like official server)
                    await ws.send(json.dumps({"type": "tts", "state": "start", "session_id": session_id}))

                    if TOOLS_ENABLED and TOOL_CALL_MODE == "function_call":
                        detected_text_new, tool_result = await _try_function_call(detected_text, device_id)
                        if tool_result:
                            memory.add(device_id, "assistant", tool_result)

                    full_reply = None
                    try:
                        full_reply = await run_llm(detected_text, device_id)
                    except Exception as e:
                        logger.error(f"LLM error: {e}")

                    if not full_reply:
                        full_reply = random.choice(LLM_FALLBACK_REPLIES)

                    memory.add(device_id, "assistant", full_reply)

                    if _use_chunk_streaming():
                        await _tts_stream_rvc(ws, full_reply, opus_encoder, session_id, device_id)
                    elif TTS_STREAMING:
                        await _tts_stream_sentences(ws, full_reply, opus_encoder, session_id, device_id)
                    else:
                        tts_text = _convert_emotion_tags_for_tts(full_reply)
                        display_text = _clean_emotion_tags(full_reply)
                        if display_text:
                            await ws.send(json.dumps({"type": "tts", "state": "sentence_start", "text": display_text, "session_id": session_id}))
                        try:
                            pcm_bytes, sr = await run_tts(tts_text)
                            if pcm_bytes:
                                rate_ctrl = AudioRateController()
                                await rate_ctrl.start(ws, opus_encoder)
                                await rate_ctrl.enqueue(_pcm_to_wav(pcm_bytes, sr))
                                await rate_ctrl.stop()
                        except Exception as e:
                            logger.error(f"TTS error: {e}")

                    await ws.send(json.dumps({"type": "tts", "state": "stop", "session_id": session_id}))

                    processing = False
                    tts_active = False
                    vad.reset()
                    _dead_air = DEAD_AIR_FRAMES

            # listen/stop from device
            elif msg_type == "listen" and state == "stop":
                logger.info(f"🎤 [{device_id}] Device sent listen/stop")
                if listening and not processing and not tts_active:
                    listening = False
                    processing = True
                    tts_active = True
                    pipeline_task = asyncio.create_task(
                        _process_audio_pipeline(
                            ws, audio_buffer, device_id,
                            opus_encoder, lambda: aborted, session_id, listen_mode
                        )
                    )

            # ping
            elif msg_type == "ping":
                await ws.send(json.dumps({"type": "pong", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}))

            # abort
            elif msg_type == "abort":
                if tts_active:
                    logger.info(f"⏹️ [{device_id}] Abort during TTS — canceling pipeline")
                    if pipeline_task and not pipeline_task.done():
                        pipeline_task.cancel()
                    tts_active = False
                    processing = False
                    aborted = True
                    pending_listen = False
                    vad.reset()
                    continue
                logger.info(f"⏹️ [{device_id}] Abort")
                audio_buffer.clear()
                listening = False
                processing = False
                aborted = True
                pending_listen = False
                vad.reset()

            # server command
            elif msg_type == "server":
                action = data.get("action", "")
                if action == "restart":
                    logger.info(f"🔄 [{device_id}] Server restart requested")
                    await ws.send(json.dumps({"type": "server", "action": "restarting"}))
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                elif action == "update_config":
                    logger.info(f"🔄 [{device_id}] Config reload requested")

            else:
                logger.info(f"📨 [{device_id}] Msg: type={msg_type} state={state}")

        # ── End-of-conversation prompt ──
        if END_PROMPT_ENABLED and hello_done and not _end_prompt_sent:
            idle_time = time.time() - _last_voice_time
            if idle_time > END_PROMPT_TIMEOUT:
                _end_prompt_sent = True
                logger.info(f"👋 [{device_id}] End-of-conversation prompt (idle {idle_time:.0f}s)")
                try:
                    await ws.send(json.dumps({"type": "tts", "state": "start", "session_id": session_id}))
                    await ws.send(json.dumps({"type": "tts", "state": "sentence_start", "text": END_PROMPT_TEXT, "session_id": session_id}))
                    pcm_bytes, sr = await run_tts(END_PROMPT_TEXT)
                    if pcm_bytes:
                        rate_ctrl = AudioRateController()
                        await rate_ctrl.start(ws, opus_encoder)
                        await rate_ctrl.enqueue(_pcm_to_wav(pcm_bytes, sr))
                        await rate_ctrl.stop()
                    await ws.send(json.dumps({"type": "tts", "state": "stop", "session_id": session_id}))
                except Exception:
                    pass

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"📱 Disconnected: {remote} (code={e.code})")
    except Exception as e:
        logger.error(f"Connection error [{remote}]: {e}", exc_info=True)
    finally:
        if device_id and device_id in _devices:
            _devices[device_id]["connected"] = False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def run_websocket():
    logger.info(f"🌐 WebSocket: ws://{BRIDGE_HOST}:{BRIDGE_PORT}/xiaozhi/v1/")
    async with websockets.serve(
        handle_connection, BRIDGE_HOST, BRIDGE_PORT,
        max_size=10 * 1024 * 1024, ping_interval=20, ping_timeout=30,
    ):
        await asyncio.Future()


async def run_http():
    config = uvicorn.Config(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")
    await uvicorn.Server(config).serve()


async def main():
    logger.info("=" * 60)
    logger.info(" XiaoZhi Server v2.0 — всё в одном")
    logger.info(f" 🖥  Dashboard: http://{MY_HOST}:{HTTP_PORT}/")
    logger.info(f" OTA : http://{MY_HOST}:{HTTP_PORT}/ota")
    logger.info(f" WS : ws://{MY_HOST}:{BRIDGE_PORT}/xiaozhi/v1/")
    logger.info(f" Docs: http://{MY_HOST}:{HTTP_PORT}/docs")
    logger.info(f" Token: {BRIDGE_TOKEN[:8]}...")
    logger.info(f" TTS: {TTS_ENGINE}")
    logger.info(f" Tools: {len(tool_registry.tools)} registered ({TOOL_CALL_MODE})")
    logger.info(f" Memory: {'ON' if MEMORY_ENABLED else 'OFF'}")
    logger.info(f" Streaming: {TTS_STREAMING_MODE}")
    logger.info(f" Exit cmds: {EXIT_COMMANDS}")
    logger.info(f" Opus: {'available' if OPUS_AVAILABLE else 'NOT installed'}")
    logger.info(f" Timezone: {OTA_TIMEZONE}")
    logger.info("=" * 60)
    await asyncio.gather(run_http(), run_websocket())


if __name__ == "__main__":
    asyncio.run(main())
