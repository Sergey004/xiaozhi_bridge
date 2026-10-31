"""
xiaozhi_bridge.py — WebSocket сервер для XiaoZhi ESP32 устройств
Использует voice_api.py как бэкенд для STT/TTS, NVIDIA NIM для LLM.

Протокол XiaoZhi:
  Девайс стримит аудио, СЕРВЕР определяет конец речи (server-side VAD)
  и отправляет listen/stop. Девайс не шлёт listen/stop сам!

Аутентификация:
  - Токен в URL: ws://host:8765?token=YOUR_TOKEN
  - Опциональный вайтлист MAC-адресов девайсов

Конфиг LLM:
  Требуется NVIDIA_API_KEY или запущенный Ollama в .env / env.
  Если LLM недоступен — мост отвечает заранее записанной фразой через TTS.
"""

import array
import asyncio
import json
import logging
import math
import os
import re
import secrets
import struct
import time
from urllib.parse import parse_qs, urlparse

import httpx
import rich.logging
import websockets
import dotenv
dotenv.load_dotenv()

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
TTS_ENGINE    = os.getenv("TTS_ENGINE", "turbo").lower()  # turbo | dramabox

# Формат аудио: "auto" (подстраиваться под девайс), "opus", "pcm"
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", "auto").lower()

# VAD параметры
# threshold — RMS * 100 порог. "auto" = калибровка из первых фреймов тишины.
VAD_ENERGY_THRESHOLD = os.getenv("VAD_ENERGY_THRESHOLD", "auto").lower()
VAD_SILENCE_FRAMES   = int(os.getenv("VAD_SILENCE_FRAMES", "25"))      # ~1.5с тишины
VAD_MIN_RECORD_MS    = int(os.getenv("VAD_MIN_RECORD_MS", "600"))       # мин. длительность
VAD_MAX_RECORD_MS    = int(os.getenv("VAD_MAX_RECORD_MS", "10000"))     # макс. 10с (до таймаута девайса)
VAD_SPEECH_BOOST     = float(os.getenv("VAD_SPEECH_BOOST", "1.5"))      # множитель порога речи (низкий = чувствительнее)

# Токен доступа (обязателен).
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "")
if not BRIDGE_TOKEN:
    BRIDGE_TOKEN = secrets.token_urlsafe(32)
    logger.warning(f"⚠️  BRIDGE_TOKEN not set! Using random token: {BRIDGE_TOKEN}")
    logger.warning("    Set BRIDGE_TOKEN in .env to make it permanent.")

# Вайтлист MAC-адресов (опционально).
_raw_devices = os.getenv("ALLOWED_DEVICES", "")
ALLOWED_DEVICES: set = {
    d.strip().lower() for d in _raw_devices.split(",") if d.strip()
}

# Размер фрейма: 60ms @ 16kHz = 960 сэмплов = 1920 байт PCM16 (аплинк от девайса)
FRAME_SAMPLES = 960
FRAME_BYTES   = FRAME_SAMPLES * 2

# Даунлинк (TTS → девайс): полное качество выхода RVC.
# Opus поддерживает только 8000/12000/16000/24000/48000 — 40k выход RVC
# нормализуем до 48kHz (совпадает с dmix-слейвом станции — без ресемпла в maind).
DOWNLINK_SR         = int(os.getenv("DOWNLINK_SR", "48000"))
DOWNLINK_FRAME_MS   = 60
DOWNLINK_FRAME_SAMPLES = DOWNLINK_SR * DOWNLINK_FRAME_MS // 1000   # 2880 @ 48k
DOWNLINK_FRAME_BYTES   = DOWNLINK_FRAME_SAMPLES * 2

# Фразы-заглушки когда LLM недоступен
LLM_FALLBACK_REPLIES = [
    "У меня сейчас проблемы со связью. Попробуй позже.",
    "Сигнал пропал. Перезвони, пожалуйста.",
    "Кажется, мой мозг завис. Подожди немного.",
]

# ─── Emotion tags → TTS-дружественные звуки ────────────────────────────────────
# Chatterbox TTS понимает теги в скобках: [laugh], [sigh], [gasp], [cough]
# DramaBox НЕ понимает [laugh] — ему нужен сцено-режиссёрский формат
#
# Два режима конвертации в зависимости от TTS двигателя:
#   chatterbox: [laugh] → *laugh*  (TTS сам озвучивает)
#   dramabox:   [laugh] → She laughs, "Hahaha,"  (сценарный формат)

EMOTION_TAG_TO_TTS = {
    # Для Chatterbox: *звук* — TTS озвучивает
    "[laugh]":         "*laugh* ",
    "[laughs]":        "*laugh* ",
    "[chuckle]":       "*chuckle* ",
    "[chuckles]":      "*chuckle* ",
    "[gasp]":          "*gasp* ",
    "[gasps]":         "*gasp* ",
    "[sigh]":          "*sigh* ",
    "[sighs]":         "*sigh* ",
    "[cough]":         "*cough* ",
    "[coughs]":        "*cough* ",
    "[groan]":         "*groan* ",
    "[groans]":        "*groan* ",
    "[sniff]":         "*sniff* ",
    "[sniffs]":        "*sniff* ",
    "[clear throat]":  "*clears throat* ",
    "[shush]":         "*shush* ",
    "[yawn]":          "*yawn* ",
    "[yawns]":         "*yawn* ",
    "[hum]":           "*hum* ",
    "[hums]":          "*hum* ",
    "[stutter]":       "U-uh... ",
    "[stammers]":      "U-uh... ",
}

# Для DramaBox: тег → (stage_direction, spoken_sound)
EMOTION_TAG_TO_DRAMABOX = {
    "[laugh]":         ("She laughs,", '"Hahaha,"'),
    "[laughs]":        ("She laughs,", '"Hahaha,"'),
    "[chuckle]":       ("She chuckles,", '"Hehe,"'),
    "[chuckles]":      ("She chuckles,", '"Hehe,"'),
    "[gasp]":          ("She gasps,", ""),
    "[gasps]":         ("She gasps,", ""),
    "[sigh]":          ("She sighs deeply,", ""),
    "[sighs]":         ("She sighs,", ""),
    "[cough]":         ("She clears her throat,", ""),
    "[coughs]":        ("She coughs,", ""),
    "[groan]":         ("She groans,", '"Ugh,"'),
    "[groans]":        ("She groans,", '"Ugh,"'),
    "[sniff]":         ("She sniffs,", ""),
    "[sniffs]":        ("She sniffs,", ""),
    "[clear throat]":  ("She clears her throat,", ""),
    "[shush]":         ("She shushes,", '"Shh,"'),
    "[yawn]":          ("She yawns,", '"Mmmm,"'),
    "[yawns]":         ("She yawns,", ""),
    "[hum]":           ("She hums softly,", '"Mmmm,"'),
    "[hums]":          ("She hums,", '"Mmmm,"'),
    "[stutter]":       ("She stammers,", '"I... I..."'),
    "[stammers]":      ("She stammers,", '"I... I..."'),
}


def _convert_emotion_tags_for_tts(text: str) -> str:
    """
    Конвертирует теги эмоций в формат для Chatterbox TTS.
    [laugh] → *laugh*
    """
    import re
    def _replace_tag(match):
        tag = match.group(0).strip()
        return EMOTION_TAG_TO_TTS.get(tag, " ")

    return re.sub(r"\[[^\]]+\]", _replace_tag, text).strip()


def _convert_emotion_tags_for_dramabox(text: str) -> str:
    """
    Конвертирует теги эмоций в DramaBox сцено-режиссёрский формат.
    [laugh] Superstar? → She laughs, "Hahaha," Superstar?
    [sigh] We need... → She sighs deeply. We need...
    """
    import re
    def _replace_tag(match):
        tag = match.group(0).strip()
        entry = EMOTION_TAG_TO_DRAMABOX.get(tag)
        if entry is None:
            return " "
        direction, spoken = entry
        if spoken:
            return f"{direction} {spoken} "
        return f"{direction} "

    return re.sub(r"\[[^\]]+\]", _replace_tag, text).strip()


def _clean_emotion_tags(text: str) -> str:
    """Удаляет ВСЕ теги эмоций (для display на экране)."""
    import re
    return re.sub(r"\[[^\]]+\]", "", text).strip()


# ─── Emotions для XiaoZhi экрана ─────────────────────────────────────────────
# Сервер отправляет {"type": "llm", "emotion": "...", "text": "😀"}
# Девайс показывает соответствующее выражение на экране.
# Полный список из firmware (21 эмоция):
XIAOZHI_EMOTIONS = {
    "neutral":     "😶",
    "happy":       "🙂",
    "laughing":    "😆",
    "funny":       "😂",
    "sad":         "😔",
    "angry":       "😠",
    "crying":      "😭",
    "loving":      "😍",
    "embarrassed": "😳",
    "surprised":   "😯",
    "shocked":     "😱",
    "thinking":    "🤔",
    "winking":     "😉",
    "cool":        "😎",
    "relaxed":     "😌",
    "delicious":   "🤤",
    "kissy":       "😘",
    "confident":   "😏",
    "sleepy":      "😴",
    "silly":       "😜",
    "confused":    "🙄",
}


def _detect_emotion(text: str) -> str:
    """
    Определяет эмоцию из текста LLM-ответа.
    Ищет совпадения по ключевым словам / контексту.
    Returns: ключ из XIAOZHI_EMOTIONS.
    """
    text_lower = text.lower()

    # Сначала проверяем явные маркеры
    emotion_hints = {
        "laughing":  ["[laugh]"],
        "funny":    ["[laugh]", "haha", "lol"],
        "sad":      ["[sigh]", "[groan]", "sorry", "sad", "unfortunately"],
        "thinking": ["[clear throat]", "hmm", "let me think", "well", "um,"],
        "surprised": ["[gasp]", "oh!", "wow", "really?"],
        "happy":    ["[chuckle]", "great", "awesome", "love", "happy"],
        "embarrassed": ["[sniff]", "[cough]", "oops"],
        "angry":    ["angry", "mad", "furious"],
        "loving":   ["love you", "my dear", "superstar"],
        "cool":     ["cool", "rock star", "awesome"],
        "confident": ["of course", "absolutely", "i'm sure", "i know"],
        "silly":    ["silly", "crazy", "ridiculous"],
        "confused": ["[shush]", "what do you mean", "i don't understand"],
        "relaxed":  ["relax", "calm down", "don't worry"],
        "sleepy":   ["tired", "sleepy", "yawn"],
    }

    for emotion, keywords in emotion_hints.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return emotion

    return "neutral"


# ─── Opus ────────────────────────────────────────────────────────────────────
OPUS_AVAILABLE = False
try:
    import opuslib
    _test_enc = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
    _test_dec = opuslib.Decoder(16000, 1)
    del _test_enc, _test_dec
    OPUS_AVAILABLE = True
    logger.info("✅ opuslib available — Opus mode supported")
except Exception as e:
    OPUS_AVAILABLE = False
    logger.warning(f"⚠️ opuslib NOT available: {e}")
    logger.warning("   Install with: pip install opuslib")


# ─── Server-Side VAD с автокалибровкой ───────────────────────────────────────

class AdaptiveVAD:
    """
    Энергетический VAD с АВТОКАЛИБРОВКОЙ шумового порога.

    Алгоритм:
      1. Первые N фреймов использует для замера «уровня тишины» (noise floor)
      2. threshold = max(noise_floor * VAD_SPEECH_BOOST, min_abs_threshold)
      3. Если RMS > threshold → речь
      4. Если RMS < threshold VAD_SILENCE_FRAMES подряд → конец речи
    """
    def __init__(self):
        self.threshold = None
        self.silence_frames_limit = VAD_SILENCE_FRAMES
        self.min_frames = max(VAD_MIN_RECORD_MS // 60, 3)
        self.max_frames = max(VAD_MAX_RECORD_MS // 60, VAD_SILENCE_FRAMES + 5)
        self.speech_boost = VAD_SPEECH_BOOST

        self.frame_count = 0
        self.silence_count = 0
        self.speech_started = False

        # Для калибровки: первые фреймы до начала речи
        self._calibration_values = []
        self._calibration_frames = 25  # ~1.5с для калибровки (берём хвост — тише)
        self._calibration_warmup = 10  # пропускаем первые 10 фреймов (ещё шум от динамика)
        self._calibrated = False

        # Для статов
        self.last_rms = 0

    def reset(self):
        """Сброс состояния (новая запись). Threshold сохраняется."""
        self.frame_count = 0
        self.silence_count = 0
        self.speech_started = False
        self._calibration_values.clear()
        self._calibrated = False

    def _compute_rms(self, pcm_data: bytes) -> int:
        """RMS * 100 для PCM int16."""
        if len(pcm_data) < 2:
            return 0
        samples = array.array('h', pcm_data[:len(pcm_data) & ~1])
        if not samples:
            return 0
        sum_sq = sum(s * s for s in samples)
        return int(math.sqrt(sum_sq / len(samples)) * 100)

    def _calibrate(self):
        """Вычисляет порог из калибровочных данных."""
        # Пропускаем warmup-фреймы (начало калибровки — ещё шум от динамика)
        # Используем только хвост калибровки — там тише
        vals = self._calibration_values[self._calibration_warmup:]

        if len(vals) < 3:
            # Недостаточно данных после warmup — берём всё
            vals = self._calibration_values

        # Фильтруем громкие кадры (остаточный шум от динамика)
        quiet = [v for v in vals if v < 150000]

        if len(quiet) < 3:
            if len(vals) >= 3:
                sorted_vals = sorted(vals)
                median = sorted_vals[0]
                self.threshold = max(int(median * self.speech_boost), 15000)
                logger.warning(
                    f"VAD: all calibration frames loud (min={median:.0f}), "
                    f"threshold={self.threshold}"
                )
            else:
                self.threshold = 15000
                logger.warning("VAD: insufficient calibration data, using default threshold 15000")
        else:
            sorted_quiet = sorted(quiet)
            median = sorted_quiet[len(sorted_quiet) // 2]
            self.threshold = max(int(median * self.speech_boost), 8000)
            logger.info(
                f"VAD calibrated: noise_median={median:.0f} "
                f"(warmup={self._calibration_warmup}f skipped, "
                f"{len(vals) - len(quiet)} loud filtered, "
                f"{len(quiet)}/{len(self._calibration_values)} total) "
                f"threshold={self.threshold} (x{self.speech_boost})"
            )
        self._calibrated = True

    def process_frame(self, pcm_data: bytes) -> bool:
        """
        Обрабатывает PCM фрейм. Returns True если нужно остановить запись.
        """
        self.frame_count += 1
        rms = self._compute_rms(pcm_data)
        self.last_rms = rms

        # ── Фаза калибровки ──
        if not self._calibrated:
            self._calibration_values.append(rms)
            if len(self._calibration_values) >= self._calibration_frames:
                self._calibrate()
            return False  # Во время калибровки не останавливаем

        # ── Определяем речь/тишину ──
        is_speech = rms > self.threshold

        if is_speech:
            self.silence_count = 0
            self.speech_started = True
        elif self.speech_started:
            self.silence_count += 1

        # Safety: максимальная длительность
        if self.frame_count >= self.max_frames:
            return True

        # Конец речи
        if self.speech_started and self.silence_count >= self.silence_frames_limit:
            return True

        return False

    def get_stats(self) -> dict:
        return {
            "rms": self.last_rms,
            "threshold": self.threshold,
            "calibrated": self._calibrated,
            "silence": self.silence_count,
            "speech": self.speech_started,
            "frames": self.frame_count,
        }


# ─── Аутентификация ───────────────────────────────────────────────────────────

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
    return secrets.compare_digest(token, BRIDGE_TOKEN)


def _check_device(mac: str) -> bool:
    if not ALLOWED_DEVICES:
        return True
    return mac.lower() in ALLOWED_DEVICES


# ─── STT / TTS / LLM ─────────────────────────────────────────────────────────

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


async def stt(pcm_bytes: bytes) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{VOICE_API_URL}/v1/audio/transcriptions",
            files={"file": ("audio.wav", _pcm_to_wav(pcm_bytes, 16000), "audio/wav")},
            data={"language": "ru"},
        )
        r.raise_for_status()
        return r.json().get("text", "")


async def tts(text: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{VOICE_API_URL}/tts",
            json={"text": text},
        )
        r.raise_for_status()
        return r.content


async def llm_query(text: str) -> str | None:
    """
    Вызывает LLM. Возвращает текст ответа или None если LLM недоступен.
    НЕ бросает исключений.
    """
    try:
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
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            logger.warning(f"⚠️ LLM not available (503) — using fallback reply")
        else:
            logger.error(f"LLM HTTP error: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


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


def _parse_wav(wav_bytes: bytes) -> tuple:
    """
    Парсит WAV: возвращает (pcm_data, sample_rate, num_channels).
    Если не WAV — возвращает (raw_bytes, 0, 1).
    """
    if len(wav_bytes) < 44 or wav_bytes[:4] != b'RIFF':
        return wav_bytes, 0, 1

    try:
        import io, wave
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, 'rb') as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        return raw, sr, n_ch
    except Exception:
        pass

    # Fallback: ручной парсинг
    pos = 12
    while pos < len(wav_bytes) - 8:
        chunk_id = wav_bytes[pos:pos + 4]
        chunk_size = struct.unpack_from('<I', wav_bytes, pos + 4)[0]
        if chunk_id == b'data':
            pcm = wav_bytes[pos + 8:pos + 8 + chunk_size]
            # Sample rate в байтах 24-27
            sr = struct.unpack_from('<I', wav_bytes, 24)[0]
            return pcm, sr, 1
        pos += 8 + chunk_size
        if chunk_size % 2 != 0:
            pos += 1
    return wav_bytes[44:], 0, 1


def _resample_pcm(pcm_bytes: bytes, src_sr: int, dst_sr: int) -> bytes:
    """
    Ресемплит PCM int16 из src_sr в dst_sr.
    Использует scipy.signal.resample_poly — полифазный FIR фильтр с анти-алиасингом.
    Стандартный метод для аудио, сохраняет тон и тембр без артефактов.
    """
    if src_sr == dst_sr or src_sr == 0:
        return pcm_bytes

    import numpy as np
    from fractions import Fraction
    from scipy import signal as scipy_signal

    samples = np.frombuffer(pcm_bytes[:len(pcm_bytes) & ~1], dtype=np.int16).astype(np.float64)
    if len(samples) == 0:
        return pcm_bytes

    # Точная рациональная дробь для соотношения частот
    frac = Fraction(dst_sr, src_sr)
    up = frac.numerator
    down = frac.denominator

    # resample_poly: сначала upsampling (вставка нулей), потом FIR фильтр,
    # потом downsampling (прореживание). Анти-алиасинг встроен.
    # window='kaiser' с beta=14.76 — максимальное подавление алиасинга
    resampled = scipy_signal.resample_poly(
        samples, up, down,
        window=('kaiser', 14.76),
    )
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def _stereo_to_mono(pcm_bytes: bytes) -> bytes:
    """Конвертирует стерео PCM16 в моно (усреднение каналов)."""
    if len(pcm_bytes) < 4:
        return pcm_bytes
    import array as arr_module
    samples = arr_module.array('h', pcm_bytes[:len(pcm_bytes) & ~1])
    n = len(samples) // 2
    mono = arr_module.array('h', [0] * n)
    for i in range(n):
        mono[i] = (samples[i * 2] + samples[i * 2 + 1]) // 2
    return mono.tobytes()


def _normalize_wav_to_16k(wav_bytes: bytes) -> bytes:
    """
    Принимает WAV от TTS (обычно 24000Hz Chatterbox / 40000Hz после RVC),
    нормализует до PCM16 @ DOWNLINK_SR моно для даунлинка девайсу.
    Returns: PCM16 bytes @ DOWNLINK_SR mono.
    """
    pcm_data, sr, n_ch = _parse_wav(wav_bytes)

    # Логируем один раз (при первом вызове) для диагностики
    if not hasattr(_normalize_wav_to_16k, '_logged_sr'):
        _normalize_wav_to_16k._logged_sr = True
        logger.info(f"🔊 TTS sample rate detected: {sr}Hz, channels={n_ch}")
        if sr != DOWNLINK_SR:
            logger.info(f"   ⚠️  Resampling {sr}Hz → {DOWNLINK_SR}Hz for device")

    # Стерео → моно
    if n_ch == 2:
        pcm_data = _stereo_to_mono(pcm_data)

    # Ресемплинг до частоты даунлинка (40k RVC → 48k, 24k Chatterbox → 48k)
    if sr != 0 and sr != DOWNLINK_SR:
        pcm_data = _resample_pcm(pcm_data, sr, DOWNLINK_SR)

    return pcm_data


async def _send_audio(ws, wav_bytes: bytes, opus_encoder=None):
    """
    Отправляет WAV аудио покадрово через WebSocket С РЕАЛЬНО-ВРЕМЕННЫМ PACING.
    Как в официальном XiaoZhi сервере:
      - Первые 5 пакетов сразу (pre-buffer ~300мс)
      - Остальные — 1 пакет каждые 60мс (AudioRateController)
    """
    # Нормализация до DOWNLINK_SR (48k для полного качества RVC)
    pcm_data = _normalize_wav_to_16k(wav_bytes)

    # Подготавливаем все фреймы (60ms @ DOWNLINK_SR)
    frames = []
    if opus_encoder:
        for i in range(0, len(pcm_data), DOWNLINK_FRAME_BYTES):
            chunk = pcm_data[i:i + DOWNLINK_FRAME_BYTES]
            if len(chunk) < DOWNLINK_FRAME_BYTES:
                chunk = chunk + b'\x00' * (DOWNLINK_FRAME_BYTES - len(chunk))
            try:
                frames.append(opus_encoder.encode(chunk, DOWNLINK_FRAME_SAMPLES))
            except Exception as e:
                logger.error(f"Opus encode error: {e}")
                break
    else:
        for i in range(0, len(pcm_data), DOWNLINK_FRAME_BYTES):
            chunk = pcm_data[i:i + DOWNLINK_FRAME_BYTES]
            if chunk:
                frames.append(chunk)

    if not frames:
        return

    total_frames = len(frames)
    pre_buffer = 5  # первых 5 фреймов сразу (~300мс)
    frame_ms = 60   # 60мс на фрейм

    logger.info(
        f"📤 Sending {total_frames} frames ({total_frames * frame_ms / 1000:.1f}s), "
        f"pre-buffer={pre_buffer}, pacing={frame_ms}ms"
    )

    # Pre-buffer: отправляем первые N фреймов сразу
    for i in range(min(pre_buffer, total_frames)):
        try:
            await ws.send(frames[i])
        except Exception:
            return

    # Pacing: остальные фреймы с интервалом 60мс (реальное время)
    start_time = time.monotonic()
    for i in range(pre_buffer, total_frames):
        # Рассчитываем сколько ждать до следующего фрейма
        target_time = start_time + (i * frame_ms / 1000)
        now = time.monotonic()
        wait = target_time - now
        if wait > 0:
            await asyncio.sleep(wait)

        try:
            await ws.send(frames[i])
        except Exception:
            return

    elapsed = time.monotonic() - start_time
    logger.info(f"📤 Audio sent in {elapsed:.1f}s (real-time)")


# ─── Hello handshake ──────────────────────────────────────────────────────────

async def handle_hello(ws, data: dict) -> tuple:
    device_id = (
        data.get("device_id") or data.get("mac") or "unknown"
    )
    if device_id == "unknown":
        try:
            header_mac = ws.request.headers.get("device-id", "")
            if header_mac:
                device_id = header_mac
                logger.info(f"   📌 device_id from header: {device_id}")
        except Exception:
            pass

    version      = data.get("version", 1)
    transport    = data.get("transport", "websocket")
    audio_params = data.get("audio_params", {})

    logger.info(f"👋 Hello: device_id={device_id} version={version} audio={audio_params}")

    try:
        h = dict(ws.request.headers)
        h.pop('sec-websocket-key', None)
        h.pop('sec-websocket-version', None)
        logger.info(f"📨 Headers: {h}")
    except Exception:
        pass

    # Whitelist
    if device_id != "unknown" and not _check_device(device_id):
        logger.warning(f"❌ Device {device_id} not allowed")
        await ws.send(json.dumps({"type": "hello", "error": "device_not_allowed"}))
        return False, device_id, "pcm"

    # Negotiation
    device_format = (audio_params.get("format") or "pcm").lower()

    if AUDIO_FORMAT == "auto":
        negotiated = device_format if device_format in ("opus", "pcm") else "pcm"
        if negotiated == "opus" and not OPUS_AVAILABLE:
            logger.warning("   ⚠️  Device wants opus but opuslib NOT installed!")
            logger.warning("   ⚠️  Install: pip install opuslib")
    else:
        negotiated = AUDIO_FORMAT
        if AUDIO_FORMAT == "opus" and not OPUS_AVAILABLE:
            negotiated = "pcm"

    logger.info(f"   🔊 Negotiated: {negotiated}")

    response = {
        "type": "hello",
        "version": version,
        "transport": "websocket",
        "session_id": secrets.token_hex(16),
        "audio_params": {
            "format": negotiated,
            "sample_rate": DOWNLINK_SR,
            "channels": 1,
            "frame_duration": DOWNLINK_FRAME_MS,
        },
        "features": {"vad": True, "mcp": False}
    }

    await ws.send(json.dumps(response))
    logger.info(f"✅ Hello OK: {device_id} | format={negotiated}")

    return True, device_id, negotiated


# ─── STT → LLM → TTS пайплайн ────────────────────────────────────────────────

async def _process_audio_pipeline(ws, audio_buffer: bytearray, device_id: str,
                                   opus_encoder, is_aborted):
    if not audio_buffer:
        return

    buf_seconds = len(audio_buffer) / (16000 * 2)
    logger.info(f"🎤 [{device_id}] Processing {len(audio_buffer)} bytes ({buf_seconds:.1f}s)")

    # ── STT ──
    try:
        text = await stt(bytes(audio_buffer))
        audio_buffer.clear()
        logger.info(f"📝 [{device_id}] STT: '{text}'")
    except Exception as e:
        logger.error(f"STT error: {e}", exc_info=True)
        audio_buffer.clear()
        return

    if not text.strip():
        logger.info(f"📝 [{device_id}] STT empty — no speech")
        return

    # Отправляем текст на девайс
    await ws.send(json.dumps({"type": "stt", "text": text}))

    # ── LLM (ДО tts/start!) ──
    full_reply = await llm_query(text)

    if not full_reply or not full_reply.strip():
        import random
        full_reply = random.choice(LLM_FALLBACK_REPLIES)
        logger.warning(f"⚠️ [{device_id}] LLM unavailable, using fallback: '{full_reply}'")

    logger.info(f"🤖 [{device_id}] Reply: '{full_reply}'")

    # ── Только СЕЙЧАС отправляем tts/start + эмоцию ──
    tts_active = True
    await ws.send(json.dumps({"type": "tts", "state": "start"}))

    # Отправляем эмоцию на экран девайса
    emotion = _detect_emotion(full_reply)
    emoji = XIAOZHI_EMOTIONS.get(emotion, "😶")
    await ws.send(json.dumps({
        "type": "llm",
        "emotion": emotion,
        "text": emoji,
    }))
    logger.info(f"😀 [{device_id}] Emotion: {emotion} {emoji}")

    # ── TTS: ОДИН вызов на весь текст ──
    # Конвертация тегов зависит от TTS двигателя:
    #   Chatterbox: [laugh] → *laugh*  (TTS озвучивает как звук)
    #   DramaBox:   [laugh] → She laughs, "Hahaha,"  (сцено-режиссёрский формат)
    if TTS_ENGINE == "dramabox":
        tts_text = _convert_emotion_tags_for_dramabox(full_reply)
        logger.info(f"🎭 [{device_id}] TTS text (DramaBox format): {tts_text[:80]}...")
    else:
        tts_text = _convert_emotion_tags_for_tts(full_reply)
    # Для экрана: убираем все теги
    display_text = _clean_emotion_tags(full_reply)

    logger.info(f"🗣️ [{device_id}] TTS: full text ({len(tts_text)} chars)")

    # Отправляем текст для отображения на экране
    if display_text:
        await ws.send(json.dumps({
            "type": "tts",
            "state": "sentence_start",
            "text": display_text,
        }))

    # Генерируем ОДИН WAV на весь ответ
    try:
        wav = await tts(tts_text)
        await _send_audio(ws, wav, opus_encoder=opus_encoder)
    except Exception as e:
        logger.error(f"TTS error: {e}")

    await ws.send(json.dumps({"type": "tts", "state": "stop"}))
    tts_active = False
    logger.info(f"✅ [{device_id}] Response complete")


# ─── Основной обработчик соединения ──────────────────────────────────────────

async def handle_connection(ws):
    remote = ws.remote_address

    # Auth
    token = _extract_token(ws)
    if not token:
        try:
            token = ws.request.headers.get("Authorization", "")
            token = token.replace("Bearer ", "")
        except Exception:
            pass

    if not _check_token(token):
        logger.warning(f"❌ Unauthorized from {remote}")
        await ws.close(1008, "Unauthorized")
        return

    logger.info(f"🌐 Path: {ws.request.path}")

    # Состояние
    audio_buffer    = bytearray()
    device_id       = "unknown"
    hello_done      = False
    listening       = False
    aborted         = False
    processing      = False
    tts_active      = False  # True пока сервер отправляет TTS аудио
    pending_listen  = False  # Девайс прислал listen/start пока TTS был активен
    negotiated_fmt  = "pcm"
    _dead_air       = 0  # кадров тишины после TTS (пропускаем шум динамика)
    DEAD_AIR_FRAMES = 50  # ~3с при 60мс/кадр — ждём пока отзвучит динамик
    opus_decoder    = None
    opus_encoder    = None
    vad             = AdaptiveVAD()

    _frame_count   = 0
    _log_timer     = time.time()
    _vad_log_timer = time.time()

    try:
        async for message in ws:

            # ═══════════════════════════════════════════════════════════
            # БИНАРНЫЙ ФРЕЙМ
            # ═══════════════════════════════════════════════════════════
            if isinstance(message, bytes):
                if not hello_done or not listening or processing:
                    continue

                _frame_count += 1

                # Декодируем
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

                # Пропускаем первые кадры после TTS (шум от динамика)
                if _dead_air > 0:
                    _dead_air -= 1
                    if _dead_air == 0:
                        logger.info(f"🔇 [{device_id}] Dead air done, VAD starting")
                    continue

                # VAD
                should_stop = vad.process_frame(pcm)

                now = time.time()
                if now - _vad_log_timer >= 2.0:
                    stats = vad.get_stats()
                    buf_s = len(audio_buffer) / (16000 * 2)
                    thr_str = f"{stats['threshold']}" if stats['threshold'] else "calibrating..."
                    logger.info(
                        f"📊 [{device_id}] VAD: rms={stats['rms']:.0f} "
                        f"thr={thr_str} "
                        f"silence={stats['silence']}/{VAD_SILENCE_FRAMES} "
                        f"speech={stats['speech']} "
                        f"buf={buf_s:.1f}s"
                    )
                    _vad_log_timer = now

                if should_stop:
                    stats = vad.get_stats()
                    buf_s = len(audio_buffer) / (16000 * 2)

                    if vad.frame_count < vad.min_frames:
                        logger.info(
                            f"🔇 [{device_id}] Too short ({buf_s:.1f}s), "
                            f"waiting for more speech"
                        )
                        # Сбрасываем счётчик тишины но продолжаем запись
                        vad.silence_count = 0
                        continue

                    logger.info(
                        f"🛑 [{device_id}] End of speech! "
                        f"({buf_s:.1f}s, {stats['silence']} silence frames, "
                        f"rms={stats['rms']:.0f})"
                    )

                    listening = False
                    processing = True

                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))

                    await _process_audio_pipeline(
                        ws, audio_buffer, device_id,
                        opus_encoder, lambda: aborted
                    )

                    processing = False
                    vad.reset()
                    _frame_count = 0
                    _vad_log_timer = time.time()
                    _dead_air = DEAD_AIR_FRAMES  # после ответа тоже ждём

                    # Если девайс просил слушать пока мы отвечали — запускаем
                    if pending_listen:
                        pending_listen = False
                        audio_buffer.clear()
                        listening = True
                        aborted = False
                        vad.reset()
                        _frame_count = 0
                        _vad_log_timer = time.time()
                        _dead_air = DEAD_AIR_FRAMES
                        logger.info(f"🎤 [{device_id}] Deferred listen started (was queued during TTS)")

                continue

            # ═══════════════════════════════════════════════════════════
            # ТЕКСТОВЫЙ ФРЕЙМ = JSON
            # ═══════════════════════════════════════════════════════════

            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                continue

            if not isinstance(data, dict):
                continue

            msg_type = data.get("type")
            state    = data.get("state")

            # Hello
            if msg_type == "hello":
                ok, device_id, negotiated_fmt = await handle_hello(ws, data)
                if not ok:
                    await ws.close(1008, "Device not allowed")
                    return
                hello_done = True

                if negotiated_fmt == "opus" and OPUS_AVAILABLE:
                    try:
                        # Аплинк декодируется на 16к (формат девайса/ASR),
                        # даунлинк энкодер — на DOWNLINK_SR (полное качество RVC)
                        opus_decoder = opuslib.Decoder(16000, 1)
                        opus_encoder = opuslib.Encoder(DOWNLINK_SR, 1, opuslib.APPLICATION_AUDIO)
                        logger.info(f"📦 [{device_id}] Opus codecs ready (uplink 16k, downlink {DOWNLINK_SR})")
                    except Exception as e:
                        logger.error(f"Opus codec error: {e}")
                        negotiated_fmt = "pcm"
                continue

            if not hello_done:
                continue

            # listen/start
            if msg_type == "listen" and state == "start":
                if processing or tts_active:
                    # НЕ обрываем TTS! Запоминаем что девайс хочет слушать
                    # и начнём слушать после завершения обработки
                    pending_listen = True
                    logger.info(
                        f"⏳ [{device_id}] listen/start while {'processing' if processing else 'TTS active'} — queued"
                    )
                    # Отправляем listen/stop чтобы девайс перестал слать аудио
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    continue

                audio_buffer.clear()
                listening = True
                aborted = False
                processing = False
                vad.reset()
                _frame_count = 0
                _vad_log_timer = time.time()
                _dead_air = DEAD_AIR_FRAMES
                logger.info(f"🎤 [{device_id}] Recording started (format={negotiated_fmt}), dead_air={DEAD_AIR_FRAMES}f")

            # listen/stop от девайса
            elif msg_type == "listen" and state == "stop":
                logger.info(f"🎤 [{device_id}] Device sent listen/stop")
                if listening and not processing and not tts_active:
                    listening = False
                    processing = True
                    await _process_audio_pipeline(
                        ws, audio_buffer, device_id,
                        opus_encoder, lambda: aborted
                    )
                    processing = False
                    vad.reset()
                    _frame_count = 0
                    # Если девайс просил слушать пока мы отвечали
                    if pending_listen:
                        pending_listen = False
                        audio_buffer.clear()
                        listening = True
                        vad.reset()
                        _frame_count = 0
                        _vad_log_timer = time.time()
                        _dead_air = DEAD_AIR_FRAMES
                        logger.info(f"🎤 [{device_id}] Deferred listen started")

            # ping
            elif msg_type == "ping":
                await ws.send(json.dumps({"type": "pong"}))

            # abort
            elif msg_type == "abort":
                if tts_active:
                    # Игнорируем abort во время TTS — не обрываем воспроизведение
                    logger.info(f"⏹️ [{device_id}] Abort ignored during TTS playback")
                    continue
                logger.info(f"⏹️ [{device_id}] Abort")
                audio_buffer.clear()
                listening = False
                processing = False
                aborted = True
                pending_listen = False
                vad.reset()

            else:
                logger.info(f"📨 [{device_id}] Msg: type={msg_type} state={state}")

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"📱 Disconnected: {remote} (code={e.code})")
    except Exception as e:
        logger.error(f"Connection error [{remote}]: {e}", exc_info=True)


# ─── Запуск ──────────────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 60)
    logger.info(f"🌐 XiaoZhi Bridge v6 (DramaBox+Chatterbox, audio pacing, adaptive VAD)")
    logger.info(f"   Listen:    ws://{BRIDGE_HOST}:{BRIDGE_PORT}/xiaozhi/v1")
    logger.info(f"   Token:     {BRIDGE_TOKEN[:8]}...")
    logger.info(f"   TTS:       {TTS_ENGINE}")
    logger.info(f"   Opus:      {'available' if OPUS_AVAILABLE else 'NOT installed'}")
    logger.info(f"   VAD:       auto-calibrating, silence={VAD_SILENCE_FRAMES}f, "
                f"boost={VAD_SPEECH_BOOST}x")
    logger.info(f"   Voice API: {VOICE_API_URL}")
    if not OPUS_AVAILABLE:
        logger.warning("   ⚠️  opuslib not installed! Install: pip install opuslib")
    logger.info("=" * 60)

    async with websockets.serve(
        handle_connection,
        BRIDGE_HOST,
        BRIDGE_PORT,
        process_request=None,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=30,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
