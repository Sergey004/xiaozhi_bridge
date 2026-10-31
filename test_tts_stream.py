"""
test_tts_stream.py — A/B качества и задержки RVC: батч против стриминга
на ОДНОЙ И ТОЙ ЖЕ TTS-генерации.

ВАЖНО: Chatterbox Turbo понимает ТОЛЬКО английский — русский текст даёт
кашу ещё на этапе TTS, независимо от RVC. Поэтому текст по умолчанию
английский.

Как работает:
  1. Одна генерация TTS (synthesize_stream отдаёт всё предложение одним чанком).
  2. Путь "как было": полный rvc_infer() на всю фразу разом.
  3. Путь "как стало": чанк → 16кГц целиком → кусочки ~100мс → RVCStreamer.push().

Источник один, поэтому длительности обоих WAV должны совпасть — если нет,
стриминг теряет/дублирует аудио. На слух отличаться должна только задержка.

Запуск:
  python test_tts_stream.py "Hey buddy! This is a streaming test."
"""

import argparse
import asyncio
import logging
import math
import os
import time
import wave

import numpy as np
import rich.logging
from scipy.signal import resample_poly

import dotenv
dotenv.load_dotenv()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("test_tts_stream")

SLICE = 1600  # порция подачи в RVCStreamer ~100мс @16кГц


def save_wav_f32(arr: np.ndarray, sr: int, path: str):
    data = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data)
    logger.info(f"💾 Saved: {path} ({sr}Hz, {len(data) / 2 / sr:.2f}s)")


def resample_f32(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    g = math.gcd(src, dst)
    return resample_poly(audio, dst // g, src // g).astype(np.float32)


def make_streamer(args, vad_db, input_sr: int = 16000):
    from rvc_py.realtime import RVCStreamer

    return RVCStreamer(
        rvc_model_path=os.getenv("RVC_MODEL_PATH", "models/freddy.pth"),
        device=os.getenv("TTS_DEVICE", "cuda"),
        f0_up_key=int(os.getenv("RVC_PITCH_SHIFT", "0")),
        f0_method=os.getenv("RVC_F0_METHOD", "rmvpe"),
        index_path=os.getenv("RVC_INDEX_PATH") or None,
        index_rate=float(os.getenv("RVC_INDEX_RATE", "0.5")),
        input_sr=input_sr,
        output_sr=16000,
        block_size=max(1, int(16000 * args.block_ms / 1000)),
        overlap=args.overlap,
        crossfade=args.crossfade,
        vad_threshold_db=vad_db,
        extra_frame=args.extra_frame,
    )


async def warmup(adapter, args, vad_db):
    """Модели/кернелы уходят в кеш ДО замеров, RMVPE грузится заранее."""
    await adapter.check_health()
    streamer = make_streamer(args, vad_db)
    t = np.arange(SLICE, dtype=np.float32) / 16000.0
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    await asyncio.to_thread(streamer.push, tone)
    await asyncio.to_thread(streamer.flush)
    streamer.reset()
    logger.info("🔥 TTS + RVCStreamer прогреты")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text", nargs="?",
        default="Hey there! This is a voice streaming test. Tell me if you can hear me clearly.")
    parser.add_argument("--voice", default=os.getenv("AUDIO_PROMPT_PATH") or None)
    parser.add_argument("--engine", default=os.getenv("TTS_ENGINE", "turbo"))
    parser.add_argument("--block-ms", type=int, default=int(os.getenv("RVC_STREAM_BLOCK_MS", "300")))
    parser.add_argument("--overlap", type=int, default=int(os.getenv("RVC_STREAM_OVERLAP", "1024")))
    parser.add_argument("--crossfade", type=int, default=int(os.getenv("RVC_STREAM_CROSSFADE", "512")))
    parser.add_argument("--vad-db", default=os.getenv("RVC_STREAM_VAD_DB", "-45"))
    parser.add_argument("--extra-frame", type=int,
                        default=int(os.getenv("RVC_STREAM_EXTRA_FRAME", "0")))
    args = parser.parse_args()

    vad_db = None if str(args.vad_db).strip().lower() in ("", "none", "off") else float(args.vad_db)

    from tts_adapter import create_tts_adapter

    config = {
        "tts": {
            "engine": args.engine,
            "device": os.getenv("TTS_DEVICE", "cuda"),
            "audio_prompt_path": args.voice,
            "rvc_enabled": os.getenv("RVC_ENABLED", "false").lower() == "true",
            "rvc_model_path": os.getenv("RVC_MODEL_PATH"),
            "rvc_index_path": os.getenv("RVC_INDEX_PATH"),
            "rvc_f0_method": os.getenv("RVC_F0_METHOD", "rmvpe"),
            "rvc_pitch_shift": int(os.getenv("RVC_PITCH_SHIFT", "0")),
            "rvc_index_rate": float(os.getenv("RVC_INDEX_RATE", "0.5")),
        }
    }

    if not config["tts"]["rvc_enabled"] or not config["tts"]["rvc_model_path"]:
        logger.error("❌ RVC выключен — тест не имеет смысла. "
                     "Проверь RVC_ENABLED и RVC_MODEL_PATH в .env")
        return

    adapter = create_tts_adapter(config, logger)
    logger.info("")
    logger.info(f"🗣️  Текст: \"{args.text}\"")
    await warmup(adapter, args, vad_db)
    logger.info("")

    # ── 1. Одна TTS-генерация для обоих путей ────────────────────────────────
    t0 = time.perf_counter()
    chunks = []
    src_sr = None
    async for chunk_np, chunk_sr in adapter.synthesize_stream(args.text):
        chunks.append(chunk_np)
        src_sr = chunk_sr
    tts_time = time.perf_counter() - t0

    if not chunks:
        logger.error("❌ TTS вернул пустой результат")
        return
    source = np.concatenate(chunks)
    logger.info(f"🎙️  TTS: {tts_time:.2f}s → {len(source) / src_sr:.2f}s @ {src_sr}Hz "
                f"({'одним чанком — нативного стриминга у движка нет' if len(chunks) == 1 else f'{len(chunks)} чанков'})")
    logger.info("")

    # ── 2. БАТЧ RVC (как было): вся фраза разом ─────────────────────────────
    from rvc_py.rvc_infer import rvc_infer

    logger.info("── Путь 1/2: БАТЧ rvc_infer (как было) ──")
    t0 = time.perf_counter()
    rvc_out = await asyncio.to_thread(
        rvc_infer,
        source.astype(np.float32), src_sr,
        config["tts"]["rvc_model_path"],
        device=config["tts"]["device"],
        index_path=config["tts"]["rvc_index_path"],
        index_rate=config["tts"]["rvc_index_rate"],
        pitch_shift=config["tts"]["rvc_pitch_shift"],
        f0_method=config["tts"]["rvc_f0_method"],
    )
    batch_rvc_time = time.perf_counter() - t0

    val1, val2 = rvc_out
    if isinstance(val1, (int, np.integer)) and val1 > 1000:
        batch_sr, batch_audio = int(val1), val2
    elif isinstance(val2, (int, np.integer)) and val2 > 1000:
        batch_audio, batch_sr = val1, int(val2)
    else:
        batch_audio, batch_sr = val1, val2
    batch_audio = np.asarray(batch_audio).squeeze().astype(np.float32)
    dur_batch = len(batch_audio) / batch_sr
    save_wav_f32(batch_audio, batch_sr, "test_rvc_batch.wav")
    logger.info(f"⏱  БАТЧ RVC: {batch_rvc_time:.2f}s — столько ждали первого звука после TTS в старом конвейере")
    logger.info("")

    # ── 3. СТРИМ RVC (как стало): кусочки через RVCStreamer ─────────────────
    logger.info("── Путь 2/2: СТРИМ RVCStreamer (как стало) ──")
    source16 = resample_f32(source, src_sr, 16000)
    streamer = make_streamer(args, vad_db, input_sr=16000)
    logger.info(f"🎬 block={args.block_ms}ms overlap={args.overlap} crossfade={args.crossfade} "
                f"vad={vad_db}dB extra_frame={args.extra_frame}")

    t0 = time.perf_counter()
    first_audio = None
    pieces = []
    blocks = 0
    for off in range(0, len(source16), SLICE):
        converted = await asyncio.to_thread(streamer.push, source16[off:off + SLICE])
        if converted is not None and len(converted) > 0:
            if first_audio is None:
                first_audio = time.perf_counter() - t0
                logger.info(f"🔊 ПЕРВЫЙ КОНВЕРТИРОВАННЫЙ БЛОК готов через {first_audio:.2f}s")
            pieces.append(converted)
            blocks += 1
    tail = await asyncio.to_thread(streamer.flush)
    if tail is not None and len(tail) > 0:
        pieces.append(tail)
        blocks += 1
    streamer.reset()
    stream_time = time.perf_counter() - t0

    if not pieces:
        logger.error("❌ Стриминг: пустой результат")
        return
    full = np.concatenate(pieces)
    dur_stream = len(full) / 16000
    save_wav_f32(full, 16000, "test_rvc_stream.wav")
    logger.info(f"⏱  СТРИМ: первый звук {first_audio:.2f}s, полный {stream_time:.2f}s, {blocks} блоков")
    logger.info("")

    # ── Итог ─────────────────────────────────────────────────────────────────
    dur_delta_ms = abs(dur_batch - dur_stream) * 1000
    logger.info("═" * 60)
    logger.info(f"📊 Первый звук: {first_audio:.2f}s против {batch_rvc_time:.2f}s у батча "
                f"→ выигрыш {batch_rvc_time - first_audio:+.2f}s")
    logger.info(f"📊 Длительности: батч {dur_batch:.2f}s vs стрим {dur_stream:.2f}s "
                f"(Δ {dur_delta_ms:.0f} мс — должно быть мало, иначе стриминг теряет аудио)")
    logger.info("   Послушай оба WAV: разборчивость должна быть одинаковой,")
    logger.info("   отличаться могут только стыки блоков (block/overlap/crossfade).")


if __name__ == "__main__":
    asyncio.run(main())
