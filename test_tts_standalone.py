"""
test_tts_standalone.py — Прямой вызов TTS+RVC в обход бриджа и voice_api.
Сохраняет результат в WAV с РЕАЛЬНЫМ sample_rate, плюс печатает ТИПЫ
значений из rvc_infer() — чтобы поймать баг с numpy.int64 vs python int.

Запуск:
  python test_tts_standalone.py "Привет, это тест."
  python test_tts_standalone.py "Привет!" --no-rvc
  python test_tts_standalone.py "Привет!" --voice samples/freddy.wav
"""

import argparse
import asyncio
import logging
import wave

import rich.logging

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("test_tts")


def pcm_to_wav_file(pcm_bytes: bytes, sample_rate: int, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    logger.info(f"💾 Saved: {path} ({sample_rate}Hz, {len(pcm_bytes)} bytes)")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Текст для синтеза")
    parser.add_argument("--voice",   default=None, help="Путь к voice sample (audio_prompt_path)")
    parser.add_argument("--no-rvc",  action="store_true", help="Отключить RVC, проверить чистый Chatterbox")
    parser.add_argument("--out",     default="test_output.wav", help="Куда сохранить")
    parser.add_argument("--engine",  default="turbo", help="turbo / multilingual / base / dramabox")
    parser.add_argument("--rvc-model", default="models/freddy.pth")
    parser.add_argument("--rvc-index", default="models/freddy.index")
    args = parser.parse_args()

    # ─── Реальная фабрика из твоего tts_adapter.py ────────────────────────
    from tts_adapter import create_tts_adapter

    config = {
        "tts": {
            "engine":            args.engine,
            "device":            "cuda",
            "audio_prompt_path": args.voice,
            "rvc_enabled":       not args.no_rvc,
            "rvc_model_path":    args.rvc_model,
            "rvc_index_path":    args.rvc_index,
            "rvc_f0_method":     "rmvpe",
            "rvc_pitch_shift":   0,
            "rvc_index_rate":    0.5,
        }
    }

    adapter = create_tts_adapter(config, logger)

    # ─── Патчим rvc_infer чтобы увидеть СЫРЫЕ типы до защитной логики ─────
    # Покажет, действительно ли sr приходит как numpy.int64 а не python int
    try:
        import tts_adapter as _ta_module
        _orig_rvc_infer = _ta_module.rvc_infer

        if _orig_rvc_infer is not None and config["tts"]["rvc_enabled"]:
            def _debug_rvc_infer(*a, **kw):
                result = _orig_rvc_infer(*a, **kw)
                if isinstance(result, tuple) and len(result) == 2:
                    v1, v2 = result
                    logger.info(
                        f"🔬 RAW rvc_infer() returned: "
                        f"type(val1)={type(v1).__name__} "
                        f"type(val2)={type(v2).__name__}"
                    )
                    for i, v in enumerate((v1, v2), 1):
                        if isinstance(v, int):
                            logger.info(f"   val{i}={v} → isinstance(int) = TRUE ✅")
                        elif hasattr(v, "dtype"):
                            logger.info(
                                f"   val{i} is array/numpy scalar, dtype={v.dtype} "
                                f"→ isinstance(int) = FALSE ⚠️ (баг, если это sr)"
                            )
                        else:
                            logger.info(f"   val{i}={v!r} ({type(v).__name__})")
                return result
            _ta_module.rvc_infer = _debug_rvc_infer
    except Exception as e:
        logger.warning(f"Couldn't patch rvc_infer for debugging: {e}")

    logger.info("🔥 Warming up...")
    await adapter.check_health()

    logger.info(f"🗣️  Synthesizing: \"{args.text}\"")
    pcm_bytes, sr = await adapter.synthesize(args.text)

    # ─── ГЛАВНАЯ ДИАГНОСТИКА ──────────────────────────────────────────────
    logger.info(f"📊 Final result: {len(pcm_bytes)} bytes @ {sr}Hz (type={type(sr).__name__})")
    if sr > 0:
        duration_sec = len(pcm_bytes) / 2 / sr
        logger.info(f"📏 Duration: {duration_sec:.2f}s")

    if config["tts"]["rvc_enabled"] and sr not in (40000, 48000):
        logger.warning(
            f"⚠️  RVC включен, но финальный sr={sr} — ожидался 40000/48000! "
            f"Смотри лог 🔬 RAW rvc_infer() выше — если там dtype вместо int, "
            f"вот и причина."
        )

    pcm_to_wav_file(pcm_bytes, sr, args.out)

    logger.info("")
    logger.info(f"✅ Готово! Открой {args.out} и послушай.")
    logger.info(f"   Звук нормальный здесь → баг в bridge (resample/opus).")
    logger.info(f"   Звук уже высокий здесь → баг в tts_adapter.py (см. 🔬 выше).")


if __name__ == "__main__":
    asyncio.run(main())
