import asyncio
import logging
import torch
import numpy as np
import sys
import os

# Настройка путей
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Проверка доступности chatterbox
_chatterbox_error = None
try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    # Дополнительная проверка - классы могут импортироваться но быть None
    if ChatterboxTTS is None or ChatterboxTurboTTS is None:
        raise ImportError("chatterbox-tts classes are None")
    CHATTERBOX_AVAILABLE = True
except Exception as e:
    CHATTERBOX_AVAILABLE = False
    ChatterboxTTS = None
    ChatterboxMultilingualTTS = None
    ChatterboxTurboTTS = None
    _chatterbox_error = str(e)

# Проверка доступности RVC
# ВАЖНО: импортируем функцию из модуля, а не сам модуль
try:
    from rvc_py.rvc_infer import rvc_infer
except Exception:
    rvc_infer = None

logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("numba.core.byteflow").setLevel(logging.ERROR)


def create_tts_adapter(config: dict, logger: logging.Logger):
    """
    Фабрика: создаёт TTS адаптер в зависимости от engine.
    Поддерживает: 'turbo', 'base', 'multilingual', 'dramabox'
    """
    tts_cfg = config.get("tts", {})
    engine = tts_cfg.get("engine", "turbo").lower()

    if engine == "dramabox":
        from dramabox_adapter import DramaBoxAdapter
        logger.info("🎭 TTS engine: DramaBox (expressive, 24GB VRAM)")
        return DramaBoxAdapter(config, logger)
    else:
        logger.info(f"🔊 TTS engine: Chatterbox-{engine}")
        return ChatterboxAdapter(config, logger)


class ChatterboxAdapter:
    """Оригинальный Chatterbox TTS адаптер (turbo/base/multilingual)."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger.getChild("TTS")

        tts_cfg = config.get("tts", {})
        self.engine = tts_cfg.get("engine", "turbo")
        self.device = tts_cfg.get("device", "cuda")
        self.language_id = tts_cfg.get("language_id")
        self.audio_prompt_path = tts_cfg.get("audio_prompt_path")
        self.cfg_weight = "0.3"

        self._speak_lock = asyncio.Lock()
        self._model = None

        # RVC Settings
        self.rvc_enabled = tts_cfg.get("rvc_enabled", False)
        self.rvc_model_path = tts_cfg.get("rvc_model_path")
        self.rvc_index_path = tts_cfg.get("rvc_index_path")
        self.rvc_f0_method = tts_cfg.get("rvc_f0_method", "rmvpe")
        self.rvc_pitch_shift = tts_cfg.get("rvc_pitch_shift", 0)
        self.rvc_index_rate = tts_cfg.get("rvc_index_rate", 0.5)

        # Отключаем RVC если модуль не импортирован
        if rvc_infer is None:
            self.rvc_enabled = False
            self.logger.warning(
                "RVC is disabled because rvc_infer module is not available"
            )

    def _get_model(self):
        if self._model is not None:
            return self._model

        if not CHATTERBOX_AVAILABLE:
            raise ImportError(
                f"chatterbox-tts library is not installed or broken. "
                f"Install with: pip install chatterbox-tts. Error: {_chatterbox_error}"
            )

        self.logger.info(f"Initializing TTS Model: {self.engine} on {self.device}")
        try:
            if self.engine.lower() == "turbo":
                if ChatterboxTurboTTS is None:
                    raise ImportError("ChatterboxTurboTTS class is None")
                self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            elif self.engine.lower() == "multilingual":
                if ChatterboxMultilingualTTS is None:
                    raise ImportError("ChatterboxMultilingualTTS class is None")
                self._model = ChatterboxMultilingualTTS.from_pretrained(
                    device=self.device
                )
            else:
                if ChatterboxTTS is None:
                    raise ImportError("ChatterboxTTS class is None")
                self._model = ChatterboxTTS.from_pretrained(device=self.device)
        except TypeError as e:
            if "'NoneType' object is not callable" in str(e):
                raise ImportError(
                    "chatterbox-tts installation is corrupted. "
                    "Try reinstalling: pip uninstall chatterbox-tts && pip install chatterbox-tts"
                )
            raise

        return self._model

    async def check_health(self) -> bool:
        try:
            if not CHATTERBOX_AVAILABLE:
                self.logger.error(
                    f"TTS health check failed: chatterbox-tts not available. {_chatterbox_error}"
                )
                return False
            await asyncio.get_event_loop().run_in_executor(None, self._get_model)
            self.logger.info("🔥 Warming up TTS+RVC pipeline...")
            await self.synthesize("Hello.")
            self.logger.info("✅ TTS+RVC warm.")
            return True
        except Exception as e:
            self.logger.error(f"TTS health check failed: {e}")
            return False

    async def synthesize(self, text: str):
        """
        Полный синтез: возвращает кортеж (PCM_BYTES, SAMPLE_RATE).
        Используется для офлайн/батч режима.
        """
        try:
            model = await asyncio.get_event_loop().run_in_executor(
                None, self._get_model
            )

            # 1. Генерация TTS
            def _generate():
                if isinstance(model, ChatterboxMultilingualTTS) and self.language_id:
                    return model.generate(
                        text,
                        language_id=self.language_id,
                        audio_prompt_path=self.audio_prompt_path,
                        cfg_weight=0.3,
                    )
                else:
                    return model.generate(
                        text, audio_prompt_path=self.audio_prompt_path, cfg_weight=0.3
                    )

            wav = await asyncio.get_event_loop().run_in_executor(None, _generate)
            sr = getattr(model, "sr", 16000)

            if isinstance(wav, torch.Tensor):
                wave = wav.detach().cpu().numpy()
            else:
                wave = np.array(wav)

            # 2. RVC обработка (если включена)
            if self.rvc_enabled and self.rvc_model_path and rvc_infer:
                try:
                    wave_np = np.asarray(wave)
                    if wave_np.ndim > 1:
                        wave_np = np.squeeze(wave_np)
                    if wave_np.ndim != 1:
                        wave_np = wave_np.reshape(-1)
                    wave_np = wave_np.astype(np.float32, copy=False)

                    rvc_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: rvc_infer(
                            wave_np,
                            sr,
                            self.rvc_model_path,
                            device=self.device,
                            index_path=self.rvc_index_path,
                            index_rate=self.rvc_index_rate,
                            pitch_shift=self.rvc_pitch_shift,
                            f0_method=self.rvc_f0_method,
                        ),
                    )

                    # rvc_infer возвращает (audio_np, sample_rate)
                    if len(rvc_result) == 2:
                        val1, val2 = rvc_result
                        if isinstance(val1, int) and val1 > 1000:
                            sr, wave = val1, val2
                        elif isinstance(val2, int) and val2 > 1000:
                            wave, sr = val1, val2

                except Exception as e:
                    self.logger.error(f"RVC Error (skipping): {e}", exc_info=True)

            # 3. Конвертация в PCM Int16
            if isinstance(wave, np.ndarray):
                wave = torch.from_numpy(wave)

            wave = wave.float()
            if wave.dim() == 2 and wave.size(0) > 1:
                wave = torch.mean(wave, dim=0, keepdim=True)
            elif wave.dim() == 1:
                wave = wave.unsqueeze(0)

            wave = wave.squeeze(0)
            wave = torch.clamp(wave, -1.0, 1.0)
            int16 = (wave.numpy() * 32767.0).astype(np.int16)

            return int16.tobytes(), sr

        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}", exc_info=True)
            return b"", 16000

    async def synthesize_stream(self, text: str):
        """
        Async generator: выдаёт (np.ndarray float32, sample_rate) чанками.

        Пробует использовать нативный стриминг Chatterbox если доступен,
        иначе fallback — генерирует полностью и выдаёт одним чанком.

        Используется с RVCStreamer для pipeline стриминга:
            async for chunk_np, sr in adapter.synthesize_stream(text):
                converted = rvc_streamer.push(chunk_np)
                if converted is not None:
                    await send(converted)
        """
        loop = asyncio.get_event_loop()

        try:
            model = await loop.run_in_executor(None, self._get_model)
        except Exception as e:
            self.logger.error(f"synthesize_stream: model load failed: {e}")
            return

        sr = getattr(model, "sr", 24000)

        # Попытка нативного стриминга Chatterbox
        stream_fn = getattr(model, "generate_stream", None)

        if stream_fn is not None:
            # Нативный стриминг — генерирует чанками в отдельном потоке
            self.logger.debug("synthesize_stream: using native generate_stream")
            q: asyncio.Queue = asyncio.Queue()

            def _producer():
                try:
                    for chunk in stream_fn(
                        text,
                        audio_prompt_path=self.audio_prompt_path,
                    ):
                        if isinstance(chunk, torch.Tensor):
                            arr = chunk.detach().cpu().numpy().squeeze().astype(np.float32)
                        else:
                            arr = np.asarray(chunk).squeeze().astype(np.float32)
                        loop.call_soon_threadsafe(q.put_nowait, (arr, sr))
                except Exception as exc:
                    self.logger.error(f"synthesize_stream producer error: {exc}")
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

            await loop.run_in_executor(None, _producer)

            while True:
                item = await q.get()
                if item is None:
                    break
                yield item

        else:
            # Fallback: полная генерация → один большой чанк
            self.logger.debug("synthesize_stream: fallback to full generate()")

            def _generate_full():
                if isinstance(model, ChatterboxMultilingualTTS) and self.language_id:
                    return model.generate(
                        text,
                        language_id=self.language_id,
                        audio_prompt_path=self.audio_prompt_path,
                        cfg_weight=0.3,
                    )
                return model.generate(
                    text, audio_prompt_path=self.audio_prompt_path, cfg_weight=0.3
                )

            try:
                wav = await loop.run_in_executor(None, _generate_full)
                if isinstance(wav, torch.Tensor):
                    arr = wav.detach().cpu().numpy().squeeze().astype(np.float32)
                else:
                    arr = np.asarray(wav).squeeze().astype(np.float32)
                yield arr, sr
            except Exception as e:
                self.logger.error(f"synthesize_stream fallback failed: {e}")

    async def speak(self, text: str, media_port):
        async with self._speak_lock:
            pcm_data, sr = await self.synthesize(text)

            if not pcm_data:
                return

            if media_port is None:
                return

            self.logger.info(
                f"Sending audio to bridge: {len(pcm_data)} bytes at {sr} Hz"
            )

            success = media_port.update_playback_data(
                pcm_data, sample_rate=sr, validate=True
            )

            if not success:
                self.logger.error("Failed to update bridge playback data")


# Алиас для обратной совместимости с проектами использующими TTSAdapter
TTSAdapter = ChatterboxAdapter
