"""
dramabox_adapter.py — DramaBox TTS адаптер для совместимости с tts_adapter.py

DramaBox от Resemble AI: https://github.com/resemble-ai/DramaBox
- 3.3B DiT + Gemma 3 12B encoder, diffusion flow matching
- 48kHz stereo, VRAM ~24GB (RTX 3090 на грани!)
- НЕ использует [laugh] теги — формат сцено-режиссёрский

Использование:
  TTS_ENGINE=dramabox
  DRAMABOX_REPO_PATH=/path/to/DramaBox
  DRAMABOX_VOICE_REF=/path/to/voice_reference.wav

Интеграция:
  DramaBox выдает файл на диск → адаптер читает и возвращает (pcm, sr)
  как обычный TTSAdapter. Совместим с RVC.
"""

import asyncio
import logging
import os
import struct
import wave
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─── Lazy import DramaBox ────────────────────────────────────────────────────
_DRAMABOX_AVAILABLE = False
_TTSServer = None

def _try_import_dramabox():
    global _DRAMABOX_AVAILABLE, _TTSServer
    if _TTSServer is not None:
        return True

    repo_path = os.environ.get("DRAMABOX_REPO_PATH", "")

    # Пытаемся импортировать из репозитория DramaBox
    import sys
    if repo_path and repo_path not in sys.path:
        src_path = os.path.join(repo_path, "src")
        if os.path.isdir(src_path):
            sys.path.insert(0, src_path)
        else:
            sys.path.insert(0, repo_path)

    try:
        from inference_server import TTSServer
        _TTSServer = TTSServer
        _DRAMABOX_AVAILABLE = True
        logger.info("✅ DramaBox imported successfully")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ DramaBox not available: {e}")
        logger.warning("   Set DRAMABOX_REPO_PATH in .env to the DramaBox repo directory")
        logger.warning("   Install: git clone https://github.com/resemble-ai/DramaBox")
        _DRAMABOX_AVAILABLE = False
        return False


class DramaBoxAdapter:
    """
    Адаптер для DramaBox TTS с интерфейсом совместимым с TTSAdapter.
    
    Возвращает (pcm_bytes, sample_rate) — 48kHz стерео.
    Мост (xiaozhi_bridge.py) сам сделает downsampling до 16kHz моно.
    """

    def __init__(self, config: dict, logger_instance: logging.Logger):
        self.config = config
        self.logger = logger_instance.getChild("DramaBox")
        self._server = None
        self._voice_ref = os.environ.get("DRAMABOX_VOICE_REF", "")

        # Параметры генерации
        self.cfg_scale = float(os.environ.get("DRAMABOX_CFG_SCALE", "2.5"))
        self.stg_scale = float(os.environ.get("DRAMABOX_STG_SCALE", "1.5"))
        self.duration_multiplier = float(os.environ.get("DRAMABOX_DURATION_MULT", "1.1"))
        self.seed = int(os.environ.get("DRAMABOX_SEED", "42"))
        self.ref_duration = float(os.environ.get("DRAMABOX_REF_DURATION", "10.0"))

        # RVC
        self.rvc_enabled = config.get("tts", {}).get("rvc_enabled", False)
        self.rvc_model_path = config.get("tts", {}).get("rvc_model_path")
        self.rvc_index_path = config.get("tts", {}).get("rvc_index_path")
        self.rvc_f0_method = config.get("tts", {}).get("rvc_f0_method", "rmvpe")
        self.rvc_pitch_shift = config.get("tts", {}).get("rvc_pitch_shift", 0)
        self.rvc_index_rate = config.get("tts", {}).get("rvc_index_rate", 0.5)

        try:
            from rvc_py.rvc_infer import rvc_infer
            self._rvc_infer = rvc_infer
        except ImportError:
            self._rvc_infer = None
            if self.rvc_enabled:
                self.logger.warning("RVC enabled but rvc_infer not available")

        self._lock = asyncio.Lock()
        self._tmp_counter = 0

    def _get_server(self):
        if self._server is not None:
            return self._server

        if not _try_import_dramabox():
            raise RuntimeError("DramaBox not available. Set DRAMABOX_REPO_PATH.")

        device = self.config.get("tts", {}).get("device", "cuda")
        self.logger.info(f"🎭 Initializing DramaBox on {device} (this takes ~30s)...")

        self._server = _TTSServer(device=device, dtype="bf16")
        self.logger.info("✅ DramaBox ready!")
        return self._server

    def _wav_to_pcm(self, wav_path: str) -> tuple:
        """Читает WAV файл, возвращает (pcm_float32_numpy, sample_rate, channels)."""
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if n_ch == 2:
            samples = samples.reshape(-1, 2)

        return samples, sr, n_ch

    async def check_health(self) -> bool:
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._get_server)
            self.logger.info("🔥 Warming up DramaBox...")
            result = await self.synthesize("Hello.")
            if result[0]:
                self.logger.info("✅ DramaBox warm.")
                return True
            return False
        except Exception as e:
            self.logger.error(f"DramaBox health check failed: {e}")
            return False

    async def synthesize(self, text: str):
        """
        Генерирует речь через DramaBox.
        Возвращает: (pcm_bytes_int16, sample_rate)
        """
        import tempfile

        async with self._lock:
            try:
                server = await asyncio.get_event_loop().run_in_executor(
                    None, self._get_server
                )

                # Генерируем во временный файл
                self._tmp_counter += 1
                tmp_path = tempfile.mktemp(
                    suffix=".wav",
                    prefix=f"dramabox_{self._tmp_counter}_"
                )

                voice_ref = self._voice_ref if self._voice_ref else None

                def _generate():
                    server.generate_to_file(
                        prompt=text,
                        output=tmp_path,
                        voice_ref=voice_ref,
                        cfg_scale=self.cfg_scale,
                        stg_scale=self.stg_scale,
                        duration_multiplier=self.duration_multiplier,
                        seed=self.seed,
                        gen_duration=0.0,  # auto
                        ref_duration=self.ref_duration,
                        watermark=False,
                    )
                    return tmp_path

                out_path = await asyncio.get_event_loop().run_in_executor(None, _generate)

                # Читаем результат
                samples, sr, n_ch = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._wav_to_pcm(out_path)
                )

                # Удаляем временный файл
                try:
                    os.remove(out_path)
                except OSError:
                    pass

                # Стерео → моно (для совместимости с мостом)
                if n_ch == 2:
                    wave_np = np.mean(samples, axis=1).astype(np.float32)
                else:
                    wave_np = samples.astype(np.float32)

                # RVC
                if self.rvc_enabled and self.rvc_model_path and self._rvc_infer:
                    try:
                        rvc_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self._rvc_infer(
                                wave_np,
                                sr,
                                self.rvc_model_path,
                                device=self.config.get("tts", {}).get("device", "cuda"),
                                index_path=self.rvc_index_path,
                                index_rate=self.rvc_index_rate,
                                pitch_shift=self.rvc_pitch_shift,
                                f0_method=self.rvc_f0_method,
                            ),
                        )
                        if len(rvc_result) == 2:
                            val1, val2 = rvc_result
                            if isinstance(val1, int) and val1 > 1000:
                                sr = val1
                                wave_np = val2
                            elif isinstance(val2, int) and val2 > 1000:
                                wave_np = val1
                                sr = val2
                    except Exception as e:
                        self.logger.error(f"RVC error (skipping): {e}")

                # Конвертация в PCM int16
                wave_tensor = torch.from_numpy(wave_np).float()
                wave_tensor = torch.clamp(wave_tensor, -1.0, 1.0)
                int16 = (wave_tensor.numpy() * 32767.0).astype(np.int16)

                self.logger.info(
                    f"🎭 DramaBox: {len(int16)} bytes, {sr}Hz, "
                    f"{len(int16) / (2 * sr):.1f}s"
                )

                return int16.tobytes(), sr

            except Exception as e:
                self.logger.error(f"DramaBox synthesis failed: {e}", exc_info=True)
                return b"", 48000
