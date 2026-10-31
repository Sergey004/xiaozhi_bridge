"""
test_tts_rvc.py — Тест TTS→RVC pipeline для сравнения HuBERT vs ContentVec.

Запуск:
    # Сравнение обоих бэкендов:
    python test_tts_rvc.py

    # Только один бэкенд:
    RVC_ENCODER=contentvec python test_tts_rvc.py

Выходные файлы:
    test_tts_only.wav        — чистый TTS без RVC
    test_rvc_hubert.wav      — TTS → RVC (HuBERT)
    test_rvc_contentvec.wav  — TTS → RVC (ContentVec)
"""
import os
import sys
import torch
import numpy as np
import soundfile as sf

# ── Конфиг ───────────────────────────────────────────────────────────────────

TEXT = "«Welcome back! Your Yandex Station is now speaking in full forty-eight kilohertz quality through the RVC voice of Glamrock Freddy"

RVC_MODEL   = "models/RVC/Atreyu_Alcantar/Atreyu_Alcantar_38e_2546s_best_epoch.pth"
RVC_INDEX   = "models/RVC/Atreyu_Alcantar/added_Atreyu_Alcantar_v2.index"    # например: "ai_core/models/RVC/PhoneGuyFNAF1/PhoneGuyFNAF1.index"
INDEX_RATE  = 0.5     # 0.0 = индекс отключён, 1.0 = полностью
AUDIO_PROMPT = "models/RVC/Atreyu_Alcantar/Atreyu_Alcantar.wav"   # путь к референсному WAV для клонирования голоса, или None
PITCH_SHIFT  = 0      # полутоны


# ── TTS ───────────────────────────────────────────────────────────────────────

def generate_tts(text: str) -> tuple[np.ndarray, int]:
    print(f"\n🔊 TTS: '{text[:60]}...'")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT)
    sr = model.sr
    if isinstance(wav, torch.Tensor):
        arr = wav.detach().cpu().numpy().squeeze().astype(np.float32)
    else:
        arr = np.asarray(wav).squeeze().astype(np.float32)
    print(f"   TTS done: {len(arr)/sr:.2f}s @ {sr}Hz")
    return arr, sr

# ── RVC ───────────────────────────────────────────────────────────────────────

def run_rvc(wav: np.ndarray, sr: int, encoder: str) -> tuple[np.ndarray, int]:
    print(f"\n🎙️  RVC [{encoder}]: processing...")
    os.environ["RVC_ENCODER"] = encoder

    # Сбрасываем кеш моделей чтобы загрузился нужный энкодер
    import rvc_py.rvc_infer as ri
    import rvc_py.hubert_contentvec as hc
    ri._rvc_cache.clear()

    # Перезагружаем модуль энкодера с новым env
    import importlib
    importlib.reload(hc)
    importlib.reload(ri)

    from rvc_py.rvc_infer import rvc_infer
    out, out_sr = rvc_infer(
        wav, sr,
        RVC_MODEL,
        device="cuda",
        pitch_shift=PITCH_SHIFT,
        f0_method="rmvpe",
        index_path=RVC_INDEX,
        index_rate=INDEX_RATE,
        use_index=RVC_INDEX is not None,
    )
    print(f"   RVC done: {len(out)/out_sr:.2f}s @ {out_sr}Hz")
    return out, out_sr

# ── Сохранение WAV ────────────────────────────────────────────────────────────

def save(path: str, wav: np.ndarray, sr: int):
    # Нормализация чтобы не было клиппинга
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    sf.write(path, wav, sr)
    print(f"   💾 Saved: {path}  ({len(wav)/sr:.2f}s @ {sr}Hz)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    encoders = os.environ.get("RVC_ENCODER")

    # 1. TTS
    tts_wav, tts_sr = generate_tts(TEXT)
    save("test_tts_only.wav", tts_wav, tts_sr)

    # 2. RVC с HuBERT
    if encoders is None or encoders == "hubert":
        try:
            out, sr = run_rvc(tts_wav, tts_sr, "hubert")
            save("test_rvc_hubert.wav", out, sr)
        except Exception as e:
            print(f"   ⚠️  HuBERT failed: {e}")

    # 3. RVC с ContentVec
    if encoders is None or encoders == "contentvec":
        try:
            out, sr = run_rvc(tts_wav, tts_sr, "contentvec")
            save("test_rvc_contentvec.wav", out, sr)
        except Exception as e:
            print(f"   ⚠️  ContentVec failed: {e}")

    print("\n✅ Done! Файлы для сравнения:")
    for f in ["test_tts_only.wav", "test_rvc_hubert.wav", "test_rvc_contentvec.wav"]:
        if os.path.exists(f):
            print(f"   {f}")

if __name__ == "__main__":
    main()
