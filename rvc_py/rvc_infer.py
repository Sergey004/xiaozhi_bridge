
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import os
from .hubert_contentvec import Hubert
from .rvc_model import RVCModel
from .f0_extractor import extract_f0
from .rmvpe_extractor import extract_f0_rmvpe
from .download_models import download_model

_rvc_cache = {}

def rvc_infer(
    wav: np.ndarray,
    sr: int,
    rvc_model_path: str,
    device: str = 'cuda',
    hubert_path: str = None,
    f0_method: str = 'rmvpe',
    rmvpe_model_path: str = None,
    index_path: str = None,
    index_rate: float = 0.0,
    fp16: bool = False,
    pitch_shift: int = 0,
    use_index: bool = False,
    sample_rate: int = None
) -> tuple[np.ndarray, int]:
    global _rvc_cache
    # Normalize/validate device
    requested_device = device
    if isinstance(device, str) and device.startswith('cuda') and not torch.cuda.is_available():
        print(f"[RVC][WARN] Запрошено устройство '{requested_device}', но CUDA недоступна. Переход на CPU.")
        device = 'cpu'
    if device == 'cuda':
        device = 'cuda:0'
    # Include device in cache key so switching CPU<->CUDA doesn't reuse wrong-device models
    cache_key = (rvc_model_path, index_path, fp16, device)
    if cache_key not in _rvc_cache:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'RVC')
        os.makedirs(models_dir, exist_ok=True)
        if hubert_path is None:
            hubert_path = os.path.join(models_dir, 'hubert_base.pt')
        if not os.path.exists(hubert_path):
            print("[RVC] HuBERT не найден, скачиваем автоматически...")
            hubert_path = download_model('hubert_base.pt', out_dir=models_dir)
        print(f"[RVC] Загрузка HuBERT (ContentVec): {hubert_path}")
        hubert = Hubert(hubert_path, device=device)
        print(f"[RVC] Загрузка RVC-модели: {rvc_model_path}")
        rvc = RVCModel(rvc_model_path, device=device, index_path=index_path, fp16=fp16)
        if index_path and index_rate > 0.0:
            rvc.set_index(index_path, index_rate=index_rate)
        _rvc_cache[cache_key] = (hubert, rvc)
        print(f"[RVC] Инициализировано. Эффективное устройство: {device}")
    else:
        hubert, rvc = _rvc_cache[cache_key]
        print(f"[RVC] Использую кэшированные модели на устройстве: {device}")

    # 1. Audio -> float32, mono, 16kHz for HuBERT
    wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    if wav16.ndim == 1:
        wav16 = wav16[None, :]
    elif wav16.ndim > 2:
        wav16 = wav16.reshape(1, -1)
    wav16_tensor = torch.from_numpy(wav16).to(device)

    # 2. HuBERT features (align with butter)
    padding_mask = torch.BoolTensor(wav16_tensor.shape).to(device).fill_(False)
    output_layer = 9 if rvc.version == 'v1' else 12
    with torch.no_grad():
        logits = hubert.model.extract_features(
            source=wav16_tensor,
            padding_mask=padding_mask,
            output_layer=output_layer,
        )
        units = hubert.model.final_proj(logits[0]) if rvc.version == 'v1' else logits[0]
    # Upsample time dimension by 2
    units = F.interpolate(units.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

    # 3. F0 extraction
    if f0_method == 'rmvpe':
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'RVC')
        os.makedirs(models_dir, exist_ok=True)
        if rmvpe_model_path is None:
            rmvpe_model_path = os.path.join(models_dir, 'rmvpe.pt')
        if not os.path.exists(rmvpe_model_path):
            print("[RVC] RMVPE не найден, скачиваем автоматически...")
            rmvpe_model_path = download_model('rmvpe.pt', out_dir=models_dir)
        f0_hz = extract_f0_rmvpe(wav, sr, rmvpe_model_path, device=device)
    else:
        f0_hz = extract_f0(wav, sr, method=f0_method, device=device)

    # Apply pitch shift in semitones to Hz curve
    if pitch_shift != 0:
        f0_hz = f0_hz * (2 ** (pitch_shift / 12))

    # Match f0 length to units length using interpolation (not librosa.resample)
    target_len = units.shape[1]
    src_len = f0_hz.shape[0]
    if src_len != target_len:
        x = np.arange(src_len, dtype=np.float32)
        xp = np.linspace(0, src_len - 1, num=target_len, dtype=np.float32)
        f0_hz_rs = np.interp(xp, x, f0_hz).astype(np.float32)
    else:
        f0_hz_rs = f0_hz.astype(np.float32)

    # Build coarse pitch (1..255) like butter get_f0
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + np.clip(f0_hz_rs, 0, None) / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)

    # Tensors for model.infer
    pitch = torch.tensor(f0_coarse, device=device).unsqueeze(0).long()
    pitchf = torch.tensor(f0_hz_rs, device=device).unsqueeze(0).float()

    # 4. Inference via model.infer
    out = rvc.infer(units, pitch=pitch, pitchf=pitchf, sid=0, use_index=use_index)
    if isinstance(out, tuple):
        wav_out = out[0]
    else:
        wav_out = out
    wav_out = wav_out.detach().cpu().numpy().squeeze()

    # 5. Output sample rate from model if available
    out_sr = rvc.sample_rate if sample_rate is None else sample_rate
    return wav_out, out_sr
