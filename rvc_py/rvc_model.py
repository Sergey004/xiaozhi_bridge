# RVC Model (адаптация под rvc-python)
import torch
import torch.nn as nn
import numpy as np
import os
try:
    import faiss
except ImportError:
    faiss = None
    print("[WARN] faiss не установлен — retrieval blending будет недоступен.")

from ai_core.rvc_py.lib.infer_pack.models_dml import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono

class RVCModel(nn.Module):
    def __init__(self, model_path, device='cpu', index_path=None, fp16=False, sample_rate=None):
        super().__init__()
        # Загрузи на cpu сначала
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)  # weights_only=False для совместимости, но unsafe если файл не trusted
        cpt = checkpoint
        version = cpt.get('version', 'v1')
        if_f0 = cpt.get('f0', 1)
        config = cpt['config']  # предполагаем list, как в RVC
        self.sample_rate = sample_rate or config[-1]  # sr последний в config
        is_half = fp16

        # Создай модель по версии
        if version == 'v1':
            if if_f0 == 1:
                self.model = SynthesizerTrnMs256NSFsid(*config, is_half=is_half)
            else:
                self.model = SynthesizerTrnMs256NSFsid_nono(*config)
        elif version == 'v2':
            if if_f0 == 1:
                self.model = SynthesizerTrnMs768NSFsid(*config, is_half=is_half)
            else:
                self.model = SynthesizerTrnMs768NSFsid_nono(*config)
        else:
            raise ValueError(f"Unknown model version: {version}")

        # Загрузи weights
        weight = cpt.get('weight', cpt)
        self.model.load_state_dict(weight, strict=False)
        self.model.eval()
        self.model = self.model.to(device)
        if is_half:
            self.model = self.model.half()

        self.device = device
        self.fp16 = fp16
        self.if_f0 = if_f0
        self.version = version
        # Retrieval index (Faiss)
        self.index = None
        self.big_npy = None
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            try:
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            except Exception:
                self.big_npy = None
        self.index_rate = 0.0

    def _get_sample_rate_from_config(self, checkpoint):
        # Fallback, если config не list
        if 'config' in checkpoint and 'sr' in checkpoint['config']:
            return int(checkpoint['config']['sr'])
        return 40000

    def set_index(self, index_path, index_rate=0.5):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            try:
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            except Exception:
                self.big_npy = None
            self.index_rate = index_rate

    def infer(self, units, pitch=None, pitchf=None, sid=0, use_index=False):
        # units: (1, T, C)
        # pitch: (1, T) Long [1..255] or None
        # pitchf: (1, T) Float nsf-f0 in Hz or None
        with torch.no_grad():
            # FP16
            if self.fp16:
                units = units.half()
                if pitchf is not None:
                    pitchf = pitchf.half()
            # Retrieval blending (Faiss)
            if use_index and self.index is not None and self.index_rate > 0.0:
                units_np = units.detach().cpu().numpy().squeeze(0)  # (T, C)
                npy = units_np.astype('float32') if units_np.dtype != np.float32 else units_np
                try:
                    k = min(8, self.index.ntotal)
                    score, ix = self.index.search(npy, k=k)  # (T,k)
                    if self.big_npy is None:
                        self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                    gather = self.big_npy[ix]  # (T,k,C)
                    # weights: inverse square of distances
                    # Avoid div by zero
                    score = np.maximum(score, 1e-6)
                    weight = (1.0 / (score ** 2))
                    weight /= weight.sum(axis=1, keepdims=True)
                    blend = np.sum(gather * weight[..., None], axis=1)  # (T,C)
                    if self.fp16:
                        blend = blend.astype('float16')
                    units = torch.from_numpy(self.index_rate * blend + (1 - self.index_rate) * units_np).unsqueeze(0).to(self.device)
                    units = units.to(torch.float16) if self.fp16 else units.to(torch.float32)
                except Exception:
                    pass
            # Длины
            phone_lengths = torch.tensor([units.shape[1]], device=self.device).long()
            sid_tensor = torch.tensor([sid], device=self.device).long()
            # Вызов инференса модели (использовать .infer, не .forward)
            if self.if_f0 == 1 and pitch is not None and pitchf is not None:
                out = self.model.infer(units, phone_lengths, pitch, pitchf, sid_tensor)
            else:
                out = self.model.infer(units, phone_lengths, sid_tensor)
            return out