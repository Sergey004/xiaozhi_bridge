# Адаптация ContentVec hubert_model.py для RVC-инференса
# Источник: https://github.com/auspicious3000/contentvec/blob/main/hubert/hubert_model.py
import torch
import torch.nn as nn
import fairseq
from fairseq.data.dictionary import Dictionary
import torch.serialization

torch.serialization.add_safe_globals([Dictionary])

class Hubert(nn.Module):
    def __init__(self, ckpt_path, device='cpu'):
        super().__init__()
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path], suffix="")
        self.model = models[0]
        self.model.eval()
        self.device = device
        self.model.to(device)

    def forward(self, wav_tensor):
        # wav_tensor: (1, T), float32, 16kHz
        with torch.no_grad():
            feats = self.model.extract_features(wav_tensor)[0]
        return feats
