# Модуль для загрузки HuBERT (квантайзер)
# Адаптация под rvc-python (https://github.com/daswer123/rvc-python)
import torch
import torch.nn as nn

class HubertSoft(nn.Module):
    def __init__(self, hubert_path, device='cpu'):
        super().__init__()
        # PyTorch >=2.6: weights_only=True по умолчанию, но HuBERT требует False
        try:
            checkpoint = torch.load(hubert_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(hubert_path, map_location=device)
        # Если это state_dict (OrderedDict), загружаем в HuBERT из fairseq
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model = checkpoint['model']
        elif isinstance(checkpoint, dict):
            try:
                from fairseq.models.hubert import HubertModel
                # Важно: task=None, но модель должна быть создана с правильным config
                self.model = HubertModel.build_model({'w2v_path': hubert_path}, task=None)
                self.model.load_state_dict(checkpoint)
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить HuBERT из state_dict: {e}\nУбедитесь, что fairseq установлен и версия модели совместима.")
        else:
            raise RuntimeError("Не удалось распознать формат чекпоинта HuBERT. Проверьте файл или используйте другой.")
        self.model.eval()
        self.device = device
        self.model.to(device)

    def forward(self, wav_tensor):
        # wav_tensor: (1, T), float32, 16kHz
        with torch.no_grad():
            units = self.model.extract_features(wav_tensor)[0]
        return units
