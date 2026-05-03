# F0 extractor (адаптация под rvc-python)
import numpy as np
import librosa

def extract_f0(wav, sr, method='rmvpe', device='cpu'):
    # wav: np.ndarray, float32, mono
    # sr: int
    # method: 'rmvpe' (по умолчанию)
    # Возвращает: f0 (np.ndarray, shape [T])
    if method == 'torchcrepe':
        try:
            import torchcrepe
        except ImportError:
            raise ImportError('Для F0 extraction требуется pip install torchcrepe')
        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav16 = wav16.astype(np.float32)
        time, f0, confidence, activation = torchcrepe.predict(wav16, 16000, viterbi=True, step_size=10)

        print(f"Shape of f0 before return: {f0.shape}")
        return f0
    else:
        raise NotImplementedError(f'F0 extraction method {method} не поддерживается')
