# Автоматическая загрузка моделей RMVPE и HuBERT с HuggingFace
import os
import requests

MODEL_URLS = {
    'rmvpe.pt': 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt',
    'rmvpe.onnx': 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx',
    'hubert_base.pt': 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt',
}

def download_model(model_name, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    url = MODEL_URLS[model_name]
    out_path = os.path.join(out_dir, model_name)
    if os.path.exists(out_path):
        print(f"[INFO] {model_name} уже загружен: {out_path}")
        return out_path
    print(f"[INFO] Скачивание {model_name}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[INFO] Скачано: {out_path}")
    return out_path

def download_all_models(out_dir="models"):
    for name in MODEL_URLS:
        download_model(name, out_dir=out_dir)

if __name__ == "__main__":
    download_all_models()
