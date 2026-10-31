[English](README.md) | Русский

# Xiaozhi Server — голосовой бэкенд для xiaozhi-клиентов

`xiaozhi_server.py` — сервер протокола xiaozhi «всё в одном»:
**OTA + WebSocket-мост + OpenAI-совместимый STT/TTS/LLM + Dashboard**.
Заточен под клиента на Яндекс Станции
([Sergey004/xiaozhi_yandex_station](https://github.com/Sergey004/xiaozhi_yandex_station)),
но работает с любым xiaozhi-устройством, включая ESP32.

## Возможности

- **WebSocket-мост** (`/ws`, Bearer-токен, whitelist устройств по MAC):
  server-side VAD с автокалибровкой шума, стриминг аудио в обе стороны,
  пер-девайсный даунлинк (HD-клиенты 48 кГц, ESP32 16 кГц), отмена/abort,
  ping/pong, контекст диалога и память по device_id.
- **Голос: Chatterbox TTS → RVC.** Клонирование голоса по референсу
  (`AUDIO_PROMPT_PATH`), батчевый `rvc_infer` (HuBERT/ContentVec, rmvpe,
  Faiss-index), **автовыравнивание высоты**: медианная f0 референса и
  генерации → точный полутоновый сдвиг, чтобы выход всегда сидел в регистре
  целевого голоса (замерено: 139 Гц при цели 142). Даунлинк нормализуется в
  `DOWNLINK_SR` (по умолчанию 48 кГц).
- **STT**: Whisper (`/v1/audio/transcriptions`, `/stt`), язык настраивается.
- **LLM**: NVIDIA NIM / Ollama (`/v1/chat/completions`), memory по устройству.
- **OTA** (`/ota`, `/xiaozhi/ota`): активация, конфиг, firmware.
- **Dashboard + API**: `/` — статус и устройства, `/health`, `/devices`,
  `/logs/stream`, `/tools`, `/memory/{device_id}`.

## Быстрый старт

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp env.example .env          # токены, RVC-модель, референс голоса, LLM
./start_server.sh            # OTA+API на :8000, WS-мост на :8765
```

Клиент (`xiaozhi_config.json`):

```json
{ "ws_url": "ws://SERVER:8765/ws",
  "ota_url": "http://SERVER:8000/xiaozhi/ota/",
  "ws_token": "BRIDGE_TOKEN" }
```

MAC устройства добавляется в `ALLOWED_DEVICES`; HD-даунлинк включается флагом
`features.downlink_hd` в hello клиента (ESP32 без флага получают 16 кГц).

## Конфигурация

Всё через `.env` (шаблон — [env.example](env.example)): движок TTS
(`turbo|multilingual|dramabox`), пути RVC-модели/index/референса,
`DOWNLINK_SR`, `TTS_STREAMING` (`off|sentence|chunk` — батч/по предложениям/
стриминг), VAD-параметры, LLM-провайдер.

## Архитектура

```
клиент (WS 8765) ──▶ server-side VAD ──▶ Whisper STT ──▶ LLM ──▶ Chatterbox
                                                                    │
клиент ◀── opus 48к/16к paced-отправка ◀── AudioRateController ◀── RVC
```

Тесты конвертации: `test_tts_rvc.py` (батч, HuBERT vs ContentVec),
`test_tts_stream.py` (A/B батч против стриминга).

## Статус

Рабочий голосовой цикл со Станцией Лайт 2 (vqe_0 → STT → LLM → RVC → динамик).
Известные ограничения: батчевый RVC добавляет задержку на фразу
(`TTS_STREAMING=sentence` смягчает); стриминговый путь — другой тембр.

## Лицензия

См. [LICENSE](LICENSE). Модели RVC/референсы — собственность их владельцев,
использовать только с голосами, на которые у вас есть права.
