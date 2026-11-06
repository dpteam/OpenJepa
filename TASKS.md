# Задачи (аналог Makefile, без make)

Все команды предполагают, что текущий каталог — корень репозитория,
а `PYTHONPATH=src`.

## Windows (PowerShell)

```powershell
$env:PYTHONPATH = "src"

python tests/run_all.py                          # тесты
python tools/verify.py                           # полная проверка
python tools/check_docs.py                       # битые ссылки в docs
python tools/check_refs.py                       # валидность arXiv-ссылок

python -m openjepa.runtime.cli train --config configs/smoke.yaml
python -m openjepa.runtime.cli train --config configs/pretrain_toy.yaml
python -m openjepa.runtime.cli train --config configs/hjepa_toy.yaml

python -m openjepa.runtime.cli eval --ckpt runs/toy/best.pt --episodes 20
python -m openjepa.runtime.cli demo --ckpt runs/toy/best.pt --out runs/demo
python -m openjepa.runtime.cli serve --ckpt runs/toy/best.pt --port 8077

python tools/probe_latent.py --ckpt runs/toy/best.pt
python tools/bench.py --ckpt runs/toy/best.pt
python tools/plot.py --run runs/toy
python tools/report.py --run runs/toy
python tools/demo.py --ckpt runs/toy/best.pt
```

## Linux / macOS

```bash
export PYTHONPATH=src
# те же команды
```

## Что чем проверяется

| Вопрос | Команда | Ожидание |
|---|---|---|
| Код не сломан? | `tests/run_all.py` | 32/32 |
| Всё работает end-to-end? | `tools/verify.py` | «ВСЁ В ПОРЯДКЕ» |
| Латент кодирует управляемые факторы? | `tools/probe_latent.py` | асимметрия < 3× |
| Сколько стоит шаг? | `tools/bench.py` | десятки мс |
| Документация не битая? | `tools/check_docs.py` | «все ссылки валидны» |
| Ссылки на статьи живые? | `tools/check_refs.py` | «все ссылки валидны» |

## Порядок при изменении модели

1. `tests/run_all.py` — формы и потери не сломались.
2. `train --config configs/smoke.yaml` — 2 эпохи без NaN.
3. `probe_latent.py` на smoke-чекпоинте — латент не вырожден.
4. `train --config configs/pretrain_toy.yaml` — полный прогон.
5. `report.py` — отчёт с диагнозом.
6. `docs/14-zhurnal-resheniy.md` — записать, если было неочевидно.
