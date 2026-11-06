# Библиография

> Все ссылки проверены на момент написания (6 ноября 2026).
> Порядок — по значимости для проекта.

---

## Основные работы

### 1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence

Позиционная статья, в которой введены JEPA, H-JEPA, cost module и
двухуровневая схема принятия решений.

* **Где:** OpenReview, `https://openreview.net/forum?id=BZ5a1r-kVsf`
* **Не на arXiv.** Версия 0.9.2, 2022-06-27.
* **Зачем:** источник терминологии и всей архитектурной рамки.

```bibtex
@article{lecun2022path,
  title   = {A Path Towards Autonomous Machine Intelligence},
  author  = {LeCun, Yann},
  year    = {2022},
  journal = {OpenReview},
  note    = {Version 0.9.2, 2022-06-27},
  url     = {https://openreview.net/forum?id=BZ5a1r-kVsf}
}
```

---

### 2. Balestriero, R., LeCun, Y. (2025). LeJEPA

**arXiv:2511.08544** — https://arxiv.org/abs/2511.08544

Теория JEPA: изотропный гауссиан как оптимум, SIGReg, устранение эвристик.
Наша функция потерь — прямая реализация этой работы.

```bibtex
@misc{balestriero2025lejepa,
  title  = {LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author = {Balestriero, Randall and LeCun, Yann},
  year   = {2025},
  eprint = {2511.08544},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2511.08544}
}
```

---

### 3. Balestriero, R., LeCun, Y. (2026). Proving and Improving the Platonic
Representation Hypothesis with JEPAs

**arXiv:2610.39302** — https://arxiv.org/abs/2610.39302

Платоновская теорема: JEPA на разных модальностях сходятся с точностью до
одного поворота. 42× эффективность по данным. Обосновывает языковой мост.

```bibtex
@misc{balestriero2026platonic,
  title  = {Proving and Improving the Platonic Representation Hypothesis with JEPAs},
  author = {Balestriero, Randall and LeCun, Yann},
  year   = {2026},
  eprint = {2610.39302},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url    = {https://arxiv.org/abs/2610.39302}
}
```

---

### 4. Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. (2026).
LeWorldModel

**arXiv:2603.19312** — https://arxiv.org/abs/2603.19312

Первый стабильный end-to-end JEPA из пикселей с двумя членами потери.
15M параметров, одна GPU, 48× ускорение планирования.

```bibtex
@misc{maes2026leworldmodel,
  title  = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  year   = {2026},
  eprint = {2603.19312},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2603.19312}
}
```

---

### 5. Zhang, W., Terver, B., Rabbat, M., LeCun, Y., Balestriero, R. (2026).
H-JEPA

**arXiv:2610.06805** — https://arxiv.org/abs/2610.06805

Иерархия action-conditioned JEPA. Visual AntMaze: 18% → 73% при меньшем
бюджете планирования.

```bibtex
@misc{zhang2026hjepa,
  title  = {H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning},
  author = {Zhang, Wancong and Terver, Basile and Rabbat, Michael and LeCun, Yann and Balestriero, Randall},
  year   = {2026},
  eprint = {2610.06805},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2610.06805}
}
```

---

### 6. Assran, M. et al. (2025). V-JEPA 2

**arXiv:2506.09985** — https://arxiv.org/abs/2506.09985

Видео-JEPA: 1M+ часов видео, action-conditioned world model на 62 ч DROID,
zero-shot планирование на Franka.

```bibtex
@misc{assran2025vjepa2,
  title  = {V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author = {Assran, Mido and Bardes, Adrien and Fan, David and Garrido, Quentin and others},
  year   = {2025},
  eprint = {2506.09985},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url    = {https://arxiv.org/abs/2506.09985}
}
```

---

## Поддерживающие работы

### 7. Huh, M., Cheung, B., Wang, T., Isola, P. (2024). The Platonic
Representation Hypothesis

**arXiv:2405.07987** — https://arxiv.org/abs/2405.07987

Эмпирическая гипотеза о сходимости представлений. Предшественник работы [3].

---

### 8. Assran, M. et al. (2023). I-JEPA

**arXiv:2301.08243** — https://arxiv.org/abs/2301.08243

Первая рабочая демонстрация JEPA на изображениях. ViT-H/14 на ImageNet-1k
за <72 ч на 16×A100.

---

### 9. Zhou, G., Pan, H., LeCun, Y., Pinto, L. (2024). DINO-WM

**arXiv:2411.04983** — https://arxiv.org/abs/2411.04983

World model на замороженных признаках DINOv2. Полезный baseline.

---

### 10. Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. (2023). DreamerV3

**arXiv:2301.04104** — https://arxiv.org/abs/2301.04104

Классический (реконструктивный) world model. Контрольная точка для сравнения.

---

### 11. Zholus, A. et al. (2026). RoboJEPA

**arXiv:2610.10515** — https://arxiv.org/abs/2610.10515

8B-предиктор, 12 embodiments, степенной закон по compute. Главный аргумент
за масштабируемость JEPA.

---

### 12. Gao, Y., Xu, X. (2026). Fast LeWorldModel

**arXiv:2606.26217** — https://arxiv.org/abs/2606.26217

Предсказание префиксов действий вместо пошагового rollout. Медленнее растёт
ошибка на длинном горизонте.

---

### 13. Caselli, N. et al. (2026). Mind the Gap: Promises and Pitfalls of
Hierarchical Planning in LeWorldModel

**arXiv:2607.12547** — https://arxiv.org/abs/2607.12547

Честный разбор, когда иерархия **не** помогает. Полезно как противоядие
от оптимизма.

---

### 14. Liu, C. et al. (2026). Temporally Centered SIGReg

**arXiv:2607.26924** — https://arxiv.org/abs/2607.26924

Показывает, что SIGReg на всём латенте смещает дисперсию к медленной
компоненте. Применение SIGReg к временно-центрированным остаткам улучшает
политики на LIBERO (63.6% → 83.8%).

---

## Бенчмарки

### 15. Riochet, R. et al. (2018). IntPhys

**arXiv:1803.07616** — https://arxiv.org/abs/1803.07616

### 16. Bordes, F. et al. (2025). IntPhys 2

**arXiv:2506.09849** — https://arxiv.org/abs/2506.09849

Современные модели — на уровне случайного угадывания (≈50%), люди — почти
идеально.

### 17. Bear, D. et al. (2021). Physion

**arXiv:2106.08261** — https://arxiv.org/abs/2106.08261

### 18. Tung, H.-Y. et al. (2023). Physion++

**arXiv:2306.15668** — https://arxiv.org/abs/2306.15668

---

## Не на arXiv (web-источники)

| Работа | Ссылка | Что это |
|---|---|---|
| LeCun 2022 position paper | https://openreview.net/forum?id=BZ5a1r-kVsf | манифест |
| LeJEPA minimal example | https://github.com/rbalestr-lab/lejepa | референсная реализация |
| Genie (DeepMind) | блог DeepMind | генеративные интерактивные миры |
| Marble (World Labs) | сайт World Labs | 3D-сцена по фото |
| Видео NeuroEditor | https://www.youtube.com/watch?v=55d0Fti7dwk | источник задачи |

---

## Как цитировать этот репозиторий

```bibtex
@misc{openjepa2026,
  title  = {OpenJEPA: Architecture, Theory and Reference Implementation},
  year   = {2026},
  note   = {JEPA-based world model stack},
  url    = {https://github.com/<your-org>/openjepa}
}
```

---

## Проверка ссылок

Скрипт (требует интернет):

```bash
python tools/check_refs.py
```

Он проходит по списку arXiv ID и проверяет, что статья существует и
название совпадает. См. `tools/check_refs.py`.
