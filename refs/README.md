# refs/ — внешние материалы

Здесь лежат **скачанные статьи и книги** (PDF, EPUB, DJVU). Они не
коммитятся: бинарники, гигабайты, лицензии у каждого свои.

Каталог в `.gitignore`, кроме этого README.

## Что здесь есть сейчас

| Файл | Что это | Источник |
|---|---|---|
| `10356_a_path_towards_autonomous_mach.pdf` | LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) | OpenReview |

## Как пополнять

```bash
# arXiv: подставить ID
curl -L -o refs/2511.08544_lejepa.pdf https://arxiv.org/pdf/2511.08544

# или через скрипт (проверяет, что файл — PDF, а не HTML с ошибкой)
python tools/fetch_refs.py 2511.08544 2610.39302 2603.19312
```

Именование: `<arxiv_id>_<короткое-имя>.pdf`.

## Список приоритетных статей

Полный список — `research/bibliography.md`. Первые по важности:

| arXiv | Что |
|---|---|
| 2511.08544 | LeJEPA (SIGReg) |
| 2610.39302 | Platonic JEPA |
| 2610.06805 | H-JEPA |
| 2603.19312 | LeWorldModel |
| 2506.09985 | V-JEPA 2 |
| 2405.07987 | Platonic Representation Hypothesis |

## Лицензии

Статьи на arXiv распространяются под CC BY / CC BY-SA / arXiv-only.
Проверяйте перед переиспользованием текста или рисунков. Для локального
чтения и цитирования проблем нет.
