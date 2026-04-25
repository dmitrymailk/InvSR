# Plan: запуск train для InvSR через `configs/sd-turbo-sr-ldis.yaml`

Дата: 2026-04-25

Цель: запустить train для `InvSR` с конфигом `auto_remaster/sandbox/InvSR/configs/sd-turbo-sr-ldis.yaml` без переписывания Python-кода. Разрешены только:

- скачивание внешних данных и весов;
- правки путей и значений в YAML;
- запуск штатного train entrypoint.

План ниже собран по локальному коду, `README.md` проекта и GitHub issues `#16`, `#19`, `#39`, `#43`, `#45`, `#46` репозитория `zsyOAOA/InvSR`.

## Что обязательно нужно подготовить

### 1. LPIPS checkpoint

Это не опционально. В трейнере latent LPIPS грузится напрямую из пути, указанного в YAML, и при отсутствии файла train падает еще на стадии `build_model`.

Текущий ожидаемый путь:

```text
weights/vgg16_sdturbo_lpips.pth
```

Источник указан в `README.md` проекта:

- `https://huggingface.co/OAOA/InvSR/resolve/main/vgg16_sdturbo_lpips.pth?download=true`

Что важно по issues:

- в issue `#45` автор прямо указывает на конфиг как на источник истины для пути к LPIPS-весу;
- класть файл в `latent_lpips/weights/...` для train не нужно, если в YAML оставлен путь `weights/vgg16_sdturbo_lpips.pth`.

### 2. SD-Turbo base model

При наличии сети его можно не раскладывать вручную: код использует `from_pretrained(...)` с `cache_dir`, и модель может быть автоматически скачана Hugging Face в кэш.

Текущая настройка в YAML:

```yaml
sd_pipe:
	params:
		pretrained_model_name_or_path: stabilityai/sd-turbo
		cache_dir: weights
```

Вывод:

- если интернет на машине есть, достаточно оставить `cache_dir: weights`;
- если запуск будет без сети, нужно заранее разложить локальный HF cache для `stabilityai/sd-turbo` в структуре, описанной в issue `#16`, и при необходимости добавить `revision` в `sd_pipe.params`.

### 3. Train datasets

Судя по текущему YAML, train использует две выборки:

- `FFHQ`
- `LSDIR`

Автор в issues `#19` и `#39` подтверждает, что для запуска train достаточно:

- скачать датасет;
- указать его корневую папку в `root_path`;
- указать подпапку с картинками в `image_path`.

То есть код не требует какого-то специального формата метаданных, если используются только изображения.

### 4. Validation dataset

Текущий YAML включает блок `data.val`, а значит validation не отключен.

Это важно, потому что:

- rank 0 создаст val dataset автоматически, если секция `data.val` присутствует;
- на первом validation создаются метрики через `pyiqa`, в том числе `lpips-vgg`.

Практический вывод:

- для полного штатного запуска нужен не только train dataset, но и валидейшн-данные;
- если интернет на машине есть, `pyiqa` обычно сможет подтянуть нужные веса сам;
- если интернет исчезнет позже или окружение ограничено, validation тоже может стать источником ошибок, даже если train data и LPIPS уже подготовлены.

## Структура данных, которую реально ждет код

### FFHQ

По аналогии с issue `#39`, если картинки лежат так:

```text
/data/FFHQ/images1024/*.png
```

то в YAML должно быть примерно так:

```yaml
source1:
	root_path: /data/FFHQ
	image_path: images1024
	im_ext: png
```

### LSDIR

Если картинки лежат так:

```text
/data/LSDIR/train/images/*.png
```

то в YAML должно быть примерно так:

```yaml
source2:
	root_path: /data/LSDIR/train
	image_path: images
	im_ext: png
```

### Validation

Судя по текущему конфигу, validation ждет две папки с совпадающими именами файлов:

- low-quality изображения;
- ground-truth изображения.

Пример:

```text
/data/imagenet512/lq/0001.png
/data/imagenet512/gt/0001.png
```

В YAML это должно выглядеть так:

```yaml
val:
	params:
		dir_path: /data/imagenet512/lq
		extra_dir_path: /data/imagenet512/gt
		im_exts: png
```

`README.md` проекта рекомендует synthetic `ImageNet-Test` для воспроизведения paper-style validation.

## Критичные блокеры, найденные в коде

### Блокер 1. Для FFHQ нельзя скачать ровно 20000 файлов и оставить `length: 20000`

В `basicsr/data/realesrgan_dataset.py` стоит проверка:

```python
assert configs.length < len(image_stems)
```

То есть код требует строгое неравенство, а не `<=`.

Следствие:

- если в YAML оставить `source1.length: 20000`, в папке FFHQ должно быть больше 20000 PNG-файлов;
- если скачать ровно 20000 картинок, train упадет на создании датасета;
- либо нужно скачать больше 20000 файлов;
- либо уменьшить `length` в YAML до числа, которое строго меньше фактического числа файлов.

Этот риск согласуется с issue `#43`, где падение происходит именно на стадии инициализации `RealESRGANDataset`.

### Блокер 2. Validation требует реальные файлы и парность имен

В `datapipe/datasets.py` для base dataset используется выборка:

```python
random.sample(file_paths_all, length)
```

Следствие:

- в `dir_path` должно быть как минимум `length` файлов;
- в `extra_dir_path` должны существовать файлы с теми же именами;
- для текущего YAML нужно минимум `16` файлов в каждой val-папке.

### Блокер 3. Validation может тянуть внешние веса через `pyiqa`

Валидация создает метрики:

- `psnr`
- `lpips-vgg`

Следствие:

- даже если train уже стартует, первый validation может потребовать сетевой доступ или заранее прогретый кэш `pyiqa`.

## Что говорят GitHub issues

### Issue #19: How to put the training dataset

Ключевой ответ автора:

- достаточно скачать train dataset и прописать путь в `root_path` конфига;
- отдельного специального процесса подготовки датасета автор не описывает.

Практический смысл:

- запуск не должен требовать переписывания пайплайна;
- основная работа действительно сводится к скачиванию данных и корректным путям в YAML.

### Issue #39: training dataset folder structure

Ключевой ответ автора:

- `root_path` указывает на корень датасета;
- `image_path` указывает на подпапку с картинками.

Это подтверждает, что структура вида `root_path/image_path/*.png` является ожидаемой.

### Issue #43: assert configs.length < len(image_stems)

Ключевой смысл:

- падение на недостаточном количестве изображений является ожидаемым при текущем коде;
- это не проблема окружения, а конкретное ограничение dataset loader.

### Issue #45: training and inference without internet

Ключевой смысл:

- путь к LPIPS checkpoint нужно брать из YAML;
- офлайн-сценарий требует заранее подготовленных локальных весов.

### Issue #16: where should I put sd_turbo files

Ключевой смысл:

- для офлайн-режима нужен не один `sd_turbo.safetensors`, а структура локального кэша Hugging Face;
- в отдельных случаях нужно добавить `revision` в `sd_pipe.params`.

Для текущей задачи это не основной путь, потому что сеть у машины есть.

### Issue #46: NaN during training

Ключевой смысл:

- если train уже идет, но позже появляются `NaN`, автор советует сначала уменьшать learning rate;
- автор отдельно отмечает, что иногда проблема бывает в `RealESRGAN` degradation pipeline.

Это не относится к начальному bring-up, но это ближайший следующий риск после успешного старта.

## Исполнимый план по шагам

### Шаг 1. Скачать обязательные веса

Нужно получить:

- `weights/vgg16_sdturbo_lpips.pth`

SD-Turbo отдельно руками не трогаем, если train будет запускаться на машине с интернетом.

### Шаг 2. Скачать train datasets

Нужно подготовить:

- FFHQ с PNG-изображениями в отдельной папке;
- LSDIR train с PNG-изображениями в подпапке `images` или в другой папке, но тогда это имя надо явно прописать в YAML.

Минимальное требование для FFHQ:

- файлов должно быть больше, чем значение `source1.length`.

Если оставить текущее значение:

```yaml
length: 20000
```

то нужно иметь минимум `20001` PNG-файл.

### Шаг 3. Скачать validation data

Нужно подготовить две папки:

- `lq`
- `gt`

Требования:

- одинаковые имена файлов;
- не менее `16` PNG-пар для текущего YAML.

Если хочется быть ближе к исходному setup автора, использовать synthetic `ImageNet-Test`, на который ссылается `README.md`.

### Шаг 4. Обновить только YAML, без правки Python

В `configs/sd-turbo-sr-ldis.yaml` нужно обновить:

- `data.train.params.data_source.source1.root_path`
- `data.train.params.data_source.source1.image_path`
- `data.train.params.data_source.source2.root_path`
- `data.train.params.data_source.source2.image_path`
- `data.val.params.dir_path`
- `data.val.params.extra_dir_path`

При необходимости также обновить:

- `source1.length`, если скачано меньше, чем ожидает текущий YAML;
- `train.batch` и `train.microbatch`, если GPU не выдерживает текущие значения.

### Шаг 5. Сделать preflight-проверку перед полным train

Перед полноценным запуском нужно проверить:

1. существует ли файл `weights/vgg16_sdturbo_lpips.pth`;
2. существует ли папка FFHQ и действительно ли в ней больше `source1.length` PNG-файлов;
3. существует ли папка LSDIR с изображениями;
4. есть ли в val `lq` и `gt` хотя бы `16` совпадающих имен файлов;
5. может ли машина достучаться до Hugging Face для автозагрузки `sd-turbo` и, при необходимости, metric weights.

### Шаг 6. Короткий пробный запуск

Нужно сделать короткий старт штатным entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 --nnodes=1 main.py --cfg_path ./configs/sd-turbo-sr-ldis.yaml --save_dir ./save_dir_smoke
```

Цель smoke test:

- пройти `build_dataloader`;
- пройти `build_model`;
- убедиться, что SD-Turbo подтягивается корректно;
- дойти хотя бы до первых итераций без ошибок путей и отсутствующих файлов.

### Шаг 7. Полный train

После успешного smoke test запускать обычный train в нужной GPU-конфигурации.

Базовая форма команды из `README.md`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --save_dir [Logging Folder]
```

Если validation включен, нужно помнить:

- первый checkpoint interval вызовет validation;
- в этот момент будут созданы `pyiqa` metrics;
- возможные проблемы сети проявятся именно там, даже если первые train-итерации уже идут.

## Что не нужно делать

- не нужно переписывать dataset pipeline;
- не нужно переносить LPIPS checkpoint в другие директории, если путь уже задан в YAML;
- не нужно вручную адаптировать Python-код под другую файловую структуру, если достаточно обновить `root_path` и `image_path`;
- не нужно скачивать только один `sd_turbo.safetensors` для офлайн-режима и ожидать, что этого достаточно: issue `#16` показывает, что нужен именно локальный HF-style cache.

## Короткий чек-лист готовности

- LPIPS checkpoint лежит в `weights/vgg16_sdturbo_lpips.pth`.
- В FFHQ файлов строго больше, чем `source1.length`.
- В LSDIR существует папка с PNG, соответствующая `image_path`.
- Validation `lq` и `gt` содержат совпадающие имена файлов.
- В каждой val-папке как минимум `16` изображений.
- В YAML обновлены только пути и необходимые численные параметры.
- Есть доступ к Hugging Face для автоматического скачивания `sd-turbo` и, при необходимости, весов `pyiqa`.

## Самый короткий практический вывод

Чтобы `train` заработал с `configs/sd-turbo-sr-ldis.yaml` без переписывания кода, нужно сделать ровно следующее:

1. скачать `vgg16_sdturbo_lpips.pth` в `weights/`;
2. скачать FFHQ и LSDIR в удобные локальные директории;
3. скачать validation-пары `lq/gt`;
4. прописать эти пути в YAML;
5. убедиться, что для FFHQ выполняется условие `count > length`;
6. сделать короткий smoke run;
7. после этого запускать основной train.

Если после этого train не стартует, следующая проверка должна быть не по коду, а по одному из трех источников:

- неверный путь в YAML;
- недостаточное количество файлов в датасете;
- отсутствие нужного веса или сетевого доступа в момент первой загрузки модели/метрик.
