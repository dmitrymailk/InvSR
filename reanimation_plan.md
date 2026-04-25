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

## Рабочий скрипт подготовки RealSR

Нужен не разбор неудачных вариантов, а один воспроизводимый путь. Он теперь вынесен в отдельный скрипт:

```text
prepare_realsr_v3_working_dataset.sh
```

Что нужно заранее:

1. положить архив RealSR как `data/RealSR (ICCV2019).tar.gz`;
2. положить `weights/vgg16_sdturbo_lpips.pth`;
3. дальше запускать только два скрипта подряд.

Команды:

```bash
cd /code/auto_remaster/sandbox/InvSR
./prepare_realsr_v3_working_dataset.sh
./run_train_realsr_v3_x4_single_gpu.sh
```

Что делает `prepare_realsr_v3_working_dataset.sh`:

1. если raw RealSR еще не распакован, распаковывает `data/RealSR (ICCV2019).tar.gz` в `data/RealSR-raw`;
2. собирает `data/RealSR-prepared/train_hr` из symlink-ов на все `*_HR.png` из:
   - `Canon/Train/2`
   - `Canon/Train/3`
   - `Canon/Train/4`
   - `Nikon/Train/2`
   - `Nikon/Train/3`
   - `Nikon/Train/4`
3. собирает `data/RealSR-prepared/val_gt_x4_native_hr_mod8` и `data/RealSR-prepared/val_lq_x4_native_lr_mod8` из `Canon/Test/4` и `Nikon/Test/4`;
4. для validation делает правильную геометрию:
   - `GT` режется до `mod8`
   - `LQ` режется до точного `x4`-соответствия `GT`
5. проверяет итоговые размеры набора:
   - `train_hr = 505`
   - `val = 30` пар.

Итоговые рабочие пути после запуска скрипта:

```text
data/RealSR-prepared/train_hr
data/RealSR-prepared/val_lq_x4_native_lr_mod8
data/RealSR-prepared/val_gt_x4_native_hr_mod8
```

Именно эти пути уже зашиты в `configs/sd-turbo-sr-ldis-realsr.yaml`, поэтому после подготовки датасета можно сразу запускать train.

### 6. Какой конфиг в итоге стал рабочим

Для real-data запуска использовался отдельный конфиг:

```text
configs/sd-turbo-sr-ldis-realsr.yaml
```

Ключевые настройки, которые были важны:

- train path:
	- `data.train.params.data_source.source1.root_path: data/RealSR-prepared/train_hr`
- validation paths:
	- `data.val.params.dir_path: data/RealSR-prepared/val_lq_x4_native_lr_mod8`
	- `data.val.params.extra_dir_path: data/RealSR-prepared/val_gt_x4_native_hr_mod8`
- ускорение первого прогона:
	- `train.batch: 2`
	- `validate.batch: 1`
	- `train.save_freq: 100`
- экономия памяти и времени:
	- `sd_pipe.enable_grad_checkpoint: False`
	- `sd_pipe.enable_grad_checkpoint_vae: False`
	- `discriminator.enable_grad_checkpoint: False`

Почему `validate.batch: 1` обязательно:

- validation-картинки имеют разное пространственное разрешение;
- при `batch > 1` dataloader пытается их `stack`-нуть и падает.

### 7. Команда запуска, которая реально заработала

Рабочий entrypoint был вынесен в отдельный скрипт:

```text
run_train_realsr_v3_x4_single_gpu.sh
```

Запуск:

```bash
cd /code/auto_remaster/sandbox/InvSR
./run_train_realsr_v3_x4_single_gpu.sh
```

Что делает скрипт:

- активирует `.venv`;
- запускает `main.py` через `torch.distributed.run` на `1` GPU;
- использует `configs/sd-turbo-sr-ldis-realsr.yaml`;
- создает отдельный timestamped `save_dir`, чтобы повторы запуска не конфликтовали по логам и чекпоинтам.

### 8. Как понять, что все действительно заработало

Успешный запуск выглядит так:

1. создается train dataset на `505` изображений;
2. создается val dataset на `30` пар;
3. train проходит шаг `100`;
4. после этого validation не падает;
5. в логе появляются строки вида:

```text
Validation Metric: PSNR=24.98, LPIPS=0.5132...
Validation Metric: PSNR=25.04, LPIPS=0.5444...
Validation Metric: PSNR=24.76, LPIPS=0.5417...
```

6. появляются checkpoint-файлы:

```text
ckpts/model_100.pth
ckpts/model_200.pth
ckpts/model_300.pth
ema_ckpts/ema_model_100.pth
ema_ckpts/ema_model_200.pth
ema_ckpts/ema_model_300.pth
```

Это и было критерием, что проблема реально решена, а не просто замаскирована стартом без валидации.

### 9. На какие картинки смотреть, чтобы понять, что обучение уже можно останавливать

Смотреть стоит не на `images/train`, а на `images/val` внутри конкретного run directory.

Практический путь такой:

```text
save_dir/realsr_v3_x4_single_gpu/<run>/<timestamp>/images/val
```

В этой папке самые полезные файлы такие:

- `LQ-1.png` ... `LQ-7.png` — входные low-quality изображения;
- `GT-1.png` ... `GT-7.png` — ground truth для тех же validation-примеров;
- `x0-progress-1.png` ... `x0-progress-7.png` — главные картинки для оценки качества SR;
- `sample-progress-1.png` ... `sample-progress-7.png` — вспомогательные картинки, по ним удобно видеть, не разваливается ли sampling.

Что важно понимать про эти файлы:

- логгер пишет их из validation, а не из train;
- при текущем `validate.log_freq: 4` и `validate.batch: 1` в лог попадают `7` validation-примеров за один validation pass;
- `x0-progress-N.png` содержит всю цепочку по inference steps, поэтому смотреть нужно в первую очередь на правую, финальную часть картинки, а не на первые промежуточные кадры.

На практике основной набор для просмотра такой:

1. открыть рядом `LQ-N.png`, `x0-progress-N.png` и `GT-N.png`;
2. в `x0-progress-N.png` смотреть на последний шаг справа;
3. сравнивать этот последний шаг с `GT-N.png`, а не только с `LQ-N.png`.

Признаки, что модель уже научилась апскейлить нормально и можно думать об остановке:

- на последних шагах `x0-progress-N.png` уже восстанавливаются устойчивые мелкие структуры, а не случайная каша;
- контуры становятся четче, но без двойных границ;
- не появляются светлые или темные ореолы вокруг контрастных границ;
- текстуры выглядят естественно и повторяемо на нескольких validation-примерах подряд;
- лицо, надписи, мелкие линии, кирпич, листья, провода и другие высокочастотные детали перестают заметно улучшаться от одной validation-точки к следующей;
- `PSNR` и `LPIPS` в логе выходят на плато одновременно с визуальным плато на `x0-progress-N.png`.

Признаки, что training уже пора останавливать, даже если loss еще меняется:

- последние validation-превью почти не отличаются между несколькими checkpoint-ами подряд;
- новые checkpoint-и начинают добавлять резкость без реального прироста деталей;
- появляются ringing/halo artifacts по краям;
- появляются повторяющиеся искусственные текстуры, которых нет в `GT`;
- на части примеров изображение становится визуально агрессивнее, хотя метрики уже не улучшаются.

На какие именно сюжеты смотреть в первую очередь:

- изображения с тонкими границами и диагоналями;
- мелкие повторяющиеся текстуры;
- участки с текстом, решетками, проводами, ветками;
- яркие контрастные переходы, где быстро проявляются halo и oversharpening.

Если нужен короткий рабочий критерий остановки, то он такой:

1. дождаться, пока `x0-progress-1..7.png` станут визуально стабильными на нескольких соседних validation;
2. убедиться, что финальный кадр справа уже близок к `GT`, а не просто стал более резким, чем `LQ`;
3. остановить обучение, когда дальнейшие checkpoint-и перестают добавлять реальные детали и начинают только усиливать артефакты.

### 10. Сколько, вероятно, обучал автор и как это соотносится с нашим запуском

Точного ответа в открытых материалах автора нет.

Что удалось установить по репозиторию и model card:

- в `README.md` автор показывает только схему запуска на `4` GPU;
- точное число часов или дней обучения автор не публикует;
- в model card указано, что основной train делался на `LSDIR + 20K FFHQ`;
- в оригинальном upstream-конфиге у автора стоит `iterations: 100000`.

То есть достоверно известно следующее:

1. автор рассчитывал на полноценный multi-GPU train;
2. авторский основной train — это не `RealSR-only`, а большой synthetic/realistic mix через `LSDIR + FFHQ`;
3. публично доступного значения вида `training took N hours on GPU X` нет.

По оригинальному конфигу автора training budget такой:

- global `batch: 64`;
- `microbatch: 16`;
- `iterations: 100000`;
- `save_freq: 5000`;
- validation на `16` изображениях;
- запуск через `torchrun --nproc_per_node=4`.

Важно: в `trainer.py` batch делится на число GPU, то есть `train.batch` у них задает global batch, а не per-GPU batch.

Практически это означает:

- у автора при `4` GPU effective per-GPU batch был `16`;
- global sample budget автора за полный schedule был примерно `64 * 100000 = 6.4M` train samples.

Наш текущий real-data запуск отличается радикально.

Текущие параметры нашего рабочего конфига:

- `batch: 2`;
- `microbatch: 1`;
- `iterations: 100000`;
- `save_freq: 100`;
- validation на `30` изображениях;
- запуск на `1` GPU;
- train dataset: `505` кадров из `RealSR`, а не `LSDIR + FFHQ`.

Это дает такую разницу относительно автора:

- по global batch мы меньше в `32` раза: `2` против `64`;
- по полному sample budget мы тоже меньше примерно в `32` раза: `0.2M` против `6.4M` samples за `100000` итераций;
- validation/checkpoint у нас происходят в `50` раз чаще: каждые `100` шагов вместо `5000`.

Последний пункт особенно важен: наш run намного сильнее тормозится не самим train, а частыми validation и сохранением тяжелых checkpoint-ов.

По текущим локальным логам видно две разные скорости:

- чистые `100` train-итераций занимают примерно `35-36` секунд;
- но wall-clock между `model_100.pth` и `model_200.pth` сейчас около `4` минут на каждые `100` шагов, потому что туда входит еще validation на `30` картинках и сохранение чекпоинтов.

Из этого следует грубая практическая оценка для нашего текущего режима:

- `1000` шагов: около `40` минут;
- `10000` шагов: около `6.5-7` часов;
- `100000` шагов: около `65-70` часов.

Это не эквивалент авторскому training budget. Это уже другой режим:

- меньше данных;
- сильно меньше batch;
- single-GPU вместо `4` GPU;
- очень частая validation ради контроля качества.

Практический вывод:

- не стоит пытаться интерпретировать наши `100000` шагов как «тот же самый train, что у автора»;
- наш текущий run — это скорее управляемый fine-tune/адаптация под `RealSR` с частым визуальным контролем;
- останавливать его нужно не по мысли «автор обучал столько-то часов», а по validation-картинкам и моменту, когда улучшения перестают быть видны.

Если цель — приблизиться именно к авторскому training regime, то нужно менять не только время запуска, но и сам setup:

- вернуть большой train set `LSDIR + 20K FFHQ`;
- вернуть большой global batch;
- уменьшить частоту validation/checkpoint хотя бы ближе к авторским `5000` шагам;
- запускать multi-GPU.

### 11. Полный минимальный рецепт с нуля

Если собрать все в один список, то рабочий порядок такой:

1. скачать `vgg16_sdturbo_lpips.pth` в `weights/`;
2. скачать RealSR archive из официального `csjcai/RealSR` и положить как `data/RealSR (ICCV2019).tar.gz`;
3. распаковать его в `data/RealSR-raw/`;
4. создать `data/RealSR-prepared/train_hr` как набор symlink-ов на все `*_HR.png` из `Canon/Nikon Train/{2,3,4}`;
5. создать `data/RealSR-prepared/val_lq_x4_native_lr_mod8` и `data/RealSR-prepared/val_gt_x4_native_hr_mod8` из raw `Canon/Nikon Test/4`;
6. использовать `configs/sd-turbo-sr-ldis-realsr.yaml`;
7. запускать через `./run_train_realsr_v3_x4_single_gpu.sh`.

Коротко: train починил не новый Python-код, а правильный real-data layout.

Конкретно сработало вот это:

- train собирается из `HR`-кадров raw RealSR;
- validation собирается из настоящих `LR4/HR` пар raw RealSR;
- `GT` в validation дополнительно обрезается до `mod8`, а `LQ` до точного `x4`-соответствия.

Именно после этого validation перестал падать и начались штатные сохранения checkpoint-ов.
