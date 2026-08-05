# 建模資料管線（影子新核心）

新資料管線位於 `trident.data.pipeline`。目前不會取代舊的
`trident.data.dataset.Iterator` 與 `DataProvider`；完成實際 workload 驗證前，兩者可並存。

## 核心資料流

```text
Dataset → sample transform → group transform → collator/tokenizer
        → batch transform → tensorizer/pin/transfer → Trainer
```

- `Dataset` 只處理資料來源與樣本存取，支援 map-style 與 streaming。
- `Iterator` 管理 batching、rank sharding、worker、bounded prefetch 與 epoch seed。
- `DataProvider` 管理 train/valid/test split。
- 樣本及 batch 都使用字串 key；`FieldSpec` 與資料值分離。
- Python 字串及 ragged data 不會為了 batching 被強制轉成 NumPy 字串或 object array。

## 一般使用方式

```python
from trident.data.pipeline import Dataset, DataProvider

provider = DataProvider(
    train=Dataset([{"input": [1, 2], "target": 1}]),
)

for batch in provider.iter("train", batch_size=32, workers=2):
    print(batch)
```

## Hugging Face 與 tokenizer

```python
from trident.data.pipeline import Dataset, DataProvider, TokenizerCollator

train = Dataset.from_huggingface(hf_dataset["train"])
provider = DataProvider(train=train)

iterator = provider.iter(
    "train",
    batch_size=32,
    collate=TokenizerCollator(
        tokenizer,
        text_field="text",
        padding="longest",
        truncation=True,
    ),
)
```

`Dataset.from_huggingface()` 不會把整份 Dataset 轉成 list。HF streaming dataset
會保留 iterable-style；此時應設定 `steps_per_epoch`。可使用 HF 來源端 shuffle，或由
`Iterator(shuffle=True, shuffle_buffer_size=...)` 提供可重現的有界 shuffle。

## 幾何同步

```python
from trident.data.pipeline import (
    DatasetSchema, FieldSpec, GeometryCompose,
    RandomCrop, RandomHorizontalFlip, Resize,
)

schema = DatasetSchema([
    FieldSpec("image", kind="image", layout="HWC"),
    FieldSpec("mask", kind="mask", layout="HWC"),
    FieldSpec("boxes", kind="bbox", coordinate_format="xyxy"),
])

transform = GeometryCompose([
    Resize((640, 640)),
    RandomCrop((512, 512)),
    RandomHorizontalFlip(0.5),
])
```

每個 transform 只抽取一次參數，再依 `FieldSpec.kind` 同步套用到 image、mask、bbox、
keypoints、landmarks 與 polygon。參數存在每個 sample 的 `TransformContext`，不存放於共用
transform instance，因此可安全用於多 worker，且 worker 數改變時仍維持相同增強結果。

## 接入既有 TrainingPlan

過渡期使用 `TrainingPlanAdapter`：

```python
schema = DatasetSchema([
    FieldSpec("input", kind="data"),
    FieldSpec("target", kind="label", metadata={"role": "target"}),
])

provider = DataProvider(train=Dataset(rows, schema=schema))
legacy_loader = provider.for_trident_trainer(
    "train",
    batch_size=32,
    workers=2,
)

plan.with_data_loader(legacy_loader)
```

所有舊式 TensorSpec-keyed batch 行為都封裝在 adapter；不要在新 Dataset 或 Iterator 中依賴
`traindata.data_template`。

## 記憶體與吞吐

- `prefetch_batches` 是有界 batch 數量，預設為 2。
- `memory_budget_mb` 會拒絕單一 batch 已超出預算的設定，並依實測 batch footprint
  收斂 prefetch 深度。
- `workers=0` 是除錯與 baseline；`executor="thread"` 適合 I/O、OpenCV 或會釋放 GIL 的
  工作，純 Python CPU transform 可用 `executor="process"`。
- `persistent_workers=True` 可跨 epoch 重用 worker；使用完畢請 `close()`。
- `TorchTensorizer`、`TensorFlowTensorizer` 只在 collate 後轉 tensor，避免 Dataset 綁定後端。

可執行 smoke benchmark：

```powershell
python benchmarks\benchmark_data_pipeline.py --workload all --workers 2
```

## 舊介面遷移對照

| 舊機制 | 新機制 |
|---|---|
| `Iterator(data=..., label=...)` | 每筆 sample 直接包含具名 input/target 欄位 |
| `image_transform_funcs` | `Iterator(..., transforms=[...])` |
| `paired_transform_funcs` | `GeometryCompose` 與 `FieldSpec.kind` |
| `batch_transform_funcs` | `batch_transforms` |
| `TextSequenceDataProvider.dynamic_padding` | `PaddingCollator` 或 `TokenizerCollator` |
| `memory_cache` | 顯式、有容量上限的 `SamplePool`／`GroupTransform` |


## 公開名稱與版本

新 API 的穩定入口是 `trident.data.pipeline`，目前 `PIPELINE_API_VERSION == 1`。若需從
`trident.data` 存取，使用不與舊類別衝突的 `PipelineDataset`、`PipelineIterator`、
`PipelineDataProvider`。詳細差異見 [遷移指南](data_pipeline_migration.md)。