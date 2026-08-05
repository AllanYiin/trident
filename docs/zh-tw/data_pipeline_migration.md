# 建模資料管線遷移指南

新核心位於 `trident.data.pipeline`。舊的 `trident.data.dataset.Iterator` 與
`trident.data.data_provider.DataProvider` 在過渡期仍存在；新程式應使用明確的新 namespace，
避免同名類別被 wildcard import 混淆。

## 建議遷移順序

1. 先為 sample 定義字串欄位名稱與 `DatasetSchema`，停止依賴 TensorSpec 物件作為資料 key。
2. 用 `Dataset.from_columns/from_iterable/from_folder/from_huggingface/from_legacy` 包住來源。
3. 將 paired image transforms 換成 `GeometryCompose`，並為空間欄位標上 `FieldSpec.kind`。
4. 將 dynamic padding／tokenization 移到 collator；模型邊界才 tensorize。
5. 以 `DataProvider.iter()` 設定 workers、prefetch、memory budget、rank/world size。
6. 先透過 `TrainingPlanAdapter` 接舊 Trainer；新訓練碼可直接轉 PyTorch DataLoader 或
   TensorFlow Dataset。

## 介面對照與刻意不相容處

| 舊介面／行為 | 新介面 | 遷移注意事項 |
|---|---|---|
| `Iterator(data=..., label=...)` | 每個 sample 是具名 mapping | 不再靠位置猜 input/target |
| TensorSpec object keys | Python string keys + schema | 舊 Trainer 轉換集中在 adapter |
| NumPy unicode/object batch | 原生 `list[str]`／`RaggedBatch` | tokenizer 可直接吃 Python 文字 |
| `paired_transform_funcs` | `GeometryCompose` | 每次只抽一次幾何參數並可 replay |
| 隱藏 `memory_cache` | `SamplePool(capacity=...)` | cache 容量必須顯式 |
| cyclic tail fill | 短尾 batch 或 `drop_last=True` | 不再用 epoch 開頭樣本補尾 |
| signature 更新會重取首批 | Iterator state checkpoint | 恢復點是明確 batch offset |
| streaming shuffle 不明確 | bounded shuffle buffer | 設定 `shuffle_buffer_size` |

## 過渡用法

```python
from trident.data.pipeline import Dataset, migrate_legacy_provider

dataset = Dataset.from_legacy(old_iterator, schema=schema)
new_provider = migrate_legacy_provider(old_provider,
                                       schema_by_split={"train": schema})
legacy_training_loader = new_provider.for_trident_trainer(
    "train", batch_size=32, workers=2)
```

`migrate_legacy_provider()` 會發出 `DeprecationWarning`，因為它只用於搬遷；新專案應直接建立
新 `DataProvider`。本次重構沒有改寫舊類別內部，因此現有舊程式不會被全域切換。

## 上線策略

- 先在相同 sample ids 上比較 legacy/new baseline、增強 record 與模型 loss。
- worker 數變更時，確認同 epoch/rank/sample id 的增強輸出一致。
- production 前依實際影像大小量測 batch footprint，再設定 `memory_budget_mb`。
- process executor 的 transforms/collator 必須可 pickle；否則使用 thread。
- checkpoint 必須連同 Iterator state 一起保存，分散式恢復時 rank/world size 必須相同。
