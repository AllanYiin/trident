# WP2：資料來源 adapters 與 Hugging Face 驗收

WP2 讓新 `Dataset` 直接包裝常見來源，不先把整份資料轉成 list 或批次 NumPy
結構。map-style 保留隨機存取；iterable-style 保留串流與來源端 sharding。

## 完成項目

- `ColumnarSource`：等長欄位映射，逐筆建立 sample，不複製底層欄位。
- `IterableFactorySource`：每次迭代重新呼叫 factory，可安全重啟 epoch。
- `FolderSource`：只建立路徑索引，內容由 loader 延遲讀取。
- `LegacySource`：延遲轉接舊 Trident data/label/unpair 欄位。
- `HuggingFaceDataset`：支援 map/iterable dataset、`with_format`、
  `with_transform`、`select_columns`、批次索引、shuffle、shard 與 `set_epoch`。
- `Iterator` 將 epoch 傳給來源，並以來源 sample id 建立可重現 transform context。

## 驗收

```powershell
python -m pytest tests\test_data_pipeline_adapters.py -q
```

目前為 6 項測試全數通過，包含 HF-compatible map／streaming doubles、不同 rank
不重疊 shard、epoch forwarding，以及 lazy loading。實際 `datasets.Dataset` 的
in-memory smoke test 亦不需要先物化為 Python list。

## 邊界

遠端 HF streaming 的網路重試、cache 配額與認證由 Hugging Face source 本身管理；
管線只保存 iterable 語意與 backpressure。跨來源 checkpoint/resume 在 WP5 驗收。
