# WP6：PyTorch、TensorFlow 與 TrainingPlan 端到端驗收

WP6 保持核心管線後端無關，僅在 collate 後或 framework 邊界轉 tensor。

## 完成項目

- `Iterator.to_torch_dataloader()`：以 `IterableDataset` 接入原生 DataLoader；batch size 與
  workers 仍由 pipeline 擁有，避免雙重 batching／multiprocessing。
- `Iterator.to_tensorflow_dataset()`：建立 `tf.data.Dataset`，可自動推導 output signature，
  將 batch 第一維放寬為 `None`，因此短尾批次有效；支援 ragged tensor 與 prefetch。
- `TorchTensorizer`／`TensorFlowTensorizer` 保留 `Batch` mapping；PyTorch 支援 ragged values
  與 row splits。
- `TrainingPlanAdapter` 將字串欄位映射集中轉為舊 TensorSpec-like key，核心 Dataset／Iterator
  不依賴 `traindata.data_template`。

## 驗收

```powershell
python -m pytest tests\test_data_pipeline_wp6_frameworks.py -q
```

四項端到端測試實際完成 PyTorch forward/backward/optimizer step、TensorFlow
GradientTape/update 與短尾 batch、TensorFlow tensorizer，以及 Trident TrainingPlan loader
registration 和 input/target contract。兩個後端在目前環境皆實際安裝並執行，非 mock。
