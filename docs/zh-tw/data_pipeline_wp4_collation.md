# WP4：Tokenizer、ragged 與模型批次驗收

WP4 將「如何形成模型 batch」留在 collator，而不是要求 Dataset 把文字、變長序列或
巢狀資料塞進 NumPy `str_`／`object` array。

## 完成項目

- `default_collate`：固定尺寸 numeric 值堆疊；文字及真正 ragged 值保留 Python 容器。
- `PaddingCollator`：逐欄設定 pad value／axis，支援 multiple-of padding、length 與 mask。
- `RaggedBatch`／`RaggedCollator`：以 flat values + row splits 保存變長序列，無 object array。
- `FieldCollator`：每個欄位可使用不同策略，不要求整批採同一種表示。
- `TokenizerCollator`：每批只呼叫 tokenizer 一次，非文字欄位照常 collate。
- `Seq2SeqCollator`：用 `text_target` 一次 tokenize source/target，並將 label padding 改為
  loss ignore id（預設 `-100`）。
- `HuggingFaceCollator`：直接包裝 transformers／datasets 生態的既有 data collator。
- PyTorch／TensorFlow tensorizer 在 collate 後才轉換；PyTorch 保留 `RaggedBatch` 結構。

## 驗收

```powershell
python -m pytest tests\test_data_pipeline_wp4_collation.py -q
```

測試涵蓋 row splits、padding mask、field-specific policy、seq2seq 單次 batched tokenizer
呼叫、`-100` labels，以及 ragged 到 PyTorch tensor 的轉換。
