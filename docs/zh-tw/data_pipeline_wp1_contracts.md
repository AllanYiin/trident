# WP1：資料契約與錯誤模型驗收

WP1 將樣本與批次的公開契約固定在 `Sample`、`Batch`、`FieldSpec` 與
`DatasetSchema`，避免舊管線以 NumPy 容器型態隱含欄位語意。

## 完成項目

- `FieldSpec` 描述 dtype、shape、variable axes、layout、座標格式、角色與 metadata。
- `DatasetSchema` 可驗證、投影、序列化及產生穩定 fingerprint。
- `Sample` 保留 `sample_id` 與 metadata；`Batch` 提供遞迴 `pin_memory()`／`to()`。
- source、transform、collation 與記憶體預算錯誤具有 stage、field、sample id 與 cause。
- Python 字串與 ragged 值不再被強制包成 NumPy `str_`／`object` array。

## 驗收

```powershell
python -m pytest tests\test_data_pipeline_contracts.py -q
```

契約測試涵蓋 schema round-trip、shape/dtype 驗證、Sample/Batch metadata、巢狀
device transfer 與結構化錯誤內容。
