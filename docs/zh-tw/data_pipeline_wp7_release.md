# WP7：公開介面、遷移與最終驗收

WP7 以明確 namespace 發布新核心，不全域取代舊 DataProvider／Iterator。

## 完成項目

- 穩定入口 `trident.data.pipeline`，API 版本為 `PIPELINE_API_VERSION = 1`。
- `trident.data` 提供無衝突別名：`PipelineDataset`、`PipelineIterator`、
  `PipelineDataProvider`；舊同名類別語意不變。
- `migrate_legacy_provider()` 延遲包裝舊 train/valid/test 欄位並發出 DeprecationWarning。
- 完整繁中 migration guide，列出刻意不相容處、切換順序、rollback 邊界與 production 注意事項。
- vision geometry、text tokenizer、low-memory PyTorch 三個可直接執行範例。
- `benchmarks/validate_data_pipeline.py` 驗證公開 aliases、API 版本與 WP0～WP7 交付物。

## 最終驗收指令

```powershell
python -m compileall -q trident\data\pipeline examples\data_pipeline
python -m pytest tests -q
python benchmarks\validate_wp0.py benchmarks\results\wp0_baseline_windows.json
python benchmarks\validate_data_pipeline.py
python examples\data_pipeline\vision_geometry.py
python examples\data_pipeline\text_tokenizer.py
python examples\data_pipeline\low_memory_training.py
git diff --check
```

Windows process executor 測試需允許 multiprocessing named pipe。TensorFlow 依賴帶入的
`httplib2` pyparsing deprecation warnings 不影響資料管線測試結果。
