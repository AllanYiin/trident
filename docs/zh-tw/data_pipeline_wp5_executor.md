# WP5：高吞吐 executor、恢復與記憶體驗收

WP5 將資料供應的併行與 backpressure 集中在 `Iterator`，來源、transform 與模型後端
不需要各自建立無界 queue。

## 完成項目

- ordered bounded prefetch，慢 worker 不改變 batch 順序。
- `executor="thread" | "process"`；process worker 用 initializer 建立本地 pipeline，避免
  每筆重建 executor。
- `persistent_workers=True` 可跨 epoch 重用 executor，並提供 `close()`／context manager。
- tensorize、pin memory 與 device transfer 在主程序 delivery 階段執行，避免 GPU tensor IPC。
- map-style deterministic shuffle；iterable-style 提供有界 shuffle buffer。
- `state_dict()` 保存 epoch、batch offset、seed、rank/world size 及可選 source state；可在
  epoch 中途從下一個 batch 恢復。
- 記憶體預算先量測第一個 batch，再決定有效 prefetch 深度；單一 batch 超限立即失敗。

## 驗收

```powershell
python -m pytest tests\test_data_pipeline_wp5_executor.py -q
```

六項測試涵蓋 thread ordering、中途 resume、streaming shuffle 的 epoch/reproducibility、
persistent executor、Windows process executor，以及 oversized batch。process 測試需要作業系統
允許 multiprocessing named pipe；受限 sandbox 內的 `WinError 5` 不屬於管線失敗。

## 實務選擇

- I/O、OpenCV 等會釋放 GIL 的工作優先使用 thread；純 Python CPU transform 才用 process。
- streaming process executor 尚未開放，因來源 iterator 的跨程序所有權與 checkpoint 語意
  應由來源 adapter 明確處理，不隱式複製串流。
- 不可重入 streaming source 若沒有 `state_dict/load_state_dict`，resume 會從 epoch 起點重播並
  跳過已完成 batch；可重入 factory 與 HF iterable source 適用。
