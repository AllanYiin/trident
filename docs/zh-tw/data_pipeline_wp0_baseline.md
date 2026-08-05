# WP0：資料管線基準與舊行為盤點

狀態：完成。

## 完成條件

- 舊、新引擎均執行 image classification、detection、pre-tokenized text 三種 workload。
- 每組至少重複三次，保存環境、吞吐、batch latency 與 peak RSS 增量。
- 建立 legacy/new 幾何語意、文字容器與尾批次行為的 characterization tests。
- 明確記錄舊機制沒有 HF streaming 與 batched tokenizer collator，而不是以替代測試掩蓋缺口。
- 基準結果可由獨立 validator 驗證完整性。

## 固定命令

```powershell
python benchmarks\benchmark_data_pipeline_wp0.py `
  --samples 512 --batch-size 16 --repeats 3 --workers 2 `
  --memory-budget-mb 256 `
  --output benchmarks\results\wp0_baseline_windows.json

python benchmarks\validate_wp0.py benchmarks\results\wp0_baseline_windows.json
python -m pytest tests\test_data_pipeline_wp0_characterization.py -q
```

## 本機基準摘要

環境：Windows、Python 3.10.10、NumPy 2.2.6、PyTorch backend。完整機器資訊與每次結果存於
`benchmarks/results/wp0_baseline_windows.json`。

| Workload | Legacy samples/s | Pipeline samples/s | Legacy p50 batch | Pipeline p50 batch |
|---|---:|---:|---:|---:|
| Image | 1,294 | 6,911 | 11.96 ms | 2.20 ms |
| Detection | 699 | 5,551 | 22.88 ms | 2.75 ms |
| Pre-tokenized text | 1,344 | 136,515 | 11.86 ms | 0.10 ms |

這是 synthetic characterization，不代表磁碟 decode、遠端 streaming 或 GPU 訓練端到端結果；後續 WP5、
WP6 必須另外量測，不能以這張表宣稱整體訓練提升。

## 舊行為決策

| 行為 | 觀察 | 重構決策 |
|---|---|---|
| paired resize | image 與 xyxy bbox 可同步縮放 | 必須保留幾何語意 |
| backend layout | PyTorch backend 在 Dataset 階段轉成 CHW | 新機制延後到 tensorizer，不保留過早轉換 |
| NumPy Unicode | 目前版本在 schema 推導時失敗 | 不相容；新機制保留 Python `str` |
| 首批讀取 | signature 更新會令第一個 batch 重複一次 | 視為 legacy defect，不保留 |
| 尾批次 | sampler 會循環資料開頭補滿 batch | 新預設保留 partial batch；需要固定尺寸時明示 `drop_last` 或專用 policy |
| HF streaming | 無原生整合 | WP2 必須新增並驗證 sharding |
| batched tokenizer | 無 collator 邊界 | WP4 必須在 batch 邊界整合 |

## 後續比較基線

- WP3：幾何結果需通過 characterization 中與舊 resize 相同的 bbox 語意。
- WP5：相同 synthetic configuration 的新管線中位吞吐不得低於本檔 pipeline baseline 的 95%。
- WP5：`memory_budget_mb` 必須可驗證，不得只記錄設定值。
- WP6：新增至少一組真實 PyTorch 與 TensorFlow 訓練 step benchmark。
- WP7：公開切換前，三種 workload 均不得低於 legacy 中位吞吐。
