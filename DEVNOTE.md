# DEVNOTE

## SNAPSHOT — 當前狀態

最後更新：2026-11-18T17:23+08:00
交接狀態：verified

### 目標與授權範圍

檢視 Trident 目前程式碼、發行與測試，評估是否值得繼續維護。

### 目前進度

已評估 master 5cad1b99。資料管線、繁中增強與 PyTorch 訓練有可執行用途。原始碼預設隔離建置與兩個舊視覺模型入口有可重現缺陷。

### 下一步

若採納聚焦維護，優先修正隔離建置與 ResNet50 的參數凍結行為，將實際使用的模型與資料流程納入 CI。是否擴大公開支援，仍待確認活躍下游。

### DECISIONS

- 評估建議優先維護資料管線、現有 PyTorch 訓練與仍有使用者的舊模型。維護範圍尚待專案擁有人決定。

### LANDMINES

- Windows 與微型語料驗證涵蓋正確性。Linux、JAX、大模型效能、活躍下游與維運人力仍待驗證。
- 前一輪 JPEG 稽核工具已提交，實際訓練資料尚未提供，仍待依來源修復或排除檔案。

### 證據索引

- setup.py、requirements.txt、.github/workflows/language.yml、PyPI tridentx 0.7.12
- trident/data/pipeline/transforms.py、trident/data/text_transforms.py、tests/test_language_core.py、examples/language/quickstart.py
- trident/models/pytorch_resnet.py、trident/models/tensorflow_resnet.py、trident/models/__init__.py
- git log、git rev-parse HEAD refs/remotes/origin/master、git lfs ls-files

### VALIDATION

- Windows、Python 3.10.10，本機既有環境執行 python -m pytest tests -q — FAIL：107 passed、2 failed。Dense 與 MoE 的 Hugging Face 匯出測試在 accelerate 匯入時失敗。
- 使用專案記錄的相依套件目錄與可寫的 Numba 快取，完整重跑 tests — PASS：同一輪測試 109 passed、15 warnings，耗時 142 秒。兩個 Hugging Face 匯出案例均通過。
- 從 HEAD 建置並安裝 wheel 後執行離線 quickstart — PASS：本機 setuptools 80.9.0 使用 --no-build-isolation 成功。CPU fp32 在範例四句語料上更新 20 次，loss 5.5827 降至 2.2951。
- 從 HEAD 使用 pip 預設隔離建置 wheel — FAIL：setup.py 第 4 行匯入 pkg_resources 失敗，建置需求階段中止。
- PyTorch 2.7.1 的 ResNet50 訓練參數與 TensorFlow 2.20.0 的模型入口 — FAIL：PyTorch 兩類輸出正常，但 pretrained=False、freeze_features=False 時 161 個參數張量僅 fc.weight 與 fc.bias 可訓練。TensorFlow 可執行張量運算，Trident 模型套件匯入時發生 resnet 循環匯入。
- 發行日期與安裝資訊核對 — PASS：PyPI 的 0.7.12 於 2026-06-10 發行，僅列 visualize、tests 兩個 extras。工作樹同版號已加入 llm 等 extras。

# HISTORY


---

## [2026-09-29T23:01+08:00] Session DEVNOTE checkpoint — first-response-without-devnote

<!-- auto-devnote-id: 069febfccf707f87b385 -->

- **宿主**：codex
- **事件**：Stop / first-response-without-devnote
- **Session**：`01a0edae-f4b6-71b3-a554-ff98e2e5e4f8`
- **證據來源**：event:last_assistant_message
- **狀態**：已由稀疏觸發策略建立；仍需 `devnote` skill 做語意精煉

### 自動保留內容

> 以下只有宿主提供的最後一則 assistant 回覆，尚未經 `devnote` skill 語意精煉：
> 
> {"suggestions":[]}


## [2026-11-13T23:43+08:00] JPEG 資料稽核與錯誤定位

<!-- devnote-review:8eb9d58df70f32b3112d270a -->

新增 JPEG 逐檔稽核與測試，改善解碼失敗時的錯誤定位；實際資料集尚待檢查。精煉舊空白 checkpoint，未推定其代表已完成任務。


## [2026-11-18T17:23+08:00] 維護價值評估與使用入口驗證

<!-- devnote-review:b9a7e4a5f5fcd112002564d3 -->

已核對前次按主題整理的 10 筆提交。本地 HEAD 與 origin/master 追蹤參照同指 5cad1b99，Git LFS 列出 11 檔。本輪驗證資料與訓練功能、模型行為及發行落差，提出維護範圍建議。
