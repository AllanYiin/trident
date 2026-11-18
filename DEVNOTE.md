# DEVNOTE

## SNAPSHOT — 當前狀態

最後更新：2026-11-18T21:33+08:00
交接狀態：verified

### 目標與授權範圍

依上次審查修正前五項，驗證舊版訓練、資料切分、PyTorch checkpoint 與梯度累積，將重現案例加入 CI。

### 目前進度

已在工作樹修正五項。訓練失敗會傳回呼叫端並保留先前的 checkpoint，fit 使用獨立驗證資料，原生 nn.Module 可載入權重，每輪涵蓋尾批，FP32 與 AMP 的累積更新以實際批次權重計算。

### 下一步

本機修正已驗收。遠端 GitHub Actions 結果待這批變更推送後確認。

### DECISIONS

- PyTorch 新 .pth 與 .pth.tar 預設使用受限制的權重載入。載入可信舊序列化物件須顯式 trusted_legacy=True，需要匯出舊整模可用 save_full_model=True。
- 訓練以有限 epoch 迭代，保留舊 Iterator.next() 跨 epoch 補滿批次的契約。
- 以實際樣本 ID、儲存失敗前後檔案內容與數值更新檢查迴歸。

### LANDMINES

- GitHub Actions 的 Linux 執行待推送驗證。本機 TensorFlow 驗證環境為 CPU。
- 先前審查的串流續訓偏移、稀疏 token 的分散式正規化、視覺標註、ResNet50 凍結、TensorFlow ResNet 匯入與隔離建置仍待個別處理。

### 證據索引

- trident/optims/trainers.py、trident/backend/model.py、trident/data/samplers.py
- trident/backend/pytorch_checkpoint.py、trident/backend/checkpoint.py、trident/optims/pytorch_trainer.py
- tests/test_legacy_training.py、tests/test_legacy_fit.py、tests/test_training_save_guardrails.py
- .github/workflows/legacy-training.yml、README.md 與本機測試輸出

### VALIDATION

- Windows、Python 3.10.10，完整 python -m pytest tests -q — PASS：最終工作樹在專案相依環境 E:/trident-training-tests/deps 與可寫的 Numba 快取下，128 passed、26 warnings，耗時 163 秒。包含 PyTorch 2.7.1 CUDA AMP 梯度累積。
- CI 對應的 PyTorch CPU 回歸檔 — PASS：21 passed、1 skipped；CPU 模式只跳過 CUDA AMP，該案例另外在實體 CUDA 上通過。
- TensorFlow 2.20.0 CPU 的 fit() 與失敗保存測試 — PASS：4 passed。涵蓋分割驗證集、外部驗證集、無驗證集與失敗時保留 checkpoint。
- 數值及跨程序的獨立重現 — PASS：相同兩筆資料，大 batch 與累積後權重皆為 0.8。原生 nn.Linear 載入權重 9 後推論為 9。可信舊式全模檔在新程序可還原原生及 Trident 模型。
- 程式與 CI 結構檢查 — PASS：15 個變更或新增 Python 檔通過 AST 解析，CI YAML 與列出的測試檔均已核對，git diff --check 通過。
- 本機 wheel 的權重還原 — PASS：使用 --no-build-isolation 建置，wheel 包含兩個新 checkpoint 模組和 filelock 依賴。在工作樹以外載入該 wheel 的模組後，還原並推論得到 9。
- 遠端 GitHub Actions — NOT_RUN：變更在本機工作樹。本次未要求推送。

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


## [2026-11-18T21:33+08:00] 修正舊訓練流程的五項正確性問題

<!-- devnote-review:4d91025f23a43f412d6470c0 -->

已核對上一輪審查的重現條件。本輪修正例外與存檔、驗證切分、原生模型權重載入、epoch 尾批與梯度累積。新增 PyTorch 與 TensorFlow 的行為測試及 CI，保留既有歷史。
