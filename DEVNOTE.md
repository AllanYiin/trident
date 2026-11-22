# DEVNOTE

## SNAPSHOT — 當前狀態

最後更新：2026-11-22T17:07+08:00
交接狀態：verified

### 目標與授權範圍

在既有 with_learning_rate_scheduler 提供四種暖身策略與更新預算比例，完成跨 trainer 驗證。

### 目前進度

四種策略、舊整數 epoch 與線性 checkpoint 相容功能已在工作樹。已補上 optimizer 配置後新增參數群組時的雙重暖身檢查及回歸測試。

### 下一步

暖身功能驗收完成。ToolRegistry 子程序逾時在原始 HEAD 也能重現，可另案處理。

### DECISIONS

- LinearWarmup、CosineWarmup、ExponentialWarmup 和 ConstantWarmup 共用每個 optimizer 的成功更新計數。
- from_ratio 依明示的 optimizer 更新預算向上取整。固定暖身維持指定倍率直到下一步交接。
- 設定時、首次更新前及群組數改變時檢查內建暖身衝突。一般模型自訂訓練 checkpoint 同時保存 optimizer 和 scheduler，語言模型由訓練 checkpoint 自動保存。

### LANDMINES

- 舊 test/ 示範另需 vocabs.txt。
- Windows 環境中 ToolRegistry 的加法工具逾時，工作樹及原始 HEAD 單獨執行皆同樣失敗。

### 證據索引

- trident/callbacks/warmup.py、trident/backend/model.py、trident/optims/language/trainer.py
- tests/test_legacy_gradients.py、tests/test_legacy_tf_gradients.py、tests/test_language_core.py
- docs/zh-tw/quickstart.md

### VALIDATION

- Windows 完整 tests/ 與基準比對 — FAIL：170 passed、1 skipped、1 failed。唯一失敗是 test_tool_execution_timeout_and_registry 的 10 秒逾時，已在獨立解出的 HEAD 同條件重現。暖身與 checkpoint 測試通過。
- TensorFlow CPU — PASS：tests/test_legacy_fit.py 和 tests/test_legacy_tf_gradients.py 共 15 passed。
- 動態參數群組暖身衝突 — PASS：修正前，首次更新前及訓練中加入帶 warmup 的 AdamW 參數群組，兩個案例均未報錯。回歸測試檢查錯誤發生時各組 LR 與成功更新計數。
- 語法與差異格式 — PASS：變更程式與測試 py_compile、git diff --check 通過。

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


## [2026-11-19T00:51+08:00] 梯度傳遞及更新正確性審查

<!-- devnote-review:92d8f0317c89331fc4ed1651 -->

核對前五項修正已提交於 73e08be3。本輪審查重現梯度斷圖、累積比例、optimizer 狀態與分散式正規化等問題，並確認 GAN 與 CycleGAN 每批提早結算的程式路徑。


## [2026-11-19T01:34+08:00] 修正梯度斷圖、累積與 optimizer 更新

<!-- devnote-review:49d90948c522d3727a66d45d -->

修正上一輪梯度審查的主要缺陷，新增參數與實際梯度的回歸，包括少量有效 token 的 Gloo、MoE 窗口、rank-local 失敗與行為策略機率。


## [2026-11-22T16:37+08:00] 新增按 optimizer 更新計步的共用 LR warmup

<!-- devnote-review:5e1e00b371853ad1d84ce7fb -->

移除 trainer 與舊 RL policy 的重複暖身程式，加入參數群組倍率、排程交接、雙重暖身檢查及中斷續訓驗證。


## [2026-11-22T16:43+08:00] 新增餘弦、指數、固定暖身及更新預算比例

<!-- devnote-review:64a91dd21652dcd3275595f0 -->

保留線性與整數 epoch 入口，新增三種曲線及 from_ratio。跨後端驗證實際 LR、曲線交接與續訓。


## [2026-11-22T17:07+08:00] 驗收四種暖身策略並修正新增參數群組的衝突檢查

<!-- devnote-review:b229e09680d6bfc5471ce54b -->

以兩個失敗案例重現 AdamW 在新增群組後可繞過雙重暖身檢查，修正後執行完整 tests/ 與獨立 TensorFlow 測試。
