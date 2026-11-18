# DEVNOTE

## SNAPSHOT — 當前狀態

最後更新：2026-11-19T01:34+08:00
交接狀態：verified

### 目標與授權範圍

修正 73e08be3 梯度審查列出的斷圖、正規化、optimizer 狀態及分散式更新問題。

### 目前進度

已修改 PyTorch／TensorFlow 的 loss、trainer、optimizer、GAN 梯度窗口、TensorFlow LSTM、舊 PPO 及原生語言模型路徑。測試和 CI 變更位於工作樹。

### 下一步

提交後可在 CI 另驗 Linux。完整 CycleGAN 舊入口及多機硬體組態仍需獨立整合驗證。

### DECISIONS

- 加權分類 loss 以有效權重正規化，其他自訂 mean loss 可透過 callback 設 accumulation_weight。零有效權重不建立空的 optimizer 更新。
- DDP 先執行 rank-local 計算，再共同確認是否成功並交換梯度。MoE 使用更新窗口的整體路由統計。
- rollout 從抽樣當下取得實際 log probability，GRPO／PPO 可對既有未重整的訓練策略做 importance weighting。

### LANDMINES

- MoE 的精確窗口正規化保留活躍 microbatch 的 forward graph，峰值記憶體可能上升。
- CycleGAN 的批尾結算已調整，完整 legacy CycleGAN 入口仍含既有初始化缺口。JAX、TPU、多機 NCCL 與遠端 CI 尚未執行。

### 證據索引

- trident/backend/model.py、trident/backend/pytorch_ops.py、trident/optims/pytorch_losses.py
- trident/optims/pytorch_trainer.py、trident/optims/tensorflow_trainer.py、trident/optims/trainers.py、trident/optims/*_optimizers.py
- trident/optims/language/runner.py、trident/optims/language/generation.py、trident/models/pytorch_causal_lm.py
- tests/test_legacy_gradients.py、tests/test_legacy_tf_gradients.py、tests/test_language_core.py、tests/test_language_runtime.py

### VALIDATION

- 完整 pytest 測試，PyTorch 主環境 — PASS：Windows Python 3.10.10，150 passed、1 skipped、27 warnings，耗時 187 秒。包含 CUDA AMP、雙 rank Gloo 單 token 與 MoE 一致性及 rank-local callback 例外。唯一 skip 為獨立 TensorFlow 測試模組，已另跑。
- PyTorch CPU 梯度回歸 — PASS：18 passed、1 skipped，CPU 模式跳過了 CUDA AMP。
- PyTorch 梯度回歸，CUDA 可用 — PASS：19 passed，包括 overflow 事件、中心參數更新及 WGAN penalty 的解析梯度。
- TensorFlow CPU 訓練與梯度回歸 — PASS：12 passed，包括不等長尾批、全域裁剪、六款 optimizer 的 momentum 和無效梯度處理，以及 LSTM 跨步梯度。
- 語法與 diff — PASS：27 個變更或新增 Python 檔經 AST 解析，git diff --check 通過。

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
