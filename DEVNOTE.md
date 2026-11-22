# DEVNOTE

## SNAPSHOT — 當前狀態

最後更新：2026-11-23T05:14+08:00
交接狀態：verified

### 目標與授權範圍

提交 PyTorch 與 TensorFlow 的 loss、metric 與 data_feed 欄位綁定，並推送至 origin/master。

### 目前進度

兩個 trainer 已通過欄位對應與損失轉存回歸。

### 下一步

以遠端 master 的 Git 記錄確認此次功能提交。

### DECISIONS

- 沿用平坦欄位名稱，支援裝飾器預設與單次註冊覆寫。多個形狀相容的候選須明示來源。
- as_metric=True 記錄脫離梯度的原始 loss 數值，包含零權重的情況。

### LANDMINES

- 本機 Hugging Face 匯出測試因 accelerate 循環匯入而失敗。

### 證據索引

- trident/optims/field_binding.py
- tests/test_legacy_field_binding.py
- docs/zh-tw/quickstart.md
- .pytest_cache/field-binding-full.log

### VALIDATION

- PyTorch 欄位綁定、訓練、梯度與儲存回歸 — PASS：73 passed，雙輸入雙輸出、註冊順序與 loss 呼叫計數均通過。
- TensorFlow 欄位綁定、fit 與梯度回歸 — PASS：30 passed。
- Windows tests/ 完整回歸 — FAIL：194 passed、6 skipped、2 failed。兩個 Hugging Face 匯出案例失敗；獨立 Python 匯入亦可重現。

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


## [2026-11-23T00:32+08:00] 補上 JAX loss 模組並驗證三後端值與梯度

<!-- devnote-review:d51e73fa95cdef51d7ffda77 -->

將 34 個公開 loss 名稱接到三後端，補齊穩定二元 logits、Huber 及 Tversky，修正 TensorFlow 歸約與軟標籤、PyTorch GP 回傳值和 hinge 反傳。


## [2026-11-23T01:15+08:00] 七項低記憶體語言模型功能與 CUDA 驗證

<!-- devnote-review:5e35253272b68aa325f29ebb -->

補齊低峰值載入、分層推論、4-bit adapter 微調與分塊詞表投影。比對原生模型的梯度、生成與預量化權重重載。上一筆 checkpoint 的 warmup 與 JAX loss 事實已由既有提交與前次交接涵蓋。


## [2026-11-23T04:02+08:00] 確認低記憶體功能已推送至 master

<!-- devnote-review:3b607206d6548c736015bfb6 -->

遠端 master 指向 bf968a27；Git 祖先檢查確認包含 91d6438a，兩者之間另有活化函數提交 cd259666。


## [2026-11-23T04:03+08:00] 確認三後端活化函數與 loss 已推送

<!-- devnote-review:7c1e213cb7d48e302a4cf8a3 -->

核對本機及遠端 master 均為 bf968a27，並確認其包含 cd259666。


## [2026-11-23T05:14+08:00] 雙後端欄位綁定與 loss 轉存驗證

<!-- devnote-review:06fd3cd1a0fb807e5e20740d -->

新增 @loss、@metric、形狀驗證與歧義提示，確認 as_metric 只使用本批已計算的 loss。
