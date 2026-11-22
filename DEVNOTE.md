# DEVNOTE

## SNAPSHOT — 當前狀態

最後更新：2026-11-23T04:03+08:00
交接狀態：verified

### 目標與授權範圍

確認活化函數及相關已提交變更已推送到 origin/master。

### 目前進度

遠端 master 已包含低記憶體功能 91d6438a、活化函數 cd259666，以及三後端 loss 對齊 bf968a27。

### 下一步

本次推送已完成。

### DECISIONS

- 既有 model.pt 適用同一載入入口。分片檔使用不可變名稱與原子索引切換；預設保留舊分片給進行中的讀取。
- 原生 NF4 以 64 個值分塊，直接載入 packed weights。HF/bitsandbytes 權重透過既有 from_huggingface。分層/offload 配置用於推論。
- 分塊 CausalLM/SFT 在詞表投影前選取有效目標並分塊，保留原有 shift、MoE 輔助 loss、梯度正規化與舊自訂 objective。

### LANDMINES

- Windows 單張 RTX 5090 Laptop GPU 驗證 CPU/GPU/disk offload。跨多張實體顯卡未測。
- 顯存結果取決於模型尺寸、序列與精度。分片目錄的舊代占用磁碟。
- Windows 的 ToolRegistry 子程序 10 秒逾時仍存在；JAX 選用依賴本輪未啟用。

### 證據索引

- trident/optims/language/pretrained.py、quantization.py、trainer.py、objectives.py、generation.py
- trident/models/pytorch_causal_lm.py、tests/test_language_low_memory.py、tests/test_language_memory_cuda.py
- docs/zh-tw/low-memory.md、.github/workflows/language.yml
- git log、git ls-remote、git merge-base；tests/test_llm_activations.py、tests/test_loss_parity.py

### VALIDATION

- CPU 與 CUDA 七項功能及尖峰顯存 — PASS：tests/test_language_low_memory.py 與 tests/test_language_memory_cuda.py 共 9 passed；單批量 512 tokens、32768 詞表、FP32，訓練增量 215543808 → 64969728 bytes，prefill 增量 68433408 → 3565568 bytes。
- 分片載入額外尖峰工作集 — PASS：兩個獨立 Windows CPU 程序載入同一個 26,219,520 參數模型；首次推論前舊路徑增加 212975616 bytes、meta＋mmap 分片增加 1286144 bytes。映射頁面延遲讀取，未計首次 forward。
- 分層推論與舊入口 — PASS：最終 dispatcher 的 auto、CPU/GPU/disk 與預量化分支通過 3 項；自訂 objective、HF logits/export、adapter 匯出等 5 項通過；NF4＋LoRA＋分塊 loss＋activation offload 組合 CPU/CUDA 2 項通過。舊 model.pt 載回後覆寫原目錄亦成功。
- 語法與差異格式 — PASS：12 個變更 Python 檔通過 AST 解析，git diff --check 成功。
- Windows tests/ 完整回歸 — FAIL：188 passed、3 skipped、1 failed。唯一失敗為先前已記錄的 test_tool_execution_timeout_and_registry。完整記錄 .pytest_cache/lowmem-all-final.log。
- 遠端分支與提交祖先 — PASS：git ls-remote origin refs/heads/master 回傳 bf968a27；git merge-base --is-ancestor 91d6438a bf968a27 結束碼 0。
- 遠端 master 與活化函數提交 — PASS：git push 回報 Everything up-to-date，ls-remote 與本機 HEAD 均為 bf968a27，cd259666 是其祖先。

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
