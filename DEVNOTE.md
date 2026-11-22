# DEVNOTE

## SNAPSHOT — 當前狀態

最後更新：2026-11-23T00:32+08:00
交接狀態：verified

### 目標與授權範圍

檢查 loss 基礎類、對齊 PyTorch／TensorFlow／JAX 定義與梯度，補上 JAX loss 入口與必要的損失函數。

### 目前進度

新增 trident/optims/jax_losses.py，三後端共用 34 個公開 loss 名稱（含 TripletMarginLoss 別名）。工作樹包含共用公式、trainer 可訓練中心參數、文件及 CI 測試。

### 下一步

JAX loss 模組與跨後端對齊已驗證。ToolRegistry 子程序逾時仍是獨立的已知問題。

### DECISIONS

- 以 Trident PyTorch 的 reduction、加權分類平均分母及 SoftIoU 符號為對照。保留 PyTorch 的 channels-first 與 TensorFlow/JAX 的 channels-last 慣例。
- 加入 BCEWithLogitsLoss、HuberLoss、TverskyLoss。JAX 用顯式參數或 PRNG key 處理可訓練中心與隨機影像／梯度懲罰。
- 測試在獨立程序依 TRIDENT_BACKEND 載入，TensorFlow 以 tf.function、JAX 以 jax.jit 與自動微分比對。

### LANDMINES

- JAX 驗證為 Windows CPU 的 jax/jaxlib 0.4.35，未驗證 JAX GPU 或 TPU。
- AdaCos 為 PyTorch 舊檔案中的實驗性分數頭，未列入 34 個公開 loss 名稱。

### 證據索引

- trident/optims/jax_losses.py、trident/optims/losses.py、trident/optims/loss_kernels.py、trident/optims/loss_weights.py
- trident/optims/pytorch_losses.py、trident/optims/tensorflow_losses.py、trident/optims/tensorflow_trainer.py
- tests/loss_backend_probe.py、tests/test_loss_parity.py、tests/test_legacy_gradients.py、tests/test_legacy_tf_gradients.py
- docs/zh-tw/losses.md、.github/workflows/legacy-training.yml

### VALIDATION

- Windows 完整 tests/ 回歸 — FAIL：178 passed、1 skipped、1 failed。唯一失敗 test_tool_execution_timeout_and_registry 是先前已記錄的 10 秒工具子程序逾時。獨立 TensorFlow 後端已另測。
- JAX 公開入口 — PASS：trident.get_loss 可取得全部 34 個名稱。三類等分布交叉熵 1.0986123085，對零 logits 的梯度為 [1/3, -2/3, 1/3]。
- TensorFlow 訓練與動態形狀 — PASS：tests/test_legacy_fit.py 與 tests/test_legacy_tf_gradients.py 合計 19 passed，包括兩種 optimizer/loss 設定順序的 CenterLoss 更新及圖模式的有效分母。
- 三後端數值、形狀與反向傳播 — PASS：tests/test_loss_parity.py 的兩組跨後端比較通過。最終 80 組案例，涵蓋 34 個公開名稱與 3 個函數式 API。比對輸出形狀、值、預測梯度和中心梯度，rtol=3e-5、atol=2e-6。
- 語法與差異格式 — PASS：變更與新增共 18 個 Python 檔案通過 AST 解析，git diff --check 通過。

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
