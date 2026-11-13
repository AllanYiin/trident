# Trident 原生語言模型訓練

這組範例示範 Trident 的流式配置如何組合語言模型訓練。資料都是程式內建的微型合成資料，不下載預訓練模型，不需 API key；loss 降低代表訓練管線能運作，不代表已訓練出有實用能力的聊天模型。

## 安裝與執行目錄

在 repository root 安裝；本模組需要 Python 3.10+、PyTorch 2.3+。

```powershell
python -m pip install -e ".[llm]"
```

HF 匯出另外安裝 `.[llm-hf]`，本機 API 另外安裝 `.[llm-serve]`。PyTorch 請依實際 CPU／CUDA 環境安裝。

所有會寫入訓練產物的範例都要求 `--output-dir`。本次使用 E 槽：

```powershell
$env:TRIDENT_BACKEND = 'pytorch'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Get-Location).Path
$env:TEMP = 'E:\trident-training-tests\tmp'
$env:TMP = $env:TEMP
$env:HF_HOME = 'E:\trident-training-tests\cache\huggingface'
$env:TORCH_HOME = 'E:\trident-training-tests\cache\torch'
```

請先建立這些輸出目錄。其他作業系統可以改成自己的可寫入路徑。

## 範例選擇

| 檔案 | 主要任務 | 等級 | 完成時可觀察的結果 |
|---|---|---|---|
| `quickstart.py` | 從零訓練微型 causal LM | Quickstart／L1 | `result.json` 的 after_loss 小於 before_loss，產生 checkpoint 與可重載模型 |
| `staged_training.py` | Pretrain → LoRA SFT → DPO 階段交接 | How-to／L2 | 三個階段產物與 `pipeline.json` 的 complete 狀態 |
| `online_rl.py` | GRPO／CISPO／PPO 更新流程 | Advanced／L2 | 有限的 loss、optimizer 更新與完整 checkpoint |
| `serve.py` | 用原生匯出模型啟動本機聊天 API | Integration／L3 | loopback `/v1/models` 與 `/v1/chat/completions` |

### 第一個訓練

```powershell
python examples/language/quickstart.py --output-dir E:/trident-training-tests/runs/quickstart --steps 30
python examples/language/quickstart.py --output-dir E:/trident-training-tests/runs/quickstart-cuda --device cuda --precision bf16 --steps 30
```

輸出包含更新次數、有效 token 數、最後一個更新窗口的 tokens/sec、峰值 CUDA 配置記憶體、訓練前後 loss。吞吐量是微型範例的實測，不應直接推估大型模型或其他硬體效能。

### 多階段訓練

```powershell
python examples/language/staged_training.py --output-dir E:/trident-training-tests/runs/stages
python examples/language/staged_training.py --output-dir E:/trident-training-tests/runs/stages --resume
```

已存在的 pipeline 必須明確指定 `--resume`。成功階段不重跑；失敗階段用已保存的 checkpoint 恢復。跨階段模型會先匯出，LoRA 會合併，下一階段重新建立 optimizer。

### 線上強化學習

```powershell
python examples/language/online_rl.py --output-dir E:/trident-training-tests/runs/grpo --algorithm grpo
python examples/language/online_rl.py --output-dir E:/trident-training-tests/runs/cispo --algorithm cispo
python examples/language/online_rl.py --output-dir E:/trident-training-tests/runs/ppo --algorithm ppo
```

範例的 reward 只用來驗證管線。請換成任務可驗證的評分或固定的 reward model；不要將範例的 reward 上升解讀為回答品質進步。

### 本機服務

```powershell
python examples/language/serve.py E:/trident-training-tests/runs/quickstart/model --port 8998
```

服務預設只綁定 `127.0.0.1`，按 Ctrl+C 停止。提供基本 chat-completions 格式、`reasoning_content`、`tool_calls`、`open_thinking` 與 buffered SSE；目前 SSE 於整段生成完成後送出，不宣稱即時逐 token 延遲，也不會替 API 呼叫者執行工具。

## 核心 API 與資料

```python
import torch
from trident.optims.language import CausalLanguageModel, SFT
from trident.optims.trainers import TrainingPlan

item = (
    CausalLanguageModel(network, tokenizer)
    .with_objective(SFT())
    .with_lora(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    .with_optimizer(torch.optim.AdamW, lr=2e-4)
    .with_accumulate_grads(8)
    .with_precision("bf16")
    .with_grad_clipping(1.0)
)
plan = (
    TrainingPlan()
    .add_training_item(item)
    .with_data_loader(loader)
    .with_checkpointing(output_directory, every_updates=100)
    .repeat_epochs(2)
)
plan.start_now()
```

`network`、`tokenizer`、`loader` 的完整建立方式見可執行範例。每個 language plan 管理一個可訓練 policy；reference、teacher 與 PPO critic 由 objective 管理。舊 TrainingPlan 的多 training item 行為保留。

- `CausalLMCollator` 接受 `{"text": "..."}`。
- `SFTCollator` 接受 `{"messages": [{"role": "user", "content": "..."}, ...]}`，也接受 MiniMind 的 `conversations`。
- `PreferenceCollator` 接受 `prompt` 加 `chosen`／`rejected` 字串，或兩份具有相同前綴的完整對話。
- `jsonl_dataset(path)` 建立索引式 JSONL 資料來源，會記錄內容 fingerprint。
- `train_bpe_tokenizer(texts, vocab_size=6400)` 從字串 iterable 訓練 BPE／ByteLevel；ByteTokenizer 僅為離線範例提供固定 UTF-8 byte 詞表。
- 預設 `ChatTemplate` 使用 Trident 的 ChatML 角色格式。接外部模型時，必須選用符合該模型訓練格式的 template；不會自動推斷任意模型的對話語意。

Labels 不預先位移，objective 只 shift 一次。SFT 只監督 assistant 回答及結束標記；偏好訓練只計入最後一個 assistant 回答。Padding 依長度標記，支援 PAD 與 EOS 共用 ID。全部 target 被忽略的窗口不更新 optimizer／scheduler；偏好回答完全截斷則回報錯誤。

`PackedCollator` 只合併完整樣本，以 segment IDs 隔離 attention、重設 position IDs 並遮蔽跨樣本 next-token loss。目前適用於原生 CausalLM；不支援 segment IDs 的外部模型會明確拒絕。

## Objective 與 callback 擴充

- `DPO(reference=...)`：reference 必須與 policy 分開建立，且不能共用 Parameter。
- `Distillation(teacher=..., temperature=2, alpha=0.5)`：teacher 與 student 必須使用對齊的詞表及 token 位置。不同 tokenizer 的教師答案應先離線轉成 SFT 資料。
- `GRPO(reference=..., loss_type="grpo" | "cispo")`：每個 prompt 至少兩份回答；相同 reward 的群組給零 advantage。
- `PPO(reference=..., critic=ValueModel(...))`：critic 有獨立 optimizer，與 policy 一起在更新邊界裁剪、跳過 overflow 及保存。
- 自訂 objective 繼承 `CausalLMObjective`，實作 `normalization_weight(batch)`、`__call__`，回傳 `ObjectiveResult(loss_sum, weight, metrics, outputs)`；兩者 weight 必須一致。Objective 不呼叫 backward／step。
- 新 language item 的額外 `with_loss` 使用 `loss(outputs, batch)`，回傳 scalar mean tensor。`with_metric` 使用相同兩個參數，回傳 scalar。舊 Model 的 signature mapping 不變。
- Callbacks 可從 `training_context["language_metrics"]` 取得 loss、有效 tokens、tokens/sec、lr、optimizer_updates；`on_optimization_step_end` 僅在成功更新後觸發。
- 原生 Torch LR scheduler 用 `.with_learning_rate_scheduler(SchedulerClass, **kwargs)`，只依 optimizer update 推進。新路徑的 progress `batch`／`step` 頻率也以 optimizer updates 計算。
- `ExperimentLogger("wandb", project=...)` 或 `ExperimentLogger("swanlab", project=...)` 可加入 `with_callbacks`。只有實際開始訓練才建立遠端 run，僅 rank zero 寫入；需要的登入由各套件處理。

## Checkpoint、DDP 與續訓

`.with_checkpointing(directory)` 保存 `latest.pt`，包括 policy、reference／teacher、PPO critic、所有 optimizer、scheduler、scaler、RNG、iterator 位置、訓練配置與 tokenizer。保存使用同目錄暫存檔及原子 replace，不在例外時覆寫最後成功 checkpoint。

用相同配置建立 plan 後，呼叫 `.resume_from(path)`。精確續訓的主要支援路徑是 deterministic pipeline Iterator；任意 PyTorch DataLoader 的 worker、shuffle 與自訂 transforms 不保證可精確重播。不要在同一個 checkpoint path 同時啟動兩個獨立工作。

DDP 使用 `.with_distributed()`，由 `torchrun` 或呼叫端建立 process group。各 rank 必須產生相同 batch 數；pipeline Iterator 自動帶入 rank／world size。CPU 使用 Gloo；Linux CUDA 可用 NCCL。PPO critic 也同步其梯度。呼叫端負責 process group 的生命週期。

預設拒絕不同 world size 的精確續訓；`.resume_from(path, reshard=True)` 明確允許重新分片，並從中斷 epoch 起點重跑，因此可能重複資料，且不等同於逐步精確重播。RL checkpoint 只在完整 rollout／update cycle 結束後保存，更新數限制也會完成當前 cycle 才停止。

`TrainingPipeline` 負責單程序的順序階段交接。DDP 請分別由 torchrun 執行各階段、由外部 launcher 接續成功產物。

## 生成與 Agentic RL

`TorchRolloutEngine` 用固定 policy snapshot 生成，記錄 action tokens、mask、raw-policy token log-prob 與 policy version。溫度／top-k／top-p 控制採樣；ratio 的 old log-prob 使用未加採樣 warper 的 policy 分布。每個 rollout cycle 更新完後才同步 snapshot。

`ExternalRolloutEngine(sample_fn, sync_fn)` 是可替換生成後端的介面。外部後端必須完成權重同步並提供逐 token 的有效 log-prob；僅有文字的 chat API 不足以訓練。vLLM／SGLang 的具體部署與權重同步需要依安裝版本另行接入、驗證。

`AgenticRolloutEngine(tokenizer, ToolRegistry(), max_turns=4)` 會保留多輪 action 與工具 observation。工具只從 registry 取得，函式必須是可 pickle 的頂層函式，以便 Windows spawn 執行 timeout；沒有預設 shell、網路或檔案工具。工具 observation／角色 header 不參與 policy loss；reward 可從 sample 的 `trajectory` 欄位檢查工具結果。工具執行器提供 timeout 與輸出限制，不是 OS sandbox。

## 匯出與驗證範圍

原生模型支援 `.export(path)` 與 `CausalLM.from_pretrained(path)`。`.export(path, format="huggingface")` 轉成 HF Qwen3／Qwen3-MoE 並使用 safetensors；LoRA 先在副本中合併，不更動訓練模型。`.save_adapter(path)` 只存 adapter。

目前已實測 HF Dense／MoE logits 等價及重載。YaRN、gradient checkpointing、packing 都有原生數值測試。尚未對外部 vLLM、SGLang、llama.cpp、Ollama、GGUF、DeepSpeed 做端到端驗證；HF 匯出不等於上述引擎全部相容。此版不包含 MiniMind-V、多模態或 TensorFlow LLM 訓練。

## 測試與清理

```powershell
python -m pytest -q tests/test_language_core.py tests/test_language_extended.py tests/test_language_runtime.py tests/test_language_examples.py --basetemp E:/trident-training-tests/tmp/pytest -o cache_dir=E:/trident-training-tests/cache/pytest
```

CUDA 測試在沒有 GPU 時跳過；HF、Tokenizer 與 API 測試需要對應的 optional extras。範例的生成文字沒有固定品質要求；驗收依 loss、參數隔離、數值等價、狀態恢復與產物重載。

所有產物均留在明確指定的 output-dir／pytest basetemp。確定不需續訓後，可手動刪除該次專用輸出資料夾；原始碼與外部資料不會被範例刪除。

參考功能來源：[MiniMind](https://github.com/jingyaogong/minimind)。本模組以 Trident 的資料、objective 與 TrainingPlan 介面重新實作。

本次實測結果、E 槽產物位置與重跑命令見 [驗證紀錄](VALIDATION.md)。
