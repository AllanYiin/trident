# 本次實作驗證紀錄

日期：2026-09-28。測試、checkpoint、模型匯出、套件下載快取與測試用相依套件均存放在 `E:/trident-training-tests`；未下載大型模型或語料。

## 本機環境

- Windows、Python 3.10.10。
- PyTorch 2.7.1+cu128、NVIDIA GeForce RTX 5090 Laptop GPU。
- Transformers 4.57.6、Tokenizers 0.22.2。
- HF 匯入原先被全域環境的 pyOpenSSL／cryptography 衝突阻擋；只在 `E:/trident-training-tests/deps` 安裝相容套件，並在測試程序的 PYTHONPATH 優先載入。沒有修改全域 Python 套件。這個目錄是本機驗證環境，不是 Trident 的新增 runtime 依賴。

## 結果

| 驗證 | 結果 |
|---|---|
| 新語言模組測試與可執行範例 | 43 passed；包括 CUDA FP16／BF16、overflow 後恢復、HF Dense／MoE logits、模型匯出與重載 |
| 兩個 Gloo 程序的 DDP | 通過 token 平均梯度等價、GRPO policy／reference 同步、PPO critic 同步 |
| 續訓 | 含 dropout 的監督式訓練、GRPO 採樣、PPO policy／critic／optimizer 恢復皆通過精確數值比對 |
| 舊 Trainer／資料管線回歸 | 38 passed，包含 PyTorch 與 TensorFlow 一步訓練 |
| 原生路徑未安裝 HF／Tokenizer／API extras | 隔離匯入測試通過 |
| 原始碼與封裝 | 新 Python 檔案 AST 解析通過；`git diff --check` 通過；setuptools 可找到 `trident.optims.language` |

43 項新測試的 JUnit 報告位於 `E:/trident-training-tests/runs/language-tests.xml`。測試輸出中的 15 個 warnings 來自第三方套件的 deprecated API，不是失敗或 skip。Linux／Python 3.12 的 GitHub Actions workflow 已加入，尚未在遠端執行。

## 小型 GPU 訓練

實際執行 `quickstart.py --device cuda --precision bf16 --steps 30`：

| 指標 | 實測 |
|---|---:|
| 驗證 loss，訓練前 | 5.582634 |
| 驗證 loss，訓練後 | 1.555667 |
| optimizer updates | 30 |
| 有效訓練 tokens | 3,840 |
| 最後一個更新窗口 tokens/sec | 約 6,704 |
| 該程序的峰值 CUDA allocated bytes | 18,325,504 |

原始結果：`E:/trident-training-tests/runs/quickstart-cuda/result.json`。

可重載原生模型：`E:/trident-training-tests/runs/quickstart-cuda/model`。

完整 checkpoint：`E:/trident-training-tests/runs/quickstart-cuda/checkpoints/latest.pt`。

這是微型合成語料與微型模型的訓練驗證，不是聊天品質測評，也不是 MiniMind 64M 訓練速度的比較。

## 重跑新模組測試

從專案以外的 E 槽工作目錄執行：

```powershell
$tridentSource = 'D:\OneDrive\Documents\trident'
$env:PYTHONPATH = "E:\trident-training-tests\deps;$tridentSource"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TRIDENT_BACKEND = 'pytorch'
$env:TEMP = 'E:\trident-training-tests\tmp'
$env:TMP = $env:TEMP
$env:HF_HOME = 'E:\trident-training-tests\cache\huggingface'
$env:TORCH_HOME = 'E:\trident-training-tests\cache\torch'
Set-Location E:\trident-training-tests
python -m pytest -q --disable-warnings `
  "$tridentSource/tests/test_language_core.py" `
  "$tridentSource/tests/test_language_extended.py" `
  "$tridentSource/tests/test_language_runtime.py" `
  "$tridentSource/tests/test_language_examples.py" `
  --basetemp E:/trident-training-tests/tmp/recheck `
  -o cache_dir=E:/trident-training-tests/cache/pytest `
  --junitxml=E:/trident-training-tests/runs/language-tests.xml
```

乾淨 Python 環境不需要本機的 `deps` workaround；安裝 README 指定的 optional extras 即可。Pytest 會管理其專用 basetemp，請勿將既有模型或其他工作檔放入該目錄。

## 目前邊界

- 沒有執行大型模型訓練、正式語言能力榜單或多張 CUDA GPU 的 NCCL 測試。
- 外部 rollout adapter 已有明確的 policy version／token log-prob 契約，但 vLLM／SGLang 實際服務與權重同步需另行整合。
- 沒有實作或驗收 DeepSpeed、GGUF 轉換、llama.cpp／Ollama 部署、MiniMind-V 或 TensorFlow LLM 訓練。
- TrainingPipeline 目前限單程序；分散式階段可各自透過 torchrun 執行，再由外部 launcher 接續。
- Chat API 使用 buffered SSE，沒有即時逐 token 串流；工具回傳格式與 API 解析已測試，沒有啟動對外服務。
- WandB／SwanLab callback 已提供，但沒有建立遠端實驗 run；實際登入及服務端整合不在本次本機驗證範圍。
