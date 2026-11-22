# 原生語言模型的低記憶體設定

## 載入整個模型或選取模組

在寫出模型時設定每片大小。單一張量超過上限時會單獨占一片。讀取時依索引載入，預設以原生模型的 FP32 精度建立浮點參數；可指定目標精度。原有的單檔 `model.pt` 也可用同一入口讀取。

```python
import torch
from trident.optims.language import CausalLM, CausalLanguageModel, SFT, load_pretrained_weights

# trained 是已訓練的原生 CausalLM
trained.save_pretrained("base", max_shard_size="512MiB")
net = CausalLM.from_pretrained("base", map_location="cpu", dtype=torch.bfloat16)

# 將 checkpoint 中的一個 block 載入既有模型對應的 block
report = load_pretrained_weights(net.model.layers[0], "base", source_prefix="model.layers.0")
print(report["loaded"])
```

載入器先比對參數名稱與形狀。預設 `strict=True`，遇到不相容鍵會在複製前回報錯誤。`strict=False` 僅載入相容項目，回傳 `loaded`、`missing`、`unexpected`、`mismatched`。分片索引可直接排除整個無關檔案。單檔或單片內的 PyTorch ZIP 權重採 memory mapping，較早的非 ZIP 格式會完整讀入該檔。訓練器也支援 `item.export("base", max_shard_size="512MiB")`。重複寫入分片目錄會發布新索引並保留舊分片，讓正在讀取舊索引的程序能完成載入。長期保存各版本時宜使用獨立目錄。

## 依 GPU、RAM 預算做完整模型推論

安裝本專案的 `llm-offload` 額外依賴後，在載入時設定裝置上限。下例的數字是權重配置預算，GPU 還需保留 KV cache、輸出及計算所需空間。

```python
net = CausalLM.from_pretrained(
    "base", device_map="auto", max_memory={0: "6GiB", "cpu": "16GiB"},
    offload_folder="offload", dtype=torch.bfloat16,
)
item = CausalLanguageModel(net)
prompt = torch.tensor([[1, 12, 34]], device=item.device)
tokens = item.generate(prompt, max_new_tokens=8)
```

可從 `net.hf_device_map` 查看實際配置。多裝置或磁碟 offload 用於推論，改變配置時重新載入。原生生成自動只將最後一個位置送入詞表輸出層；直接呼叫 `net(prompt)` 仍可取得所有位置的 logits。

## 在單一裝置微調

```python
base = CausalLM.from_pretrained(
    "base", map_location="cuda", dtype=torch.bfloat16,
    quantization_config={"format": "nf4"},
)
base.save_pretrained("base-nf4", max_shard_size="512MiB")

# 後續微調可直接讀取已打包的權重與 scale
base = CausalLM.from_pretrained("base-nf4", map_location="cuda", dtype=torch.bfloat16)
item = (
    CausalLanguageModel(base)
    .with_lora(rank=8, alpha=16, adapter_dtype=torch.float32)
    .with_objective(SFT(logits_chunk_size=256))
    .with_gradient_checkpointing(offload_to_cpu=True)
    .with_optimizer(lr=2e-4)
)
# 將 item 加到既有 TrainingPlan
```

- **4-bit 基座**使用原生 PyTorch NF4，每 64 個權重共用一個 scale。decoder 的線性投影以兩個值一 byte 儲存，embedding、正規化與輸出頭維持浮點。運算時會暫時展開當前投影權重。可直接讀回 Trident 的預量化模型；Hugging Face 的 bitsandbytes 模型走 `CausalLanguageModel.from_huggingface(..., quantization_config=...)`。原生模型要分層推論時，可先儲存已打包版本，再使用 `device_map` 載入。
- **分塊 loss**的單位是有效目標 token。對每塊做詞表投影與交叉熵，反向時重算。適用原生 `CausalLMObjective` 與 `SFT`。此模式的額外指標可讀取 `hidden_states`、`nll_sum` 和 `aux_loss`；需要完整 logits 的自訂指標應使用預設 objective。
- **Activation CPU offload**在 CUDA 上將梯度檢查點保存的 block 輸入搬到主記憶體，反向再載回。只需重算時使用 `with_gradient_checkpointing()`。CPU 執行使用一般重算模式。
- 使用 `item.save_adapter(path)` 可僅儲存 adapter。原生 `item.export(directory)` 會合併 adapter，已合併的投影以浮點儲存。要繼續完全相同的訓練，使用 TrainingPlan 的訓練 checkpoint 以保留 optimizer 與進度。
