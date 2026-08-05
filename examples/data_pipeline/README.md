# Trident data pipeline examples

- `vision_geometry.py`：image/mask/boxes 的同步幾何增強與 replay record。
- `text_tokenizer.py`：原生中文字串、batched tokenizer 與 padding。
- `low_memory_training.py`：記憶體預算、bounded prefetch 與 PyTorch 一步訓練。

從 repository root 執行，例如：

```powershell
python examples\data_pipeline\vision_geometry.py
```
