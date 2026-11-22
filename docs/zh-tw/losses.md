# Loss 的跨後端定義

Trident 的 PyTorch、TensorFlow 與 JAX 後端都可用 `get_loss("CrossEntropyLoss")` 取得類別。選擇後端後，再建立 loss 並以 `loss(output, target)` 呼叫。分類的預設類別軸在 PyTorch 是 1，在 TensorFlow 與 JAX 是 -1。從另一個後端移植相同排列的張量時，請明確傳入 `axis`。

```python
from trident import get_loss

criterion = get_loss("CrossEntropyLoss")(
    axis=-1, from_logits=True, sample_weight=[1., 3., .5], ignore_index=-100,
)
value = criterion(scores, labels)
```

## 輸入、權重與平均

| 情境 | 使用方式 |
|---|---|
| 多類別分類 | `CrossEntropyLoss(from_logits=True)` 接收未正規化分數，`False` 接收機率。省略時沿用 Trident 對 logits、機率與 log-probability 的辨識。已計算 log-softmax 的模型可直接用 `NLLLoss`。 |
| 非等權類別或遮罩 | `sample_weight` 是一維類別權重。稀疏標籤的交叉熵以有效標籤的權重總和為平均分母。整批都是忽略標籤時回傳零，預測梯度也是零。 |
| 軟標籤 | 傳入與預測同形的目標分布。交叉熵保留各類別的目標機率，依類別權重加總，再按有效位置數平均。`label_smooth=True` 表示 0.1，也可指定 0 到 1 的平滑係數。 |
| 多標籤二元分類 | `BCEWithLogitsLoss(pos_weight=...)` 使用數值穩定的 logits 公式，支援可廣播的正例權重和元素權重。`reduction="none"` 保留每個元素。既有的 `BCELoss` 會沿類別軸加總。 |
| 穩健迴歸 | `HuberLoss(delta=.3)` 可調整平方誤差轉為線性誤差的門檻。`SmoothL1Loss` 沿用 PyTorch 的預設門檻 1。`L2Loss` 是逐元素平方誤差的一半。 |
| 分割時誤判與漏判的成本不同 | `TverskyLoss(alpha=.3, beta=.7, from_logits=True)` 對 false positive 使用 `alpha`，對 false negative 使用 `beta`。較大的 `beta` 會提高漏判代價。 |
| 指標轉成訓練目標 | `DiceLoss`、`GeneralizedDiceLoss` 與 `IoULoss` 計算 1 減相似度。`GeneralizedDiceLoss` 使用 Trident 的類別權重約定。`SoftIoULoss` 保留既有 PyTorch 的負平均 IoU，三個後端符號相同。 |

`mean` 和舊名稱 `batch_mean` 都取元素平均。`sum` 加總所有元素。`batch_sum` 沿 batch 軸加總，再平均其他維度。`batchmean` 則是總和除以 batch 大小，適用於與 `torch.nn.KLDivLoss(reduction="batchmean")` 對照。

`none` 傳回個別 loss 定義的未歸約結果。Trident 的 L1、L2、MSE、Smooth L1 對有 batch 的輸入沿用 PyTorch 版的 `(batch, features)` 展平約定。Dice 與 IoU 在計算交集及聯集時已聚合空間維度。

## 訓練與驗證

- `CenterLoss` 的中心是可訓練參數。TensorFlow trainer 在 optimizer 和 loss 都已設定後註冊中心。JAX 可以對 `loss.with_centers(features, labels, centers)` 的顯式中心參數取梯度。
- `GPLoss` 可傳入 `alpha` 重現同一批插值樣本。`TransformInvariantLoss` 可指定 `angle`（度）及 `scale`。JAX 隨機取樣時傳入 `key`，可控制每步的隨機性。
- 對有效輸入，跨後端測試比較輸出形狀、數值和預測梯度。另測量中心梯度和 WGAN-GP 對 critic 的梯度。TensorFlow 在 `tf.function` 中執行，JAX 在 `jax.jit` 和自動微分下執行。
- 安裝三個後端後，可從專案根目錄執行 `python -m pytest -q tests/test_loss_parity.py`。測試會為每個後端啟動獨立程序，以符合 Trident 在匯入時選擇後端的設計。
