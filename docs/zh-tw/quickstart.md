# 如何建構圖像識別模型AB Test

## 場景

幾乎所有深度學習的hello world就是mnist手寫數據集，我們在這也不例外，我們將會透過這個案例的實作來讓大家理解trident的主要特性與使用邏輯。

## 需求

基於mnist手寫數據集，建構一個識別10個數字的分類模型。但是我們想要知道到底使用傳統的relu活化函數，還是使用參數化relu (parametered relu)哪個效果好一點。這時候我們會需要一個AB Test來協助我們比較兩者的效果。

## 代碼

```python
import os
os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident as T
from trident import *

data_provider = T.load_mnist('mnist')  
data_provider.image_transform_funcs = [add_noise(0.1),normalize(127.5, 127.5)]  

def convnet(activation='relu'):
    return Sequential(
        Conv2d((5,5),16,strides=2,auto_pad=True,activation=activation),
        Conv2d((3, 3),32, strides=2, auto_pad=True,activation=activation),
        Conv2d((3, 3), 64, strides=1, auto_pad=True,activation=activation),
        Flatten(),
        Dense(10)
    )

net1=Model(input_shape=(1,28,28),output=convnet('relu'))
net2=Model(input_shape=(1,28,28),output=convnet('p_relu'))
net1.summary()
net2.summary()


net1.with_optimizer(optimizer='Adam',lr=1e-3) \
    .with_loss(CrossEntropyLoss)\
    .with_metric(accuracy)\
    .with_regularizer('l2')\


net2.with_optimizer(optimizer='Adam',lr=1e-3)\
    .with_loss(CrossEntropyLoss)\
    .with_metric(accuracy)\
    .with_regularizer('l2')


plan=TrainingPlan()\
    .add_training_item(net1,name='net1')\
    .add_training_item(net2,name='net2')\
    .with_data_loader(data_provider)\
    .repeat_epochs(10)\
    .within_minibatch_size(64)\
    .print_progress_scheduling(10,unit='batch')\
    .display_loss_metric_curve_scheduling(100,unit='batch',imshow=True)

plan.start_now()
```



設定 `with_loss()` 時，各後端的類別軸、權重與 reduction 可查閱[跨後端 loss 定義](losses.md)。

## 設定 learning rate warmup

在設定 optimizer 後、呼叫 `plan.start_now()` 前，可以替每個模型分別設定暖身。例如，讓第一組模型先暖身 500 次參數更新，再進入 cosine 排程，第二組則在暖身後維持固定 LR。

```python
from trident.callbacks import CosineLR, LinearWarmup

warmup = LinearWarmup(steps=500, start_factor=0.01)
net1.with_learning_rate_scheduler(CosineLR(period=2000), warmup=warmup)
net2.with_learning_rate_scheduler(warmup=warmup)
```

第一次更新使用各參數群組基準 LR 的 1%，第 500 次更新達到 100%。可依希望 LR 如何變化選擇暖身方式。

| 希望的更新方式 | 暖身設定 | 前五次更新的基準 LR 倍率（約） |
|---|---|---|
| 每次增加相同的量 | `LinearWarmup(5, .01)` | .01、.258、.505、.753、1 |
| 起步與收尾平緩，中段增幅最大 | `CosineWarmup(5, .01)` | .01、.155、.505、.855、1 |
| 從很小的 LR 按固定倍數放大 | `ExponentialWarmup(5, .01)` | .01、.032、.1、.316、1 |
| 先用固定低 LR 訓練，再切換到主排程 | `ConstantWarmup(5, .01)` | .01、.01、.01、.01、.01 |

前三種曲線在最後一次暖身更新達到基準 LR，因此 `steps=1` 時直接使用基準 LR。固定暖身會用指定倍率完成全部更新，下一次才交給主排程。指數暖身需要大於零的 `start_factor`。

知道訓練預計會執行幾次 optimizer 更新時，也可以用比例設定長度。以下設定把 12,000 次更新的前 3%（360 次）分配給暖身。四種曲線都支援 `from_ratio()`。比例範圍為 `0 < ratio <= 1`，遇到不足一步時向上取整。

```python
from trident.callbacks import CosineWarmup

warmup = CosineWarmup.from_ratio(0.03, total_updates=12_000, start_factor=0.01)
net1.with_learning_rate_scheduler(CosineLR(period=2000), warmup=warmup)
```

這組設定也可簡寫為 `net1.with_learning_rate_scheduler(CosineLR(period=2000), warmup="cosine")`。從 INI 讀取 `warmup="cosine"` 時，可將讀到的值直接傳給同一個參數。此簡寫固定使用 360 次成功更新，從各參數群組基準 LR 的 1% 升至 100%。若要依其他訓練預算計算暖身長度，使用上面的 `from_ratio()` 傳入自己的 `total_updates`。

`total_updates` 填入該 optimizer 的預計更新次數。梯度累積完成一次才計一步，AMP overflow 跳過的更新不計步。串流資料或更新總數未知時，可直接指定 `steps`。整數寫法 `warmup=5` 的單位仍是 epoch。

使用外部暖身時，請將 AdamW 或 Ranger21 的內建 warmup 關閉。若在設定後才新增參數群組，trainer 會在下一次 optimizer 更新前檢查新群組的暖身設定。

`CausalLanguageModel` 接受相同的 `warmup`，主排程沿用 PyTorch 原生 scheduler。語言模型的訓練 checkpoint 會自動保存進度。一般模型的 `save_model()` 用於儲存模型權重。自訂訓練 checkpoint 時，請一併保存 `model.optimizer.state_dict()` 和 `model.lr_scheduler.state_dict()`，並依相同順序載回。

原生 PyTorch 語言模型的分片載入、模組權重選取、分層推論與低顯存微調範例，請見[低記憶體模型設定](low-memory.md)。

## 成果

```python
import os
os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident as T
from trident import *
```

```python

data_provider = T.load_mnist('mnist')  
data_provider.image_transform_funcs = [add_noise(0.1),normalize(127.5, 127.5)]  
```



```python
def convnet(activation='relu'):
    return Sequential(
        Conv2d((5,5),16,strides=2,auto_pad=True,activation=activation),
        Conv2d((3, 3),32, strides=2, auto_pad=True,activation=activation),
        Conv2d((3, 3), 64, strides=1, auto_pad=True,activation=activation),
        Flatten(),
        Dense(10)
    )

net1=Model(input_shape=(1,28,28),output=convnet('relu'))
net2=Model(input_shape=(1,28,28),output=convnet('p_relu'))
net1.summary()
net2.summary()
```

![](../images/quickstart_summary.png)

```

net1.with_optimizer(optimizer='Adam',lr=1e-3) \
    .with_loss(CrossEntropyLoss)\
    .with_metric(accuracy)\
    .with_regularizer('l2')\
```



```python

plan=TrainingPlan()\
    .add_training_item(net1,name='net1')\
    .add_training_item(net2,name='net2')\
    .with_data_loader(data_provider)\
    .repeat_epochs(10)\
    .within_minibatch_size(64)\
    .print_progress_scheduling(10,unit='batch')\
    .display_loss_metric_curve_scheduling(100,unit='batch',imshow=True)

plan.start_now()
```



![](../images/quickstart_ouput.png)






## 多輸入、多輸出的欄位對應

PyTorch 與 TensorFlow trainer 共用平坦的欄位名稱。模型的具名輸出可由多個 loss 或 metric 共用。以下模型的 `forward(front, side)` 回傳含 `color_logits` 和 `size_logits` 的字典，資料集另提供兩組標籤。

```python
from trident import loss, metric

model.data_feed = {"front": "front_image", "side": "side_image"}

@loss
def garment_loss(color_logits, color_id, size_logits, size_id):
    return color_criterion(color_logits, color_id) + size_criterion(size_logits, size_id)

@metric(input_names={"pred": "size_logits", "truth": "size_id"})
def head_accuracy(pred, truth):
    return accuracy(pred, truth)

model.with_loss(garment_loss, as_metric=True)
model.with_metric(head_accuracy, name="size_accuracy")
model.with_metric(head_accuracy, name="color_accuracy",
                  input_names={"pred": "color_logits", "truth": "color_id"})
```

範例中的 `color_criterion`、`size_criterion` 和 `accuracy` 是使用者選定的損失與指標函式。字典左側為函式參數，右側為實際欄位。單次註冊可覆寫裝飾器的對應，其次採用 trainer 的 `data_feed`。同名欄位可直接使用；常見的 `output`、`target` 也可使用角色與形狀相容的唯一候選。若有兩個候選，例外會列出欄位名稱與指定方式。

函式可用 Trident 的 `TensorSpec` 實例註解參數，例如 `pred: TensorSpec(shape=TensorShape([None, 3]))`。這表示第二維固定為 3，batch 大小可變。`as_metric=True` 會把這次運算的原始 loss 數值脫離梯度後寫入指標紀錄；`loss_weight` 僅影響訓練用的加權損失。
