![trident](trident_logo.png)
**Make PyTorch and TensorFlow two become one.**


| version                                                                                                                                                                                      | pytorch                                                                                           | tensorflow                                                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [![version](https://img.shields.io/static/v1?label=&message=0.7.6&color=377EF0&style=for-the-badge)](https://img.shields.io/static/v1?label=&message=0.7.5&color=377EF0&style=for-the-badge) | ![pytorch](https://img.shields.io/static/v1?label=&message=>1.4&color=377EF0&style=for-the-badge) | ![tensorflow](https://img.shields.io/static/v1?label=&message=>2.2.0&color=377EF0&style=for-the-badge) |

**Trident** is a deep learning dynamic calculation graph api based on PyTorch and TensorFlow (pure Eager mode, no Keras dependency). Through Trident, not only can you use the same developer experience (more than 99% of the same code) within PyTorch and Tensorflow, it is also designed to simplify deep learning developers routine work. It's functions not only cover computing vision, natural language understanding and reinforcement learning, but also include a simpler network structure declaration, a more powerful but easier training process control, intuitive data access and data augmentation.

**Trident** 是基於PyTorch和TensorFlow的深度學習動態計算圖API（純Eager模式，無Keras依賴）。 通過Trident，您不僅可以在PyTorch和Tensorflow中使用相同的開發經驗（超過99％的相同代碼），它的誕生的目的就是希望簡化深度學習開發人員的日常工作，Trident的功能不但覆蓋機器視覺、自然語言與強化學習。它的功能還包括更簡單的網絡結構宣告，更強大但更容易實現的訓練流程控制，直觀的數據訪問和數據增強。

## Key Features

- Integrated Pytorch and Tensorflow experience (from ops operation, neural network structure announcement, loss function and evaluation function call...)
- Able to automatically transpose the tensor direction according to the background type (PyTorch (CHW) or Tensorflow (HWC))
- Only one original neural block is used to meet more needs. For example, Conv2d_Block integrates five functions including convolution layer, normalization, activation function, dropout, and noise.
- The amount of padding can be automatically calculated through Autopad during neural layer design. Even PyTorch can delay shape inference, and use Summary to view model structure and computing power consumption information.
- Rich built-in visualization, evaluation function and internal information can be inserted into the training plan.
- Training Plan can be flexible like building blocks stacked to design the training process you want, while using fluent style syntax to make the overall code easier to read and easier to manage.
- Provide the latest optimizers (Ranger, Lars, RangerLars, AdaBelief...) and optimization techniques (gradient centralization).
- 整合一致的Pytorch與Tensorflow體驗(從ops操作、神經網路結構宣告、損失函數與評估函數調用....)
- 能夠根據後台種類(PyTorch (CHW) 或是 Tensorflow (HWC))自動進行張量方向轉置
- 僅用神經區塊(block)元件來滿足更多的建模需求，例如Conv2d_Block整合了卷積層、正規化、活化函數、Dropout、噪音等五種功能，同時可以在block中執行神經層融合。
- 神經層設計時可以透過Autopad 自動計算padding量，就連PyTorch也可以延遲形狀推斷，以及使用Summary檢視模型結構與算力耗用信息。
- 豐富的內建視覺化、評估函數以及內部訊息，可供插入至訓練計畫中。
- 訓練計畫(Training Plan) 可以如同堆積木般彈性設計你想要的訓練流程，同時使用fluent style語法讓整體代碼易讀更容易管理。
- 提供最新的優化器( Ranger,Lars, RangerLars, AdaBelief...)以及優化技巧(gradient centralization)。

a

## New Release version 0.7.4

- Experimental: Keras model (at tensorflow backend) and Primitive pytorch model (at pytorch backend) support in TrainingPlan
- print_gpu_utilization in TrainingPlan
- Experimental: Layer fusion (conv+norm=>conv) in  ConvXd_Block, FullConnect_Block
- Experimental: Automatic inplace of Relu and LeakyRelu, switch back to False when it detect in leaf layer.
- Experimental: MLFlow support
- New optimizer LAMB, Ranger_AdaBelief
- Rewrite lots of loss function.
  . List[String](image path) as output of ImageDataset
  .
  . More stable and reliability.

## New Release version 0.7.3

![Alt text](images/text_process.png)

- New with_accumulate_grad for accumulating gradient.
- Enhancement for TextSequenceDataset and TextSequenceDataprovider.
- New TextTransform: RandomMask,BopomofoConvert,ChineseConvert,RandomHomophonicTypo,RandomHomomorphicTypo
- New VisionTransform: ImageMosaic, SaltPepperNoise
- Transformer, Bert, Vit support in pytorch backend.
- New layers and blocks: FullConnect_Block,  TemporalConv1d_Block
- Differentiable color space convertion function: rgb2hsv, rgb2xyz rgb2lab....
- Enhancement for GANBuilder, now conditional GAN and skip-connections networks is support.
- LSTM support attention in pytorch backend, and LSTM comes in tensorflow mode.

## New Release version 0.7.1

![Alt text](images/vision_transform.png)

- New Vision Transform.

## New Release version 0.7.0

![Alt text](images/tensorboard.png)

- Tensorboard support.
- New optimizer: AdaBelief, DiffGrad
- Initializers support.

## How To Use

#### Step 0: Install

Simple installation from PyPI

```bash
pip install tridentx  --upgrade
```

#### Step 1: Add these imports

```python
import os
os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident as T
from trident import *
from trident.models.pytorch_densenet import DenseNetFcn
```

#### Step 2: A simple case both in PyTorch and Tensorflow

```
data_provider=load_examples_data('dogs-vs-cats')
data_provider.image_transform_funcs=[
    random_rescale_crop(224,224,scale=(0.9,1.1)),
    random_adjust_gamma(gamma=(0.9,1.1)),
    normalize(0,255),
    normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]

model=resnet.ResNet50(include_top=True,pretrained=True,freeze_features=True,classes=2)\
    .with_optimizer(optimizer=Ranger,lr=1e-3,betas=(0.9, 0.999),gradient_centralization='all')\
    .with_loss(CrossEntropyLoss)\
    .with_metric(accuracy,name='accuracy')\
    .unfreeze_model_scheduling(200,'batch',5,None) \
    .unfreeze_model_scheduling(1, 'epoch', 4, None) \
    .summary()

plan=TrainingPlan()\
    .add_training_item(model)\
    .with_data_loader(data_provider)\
    .repeat_epochs(10)\
    .within_minibatch_size(32)\
    .print_progress_scheduling(10,unit='batch')\
    .display_loss_metric_curve_scheduling(200,'batch')\
    .print_gradients_scheduling(200,'batch')\
    .start_now()
```

#### Step 3: Examples

- mnist classsification [pytorch](https://github.com/AllanYiin/DeepBelief_Course5_Examples/blob/master/epoch001_%E5%8F%A6%E4%B8%80%E7%A8%AE%E8%A7%92%E5%BA%A6%E7%9C%8Bmnist/HelloWorld_mnist_pytorch.ipynb)  [tensorflow](https://github.com/AllanYiin/DeepBelief_Course5_Examples/blob/master/epoch001_%E5%8F%A6%E4%B8%80%E7%A8%AE%E8%A7%92%E5%BA%A6%E7%9C%8Bmnist/HelloWorld_mnist_tf.ipynb)
- activation function [pytorch](https://github.com/AllanYiin/DeepBelief_Course5_Examples/blob/master/epoch002_%E6%B4%BB%E5%8C%96%E5%87%BD%E6%95%B8%E5%A4%A7%E6%B8%85%E9%BB%9E/%20Activation_Function_AllStar_Pytorch.ipynb)  [tensorflow](https://github.com/AllanYiin/DeepBelief_Course5_Examples/blob/master/epoch002_%E6%B4%BB%E5%8C%96%E5%87%BD%E6%95%B8%E5%A4%A7%E6%B8%85%E9%BB%9E/Activation_Function_AllStar_tf.ipynb)
- auto-encoder [pytorch](https://github.com/AllanYiin/DeepBelief_Course5_Examples/blob/master/epoch003_%E8%87%AA%E5%8B%95%E5%AF%B6%E5%8F%AF%E5%A4%A2%E7%B7%A8%E7%A2%BC%E5%99%A8/Pokemon_Autoencoder_pytorch.ipynb)  [tensorflow](https://github.com/AllanYiin/DeepBelief_Course5_Examples/blob/master/epoch003_%E8%87%AA%E5%8B%95%E5%AF%B6%E5%8F%AF%E5%A4%A2%E7%B7%A8%E7%A2%BC%E5%99%A8/Pokemon_Autoencoder_tf.ipynb)

## BibTeX

If you want to cite the framework feel free to use this:

```bibtex
@article{AllanYiin2020Trident,
  title={Trident},
  author={AllanYiin, Taiwan},
  journal={GitHub. Note: https://github.com/AllanYiin/trident},
  volume={1},
  year={2020}
}
```
