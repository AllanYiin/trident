"""trident callbacks"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from trident.callbacks.callback_base import *
from trident.callbacks.lr_schedulers import AdjustLRCallbackBase,ReduceLROnPlateau,reduce_lr_on_plateau,lambda_lr,LambdaLR,RandomCosineLR,random_cosine_lr
# from trident.callbacks.saving_strategies import *
from trident.callbacks.visualization_callbacks import *
#
from trident.callbacks.regularization_callbacks import RegularizationCallbacksBase, MixupCallback, CutMixCallback
# from trident.callbacks.data_flow_callbacks import DataProcessCallback
# from trident.callbacks import gan_callbacks
#
