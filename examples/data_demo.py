import os
os.environ['TRIDENT_BACKEND'] = 'pytorch'

from  trident.data import *
import torch.nn as nn
raw_imgs,raw_labels=load_stanford_cars('cars','train')

dataset=load_mnist('mnist','train',is_flatten=False)


nn.Sequential

