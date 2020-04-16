import os
os.environ['TRIDENT_BACKEND']='tensorflow'

import pylab
import numpy as np

import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow_core.python.keras.layers import *
from tensorflow_core.python.keras.models import *
import trident as T
#from trident.backend.common import floatx
import matplotlib
import matplotlib.pyplot as plt


dataset=T.load_mnist('mnist','train',is_flatten=True)


input_var =Input(shape=(28*28,),dtype=T.floatx())
label_var =Input(shape=(10,),dtype=T.floatx())

nn= Dense(64,activation='relu',use_bias=False)(input_var)
nn= Dense(10,activation='softmax',use_bias=False)(nn)
model = Model(input_var, nn,)
model.summary()

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
#
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

tf.keras.utils.plot_model(model, to_file='model.png',show_shapes=True)

loss=tf.keras.losses.categorical_crossentropy(label_var,model.output)

adam=tf.keras.optimizers.Adam()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss='binary_crossentropy',metrics=['accuracy'],target_tensors=label_var)

weights = model.trainable_weights
gradients = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
get_gradients = K.function(inputs=[input_var, label_var], outputs=gradients)

loss_list=[]
metrics_list=[]
grads_list=[]
weights_list=[]

for i in range(9):
    x, y =dataset.next_bach(16)
    grads = get_gradients([x,y])
    grads_list.append(grads[0])
    # ax1.set_xlim([-4, 4])
    # ax1.set_ylim([-4, 4])


    loss, metrics = model.train_on_batch(x,y)
    loss_list.append(loss)
    metrics_list.append(metrics)
    print('loss: {1:.5f} metric:{1:.3%}'.format(loss, metrics))
    weights_list.append(model.get_weights()[0])


#
plt.figure()
plt.plot(np.arange(9),np.array(loss_list))
plt.plot(np.arange(9),np.array(metrics_list),ls=':')
plt.show(block=False)
plt.savefig('loss_metrics.png', dpi=300, format='png')
plt.clf()

fig,ax1=plt.subplots(nrows=3,ncols=3,sharex='all',sharey='all')
for i in range(3):
    for j in range(3):
        ax1[i, j].hist(np.reshape(grads_list[i+j],-1), 8, density=True, histtype='bar', facecolor='b', alpha=0.5)
        ax1[i, j].set_title('step {0}'.format(i*3+j))
plt.show(block=False)
fig.savefig('gradients_histogram.png', dpi=300, format='png')
plt.clf()

fig,ax1=plt.subplots(nrows=3,ncols=3,sharex='all',sharey='all')
for i in range(3):
    for j in range(3):
        ax1[i, j].pcolor(grads_list[i + j].mean(-1).reshape(28, 28))
        ax1[i, j].set_title('step {0}'.format(i * 3 + j))
plt.show(block=False)
fig.savefig('gradients.png', dpi=300, format='png')
plt.clf()

fig,ax1=plt.subplots(nrows=3,ncols=3,sharex='all',sharey='all')
base=weights_list[0].mean(-1).reshape(28,28)
for i in range(3):
    for j in range(3):
        ax1[i, j].pcolor(weights_list[i+j].mean(-1).reshape(28,28) -weights_list[i+j-1].mean(-1).reshape(28,28) if i+j>0 else base)
        ax1[i, j].set_title('step {0}'.format(i*3+j))
plt.show(block=False)
fig.savefig('weights.png', dpi=300, format='png')
plt.clf()

#cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])

#cbar_ax2= fig2.add_axes([0.85, 0.15, 0.05, 0.7])
#fig2.colorbar(im2, cax=cbar_ax2)




