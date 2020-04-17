import os
import tkinter as tk
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from IPython import display

os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident as T
from trident import *
from trident.layers.pytorch_activations import __all__


matplotlib.use('TKAgg')
fig=plt.figure()
n=1
items= __all__[int( len(__all__)//2):-1]
plt.clf()
for k in  items :
    if k not in ('p_relu','prelu'):
        try:
            act_fn=get_activation(camel2snake(k))
            x =np.arange(-10, 10, 0.1).astype(np.float32)
            tensor_x=to_tensor(x)
            y=to_numpy(act_fn(tensor_x))
            ax1 = fig.add_subplot(5, 5, n)
            ax1.plot(x,y)
            ax1.plot(x[1:], np.diff(y) /(np.diff(x)+1e-8),ls=':')
            ax1.set_title(k)
            plt.tight_layout(pad=0)
        except Exception as e:
            print(e)
            pass
        n+=1

plt.show()


#display.display(fig)