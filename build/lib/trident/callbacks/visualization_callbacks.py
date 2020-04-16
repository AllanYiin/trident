import warnings
import math
import numpy as np
import os
from  ..callbacks import CallbackBase
from ..backend.common import *
from ..backend.load_backend import get_backend
from ..misc.ipython_utils import is_in_ipython,is_in_colab
from ..misc.visualization_utils import *
_session=get_session()
_backend=get_backend()

if get_backend()=='pytorch':
    from ..backend.pytorch_backend import to_numpy,to_tensor
elif get_backend()=='tensorflow':
    from ..backend.tensorflow_backend import  to_numpy,to_tensor
elif get_backend()=='cntk':
    from ..backend.cntk_backend import  to_numpy,to_tensor


__all__ = ['VisualizationCallbackBase']

class VisualizationCallbackBase(CallbackBase):
    def __init__(self,epoch_inteval,batch_inteval,save_path: str = None, imshow=False):
        super(VisualizationCallbackBase, self).__init__()
        self.is_in_ipython=is_in_ipython()
        self.is_in_colab=is_in_colab()
        self.epoch_inteval=epoch_inteval
        self.batch_inteval=batch_inteval
        self.save_path=make_dir_if_need(save_path)
        self.imshow = imshow
    pass


class TileImageCallback(VisualizationCallbackBase):
    def __init__(self,epoch_inteval,batch_inteval,save_path: str = None,
                                      name_prefix: str = 'tile_image_{0}.png', include_input=True, include_output=True,
                                      include_target=True, include_mask=None, imshow=False):
        super(TileImageCallback,self).__init__(epoch_inteval,batch_inteval,save_path,imshow)
        self.is_in_ipython=is_in_ipython()
        self.is_in_colab=is_in_colab()
        self.tile_image_name_prefix = name_prefix

        self.include_input = include_input
        self.include_output = include_output
        self.include_target = include_target
        self.include_mask = include_mask

    def plot_tile_image(self,training_context):
        tile_images_list=[]
        input=training_context['current_input']
        target=training_context['current_target']
        output = training_context['current_output']

        if self.include_input:
            input_arr = to_numpy(input).transpose([0, 2, 3, 1]) if _backend != 'tensorflow' else to_numpy(input)
            tile_images_list.append(input_arr * 127.5 + 127.5)
        if self.include_target:
            target_arr = to_numpy(target).transpose([0, 2, 3, 1]) if _backend != 'tensorflow' else to_numpy(target)
            tile_images_list.append(target_arr * 127.5 + 127.5)
        if self.include_output:
            output = self.training_items[0].training_context['current_output']
            output_arr = to_numpy(output).transpose([0, 2, 3, 1]) if _backend != 'tensorflow' else to_numpy(output)
            tile_images_list.append(output_arr * 127.5 + 127.5)

        # if self.tile_image_include_mask:
        #     tile_images_list.append(input*127.5+127.5)
        tile_rgb_images(*tile_images_list,
                        save_path=os.path.join(self.tile_image_save_path, self.self.tile_image_name_prefix),
                        imshow=self.tile_image_imshow)

    def on_batch_end(self, training_context):
        pass

    def on_epoch_end(self, training_context):
        pass
