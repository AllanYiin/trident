
import os
import cv2
os.environ['TRIDENT_BACKEND'] = 'tensorflow'
#!pip install tridentx --upgrade
import trident as T
from trident import *
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50

from  trident.models import resnet

res50=resnet.ResNet50(pretrained=False,input_shape=(224,224,3),classes=1000)
#res50.save_model('resnet50.pth.tar')
#tf.saved_model.save(res50.model.state_dict(),'resnet50')




res50_old=ResNet50(include_top=True,input_shape=(224,224,3),weights='imagenet',classes=1000)


layers_old=[m for m in res50_old.layers if len(list(m.variables))>0]
# for i in range(len(layers_old)):
#     layer=layers_old[i]
#     if '_0_conv' in layer.name:
#        if  '_0_bn' in   layers_old[i+2].name:
#            bn=layers_old.pop(i+2)
#            conv=layers_old.pop(i)
#            layers_old.insert(i-4,conv)
#            layers_old.insert(i - 3, bn)
#

print(len(layers_old))


layers_old_dict=OrderedDict()
for layer in layers_old:
    layers_old_dict[layer.name]=str(layer.get_config())

def is_same_layer(m_new,m_old):
    if 'Conv' in m_new.__class__.__name__ and 'Conv' in m_old.__class__.__name__:
        return True
    elif 'Batch' in m_new.__class__.__name__ and 'Batch' in m_old.__class__.__name__:
        return True
    elif 'Dense' in m_new.__class__.__name__ and 'Dense' in m_old.__class__.__name__:
        return True
    else:
        return False



layers_new=[m for m in res50.model.named_modules() if len(m[1]._parameters)+len(m[1]._buffers)>0]
layers_new_dict=OrderedDict()
for k,v in layers_new:
    layers_new_dict[k]=str(v)

print(len(layers_new))
weights_new_dict=OrderedDict()
mappings=OrderedDict()

# for i in range(len(layers_new)):
#     m_new=layers_new[i][1]
#     m_old = layers_old[i]
#     if is_same_layer(m_new,m_old):
#         pass
#     else:
#         if  is_same_layer(m_new, layers_old[i+1]) and  is_same_layer(m_old, layers_new[i+1][1]) :
#            item= layers_old.pop(i)
#            layers_old.insert(i+1,item)
#            m_old = layers_old[i]
#            if is_same_layer(m_new, m_old):
#                pass
#            else:
#                print('')







for item,m_old in zip(layers_new,layers_old):
    name_new,m_new =item
    for  k,v in m_new._parameters.item_list:
        key=name_new + '/' + k+':0'
        mappings[key]=None
    for  k,v in m_new._buffers.item_list:
        key=name_new + '/' + k+':0'
        mappings[key]=None
    if 'Conv' in m_new.__class__.__name__ and  'Conv' in m_old.__class__.__name__:
        if m_new.get_weights()[0].shape==m_old.get_weights()[0].shape:
            mappings[name_new+'/'+m_new.weight.name]= m_old.get_weights()[0]

            mappings[name_new+'/'+m_new.bias.name] =  m_old.get_weights()[1]
        else:
            mappings[name_new+'/'+m_new.weight.name] = None
            mappings[name_new+'/'+m_new.bias.name] = None
    elif   'Batch' in m_new.__class__.__name__ and  'Batch' in m_old.__class__.__name__:
            for w in m_old.weights:
                if  'gamma' in w.name and m_new.weight.shape==w.shape:
                    mappings[name_new+'/'+m_new.weight.name] = to_numpy(w)
                elif  'beta' in w.name and m_new.bias.shape==w.shape:
                    mappings[name_new+'/'+m_new.bias.name] =to_numpy(w)
                elif  'mean' in w.name and m_new.running_mean.shape==w.shape:
                    mappings[name_new+'/'+m_new.running_mean.name] =to_numpy(w)
                elif  'variance' in w.name and m_new.running_var.shape==w.shape:
                    mappings[name_new+'/'+m_new.running_var.name] = to_numpy(w)
                elif  'gamma' in w.name or 'beta' in w.name or   'mean' in w.name or 'variance':
                    mappings[name_new+'/'+m_new.weight.name] = None
                    mappings[name_new + '/' + m_new.bias.name] = None
                    mappings[name_new + '/' + m_new.running_mean.name] = None
                    mappings[name_new + '/' + m_new.running_var.name] = None
                else:
                    print(w.name)


    elif 'Dense' in m_new.__class__.__name__ and 'Dense' in m_old.__class__.__name__:
        if m_new.get_weights()[0].shape==m_old.get_weights()[0].shape:
            mappings[name_new+'/'+m_new.weight.name] =  m_old.get_weights()[0]
            mappings[name_new+'/'+m_new.bias.name] =  m_old.get_weights()[1]
        else:
            mappings[name_new+'/'+m_new.weight.name] = None
            mappings[name_new+'/'+m_new.bias.name] = None
    else:
        print(m_old)

print(len(mappings))
nn=0
for name,module in res50.model.named_modules():

    for  k,v in module._parameters.item_list:
        key=name + '/' + k+':0'
        if key in mappings and mappings[key] is not None:
            if v.shape==mappings[key].shape:
                #v.assign(tf.Variable(mappings[key]))
                v.assign(tf.Variable(np.reshape(mappings[key], (v.shape.as_list())).astype(np.float32)))
                nn+=1
    for  k,v in module._buffers.item_list:
        key=name + '/' + k+':0'
        if key in mappings and mappings[key] is not None:
            if v.shape == mappings[key].shape:
                #v.assign(tf.constant(mappings[key]))
                v.assign(tf.Variable(np.reshape( mappings[key],(v.shape.as_list())).astype(np.float32)))
                nn+=1


print(res50.infer_single_image('cat.jpg',3))



# for name,module in res50.model.named_modules():
#     for  k,v in module._parameters.item_list:
#         key=name + '/' + k+':0'
#         if key in mappings and mappings[key] is not None:
#             if v.shape==mappings[key].shape:
#                 if (v!=mappings[key]).numpy().any():
#                     print(key)
#     for  k,v in module._buffers.item_list:
#         key=name + '/' + k+':0'
#         if key in mappings and mappings[key] is not None:
#             if v.shape == mappings[key].shape:
#                 if (v != mappings[key]).numpy().any():
#                     print(key)
#cat_array=np.expand_dims((resize((224,224),True)(image2array('cat.jpg'))-127.5)/127.5,0)
#print('cat',cat_array.shape)
# new_list=list(res50.model.modules())
# conv1_new=new_list[2]
# bn_new=new_list[3]
#
# padding_old=res50_old.layers[1]
# conv1_old=res50_old.layers[2]
# bn_old=res50_old.layers[3]
# print('is weight the same? {0}'.format(np.array_equal(conv1_new.get_weights()[0],conv1_old.get_weights()[0])))
# print('is bias the same? {0}'.format(np.array_equal(conv1_new.get_weights()[1],conv1_old.get_weights()[1])))
# conv1_weights_new=conv1_new.get_weights()
# conv1_weights_old=conv1_old.get_weights()
# print(np.abs(conv1_weights_new[0]-conv1_weights_old[0]).sum())
# print(np.abs(conv1_weights_new[1]-conv1_weights_old[1]).sum())
#
# results_new_conv1=bn_new(conv1_new(to_tensor(cat_array.copy())))[0].numpy()
# results_old_conv1=bn_old(conv1_old(padding_old(to_tensor(cat_array.copy()))))[0].numpy()
# is_ok=np.array_equal(results_new_conv1,results_old_conv1)
# print('is result the same? {0}'.format(is_ok))
# print(results_new_conv1.shape,results_old_conv1.shape)
# print(results_new_conv1.mean(),results_old_conv1.mean())
# print(results_new_conv1[0,0,:].mean(),results_old_conv1[0,0,:].mean())
# print(results_new_conv1[-1,-1,:].mean(),results_old_conv1[-1,-1,:].mean())
# if not is_ok:
#     print(results_new_conv1[0,0,:]-results_old_conv1[0,0,:])



res50.model.eval()
print(res50.infer_single_image('cat.jpg',5))
save(res50.model,'resnet50_tf.pth')
res50.save_model('resnet50_tf.pth.tar')

print('finish')





