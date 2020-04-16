
import os
import cv2
os.environ['TRIDENT_BACKEND'] = 'pytorch'
#!pip install tridentx --upgrade
import trident as T
from trident import *
import torch
from trident.models.pytorch_mobilenet import *

mobile_old=MobileNetV2(pretrained=True,include_top=True,num_classes=1000)
mobile=MobileNetV2(pretrained=False,include_top=True,num_classes=1000)
#
mobile.model.trainable=True

paras_old=list(mobile_old.model.named_parameters())
paras_old_dict=OrderedDict()
for k,v in paras_old:
    paras_old_dict[k]=v

paras=list(mobile.model.named_parameters())

print('paras_old',len(paras_old))
print('paras',len(paras))

buffer_old=list(mobile_old.model.named_buffers())
buffer=list(mobile.model.named_buffers())


buffer_old=[b for b in buffer_old  if 'running_mean' in b[0] or 'running_var' in b[0]]
buffer_old_dict=OrderedDict()
for k,v in buffer_old:
    buffer_old_dict[k]=v


buffer=[b for b in buffer  if 'running_mean' in b[0] or 'running_var' in b[0]]

print('buffer_old',len(buffer_old))
print('buffer',len(buffer))

for i in range(len(paras)):
    p = paras[i]
    if p[0] not in paras_old_dict:
        p_old = paras_old_dict[p[0].replace('conv.', 'conv.conv1.')]
    else:
        p_old = paras_old_dict[p[0]]
    if 'norm' in p[0] and (len(p[1].shape)>1or len(p_old.shape)>1):
        print(p[0])
    if  p[1].shape==p_old.shape: #p[0]==p_old[0] and
        p[1].data=p_old.data
    else:
        print(p[0],p[1].shape, p[0], p_old.shape)

for i in range(len(buffer)):
    b = buffer[i]
    b_old = buffer_old_dict[b[0]]
    if 'running_mean' in b[0] or 'running_var' in b[0]:
        if len(b[1].shape)>1 or len( b_old.shape)>1:
            print(b[0])
        if  b[1].shape == b_old.shape: #b[0] == b_old[0] and
            b[1].data = b_old.data
    else:
        print(b[0],b[1].shape, b[0],b_old.shape)




mobile.model.eval()
mobile.model.cuda()
print(mobile.infer_single_image('cat.jpg',5))




mobile.save_onnx('mobilenet_v2.onnx')

#
# example = torch.rand(1, 3, 224, 224).cuda()
#
# traced_script_module = torch.jit.trace(mobile.model, example)
# traced_script_module.save("efficientnet-b0.pt")

mobile.model.cpu()

torch.save(mobile.model,'mobilenet_v2.pth')
mobile.model.cuda()















#
#
#
#
# mobile=EfficientNet(1.0, 1.1, 240, 0.2, model_name='efficientnet-b1',include_top=True,num_classes=1000)
# pretrain_weights1 =list(mobile.model.named_parameters())
# buf=list(mobile.model.named_buffers())
# buf=[ item for item in buf if  'running_mean' in  item[0] or  'running_var' in  item[0]]
# pretrain_weights1.extend(buf)
#
#
#
# effnetb0=torch.load('C:/Users/Allan/Downloads/efficientnet-b1.pth')
# pretrain_weights=OrderedDict()
#
# for k,v in effnetb0.items():
#     if 'weight' in k or 'bias' in k or  'running_mean' in k or  'running_var' in k:
#         pretrain_weights[k]=v
#
#
# mapping=OrderedDict()
# pretrain_weights_dict=OrderedDict()
# for  item in pretrain_weights.item_list:
#     mapping[item[0]]=''
#     pretrain_weights_dict[item[0]]=item[1]
#
# keyword_map1={"_blocks.0.":"block1a",
# "_blocks.1.":"block2a",
# "_blocks.2.":"block2b",
# "_blocks.3.":"block3a",
# "_blocks.4.":"block3b",
# "_blocks.5.":"block4a",
# "_blocks.6.":"block4b",
# "_blocks.7.":"block4c",
# "_blocks.8.":"block5a",
# "_blocks.9.":"block5b",
# "_blocks.10.":"block5c",
# "_blocks.11.":"block6a",
# "_blocks.12.":"block6b",
# "_blocks.13.":"block6c",
# "_blocks.14.":"block6d",
# "_blocks.15.":"block7a",
# "fc":"fc"}
# keyword_map={}
# for k,v in keyword_map1.items():
#     keyword_map[v]=k
#
#
#
# no_match=[]
#
# for  i in range(len(pretrain_weights1)):
#     item=pretrain_weights1[i]
#     k1 = None
#     for k in keyword_map.keys():
#         if k in item[0]:
#             k1 = keyword_map[k]
#
#     if ('norm' in item[0] or 'bn' in  item[0] ) and 'weight'in item[0] :
#         for k,v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or ( k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k]=='' and  ('bn' in k or 'norm' in k) and 'weight'in k and item[1].shape==v.shape:
#                     mapping[k]=item[0]
#                     break
#
#     elif ('norm' in item[0] or 'bn' in  item[0] ) and 'bias' in item[0]:
#         for k,v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or ( k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k]==''and  ('bn' in k or 'norm' in k) and 'bias'in k and item[1].shape==v.shape:
#                     mapping[k]=item[0]
#                     break
#
#     elif  'running_mean'in item[0] :
#         for k,v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or (k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k]==''and   'running_mean'in k and item[1].shape==v.shape:
#                     mapping[k]=item[0]
#                     break
#
#     elif 'running_var' in item[0]:
#         for k,v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or ( k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k]==''  and 'running_var'in k and item[1].shape==v.shape:
#                     mapping[k]=item[0]
#                     break
#
#
#     elif  'squeeze.weight' in item[0]:
#         for k,v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or (k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k]==''and   '_se_reduce.weight'in k and item[1].shape==v.shape:
#                     mapping[k]=item[0]
#                     break
#
#     elif 'squeeze.bias' in item[0]:
#         for k, v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or (
#                     k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k] == '' and '_se_reduce.bias' in k and item[1].shape == v.shape:
#                     mapping[k] = item[0]
#                     break
#
#     elif 'excite.weight' in item[0]:
#         for k, v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or (
#                     k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k] == '' and '_se_expand.weight'in k and item[1].shape == v.shape:
#                     mapping[k] = item[0]
#                     break
#     elif 'excite.bias' in item[0]:
#         for k, v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or (
#                     k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k] == '' and '_se_expand.bias' in k and item[1].shape == v.shape:
#                     mapping[k] = item[0]
#                     break
#
#     elif ('conv' in item[0]  or 'depthwise'in item[0]  ) and 'weight' in item[0]:
#         for k,v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or ( k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k]==''and ( 'conv' in k or 'feature' in k) and 'weight'in k and item[1].shape==v.shape:
#                     mapping[k]=item[0]
#                     break
#
#     elif ('conv' in item[0]  or 'depthwise'in item[0] )  and 'bias' in item[0]:
#         for k, v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or ( k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k] == '' and ( 'conv' in k or 'feature' in k)  and 'bias' in k and item[1].shape == v.shape:
#                     mapping[k] = item[0]
#                     break
#
#     elif 'fc' in item[0] and 'weight' in item[0]:
#         for k, v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or ( k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k] == '' and 'fc' in k and 'weight' in k and item[1].shape == v.shape:
#                     mapping[k] = item[0]
#                     break
#
#     elif 'fc' in item[0] and 'bias' in item[0]:
#         for k, v in pretrain_weights_dict.items():
#             if (k1 is not None and k1 in k) or ( k1 is None and len([vv for vv in keyword_map.values() if (vv in k)]) == 0):
#                 if mapping[k] == '' and 'fc' in k and 'bias' in k and item[1].shape == v.shape:
#                     mapping[k] = item[0]
#                     break
#
#     else:
#         no_match.append(item)
#
#
# print(len(set(list(mapping.key_list))))
# print(len(set(list(mapping.value_list))))
# mapping1=OrderedDict()
# for k,v in mapping.items():
#     mapping1[v]=k
#
#
#
# for name,para in mobile.model.named_parameters():
#     if name in mapping1:
#         para.data=pretrain_weights_dict[mapping1[name]].data
#
# for name, buf in mobile.model.named_buffers():
#     if name in mapping1:
#         buf.data = pretrain_weights_dict[mapping1[name]].data
#
#
# print(mobile.infer_single_image('cat.jpg',5))
#
# mobile.model.cpu()
# #vgg19.save_model('vgg19.pth')
# torch.save(mobile.model,'efficientnet-b0.pth')
#
#
