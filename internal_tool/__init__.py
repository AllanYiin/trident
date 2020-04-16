
import torchvision
import torch

#torch_resnet152=torchvision.models.resnet152(pretrained=True)
#pretrain_weights_dict=torch.load('_resnet152.pth')
#torch_resnet152.cuda()


resnet152=ResNet(bottleneck,[3, 8, 36, 3],(3,224,224))


mapping=OrderedDict()
pretrain_weights_dict=torch.load('_resnet152.pth')

for k,v in pretrain_weights_dict.items():
    mapping[k]=''
#pretrain_weights_dict=OrderedDict()
# for  item in pretrain_weights:
#     mapping[item[0]]=''
#     pretrain_weights_dict[item[0]]=item[1]

pretrain_weights1=list(resnet152.model.named_parameters())
buf=list(resnet152.model.named_buffers())
buf=[ item for item in buf if  'running_mean' in  item[0] or  'running_var' in  item[0]]
pretrain_weights1.extend(buf)

for  i in range(len(pretrain_weights1)):
    item=pretrain_weights1[i]
    if 'norm' in item[0] and 'weight'in item[0] :
        for k,v in pretrain_weights_dict.items():
            if mapping[k]=='' and  ('bn' in k or 'downsample' in k) and 'weight'in k and item[1].shape==v.shape:
                mapping[k]=item[0]
                break
    elif 'norm' in item[0] and 'bias' in item[0]:
        for k,v in pretrain_weights_dict.items():
            if mapping[k]=='' and  ('bn' in k or 'downsample' in k) and 'bias'in k and item[1].shape==v.shape:
                mapping[k]=item[0]
                break
    if 'norm' in item[0] and 'running_mean'in item[0] :
        for k,v in pretrain_weights_dict.items():
            if mapping[k]=='' and  ('bn' in k or 'downsample' in k) and 'running_mean'in k and item[1].shape==v.shape:
                mapping[k]=item[0]
                break
    elif 'norm' in item[0] and 'running_var' in item[0]:
        for k,v in pretrain_weights_dict.items():
            if mapping[k]=='' and  ('bn' in k or 'downsample' in k) and 'running_var'in k and item[1].shape==v.shape:
                mapping[k]=item[0]
                break
    elif 'conv' in item[0] and 'weight' in item[0]:
        for k,v in pretrain_weights_dict.items():
            if mapping[k]=='' and ( 'conv' in k or 'downsample' in k) and 'weight'in k and item[1].shape==v.shape:
                mapping[k]=item[0]
                break
    elif 'fc.' in item[0] and 'weight' in item[0]:
        for k, v in pretrain_weights_dict.items():
            if mapping[k] == '' and 'fc' in k and 'weight' in k and item[1].shape == v.shape:
                mapping[k] = item[0]
                break
    elif 'fc.' in item[0] and 'bias' in item[0]:
        for k, v in pretrain_weights_dict.items():
            if mapping[k] == '' and 'fc' in k and 'bias' in k and item[1].shape == v.shape:
                mapping[k] = item[0]
                break
mapping1=OrderedDict()
for k,v in mapping.items():
    mapping1[v]=k


for name,para in resnet152.model.named_parameters():
    if name in mapping1:
        para.data=pretrain_weights_dict[mapping1[name]].data

for name, buf in resnet152.model.named_buffers():
    if name in mapping1:
        buf.data = pretrain_weights_dict[mapping1[name]].data

    # if 'running' in v:
    #     resnet152.model.__setattr__(v,pretrain_weights_dict[k])
    #     #resnet152.model._buffers[v] =
    # else:
    #     resnet152.model._parameters[v]=pretrain_weights_dict[k]

resnet152.model.cpu()
resnet152.save_model('resnet152.pth')
w=np.array(list(resnet152.model.named_parameters()))
np.save('resnet152_weights.npy',w)

print(resnet152.infer_single_image(read_image('dog.jpg'),5))
print(resnet152.infer_single_image(read_image('cat.jpg'),5))
