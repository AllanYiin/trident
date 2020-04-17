import os
os.environ['TRIDENT_BACKEND']='tensroflow'
import trident as T



# data=T.data.load_text('C:/Users/Allan/PycharmProjects/DeepTrident/trident/data/iris.txt',label_index=-1)
# print(data[0].shape)
# print(data[0].dtype)
# print(data[1].shape)
# print(data[1].dtype)

data1=T.data.load_birdsnap('birdsnap','train')



data1=T.data.load_cifar('cifar100','train')

dataset=T.data.Dataset('fashion-mnist')
data=T.data.load_mnist('fashion-mnist','train',is_flatten=False)




print(data[0].shape)
print(data[0].dtype)
print(data[1].shape)
print(data[1].dtype)

#dataset.mapping(data=images,labels=labels,scenario='train')

dataset.binding_class_names(['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'])
dataset.binding_class_names(['T恤/上衣','褲子','套衫','連衣裙','外套','涼鞋','襯衫','運動鞋','包包','短靴'],'zh-tw')
print(dataset.get_language())
print(dataset.index2label(3))
print(dataset.label2index('運動鞋'))

#
# fn=T.adjust_blur
# for att in dir(fn):
#     print (att, getattr(fn,att))
#
# fn1= T.get_normalization('bn')
#
# print(fn1.__name__)
#
# fn2=T.get_activation('hard_tanh')
# print(fn2.__name__)
#



# result=T.backend.serialize_object(smooth_relu_func)
# print(result)