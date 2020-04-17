import os
import sys
import codecs

os.environ['TRIDENT_BACKEND'] = 'tensorflow'
import trident as T
import math
import numpy as np
import linecache

import tensorflow as tf




# C.debugging.set_computation_network_trace_level(1000)
# C.debugging.set_checked_mode(True)
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))




# data = T.load_cifar('cifar100', 'train', is_flatten=False)
# dataset = T.Dataset('cifar100')
# dataset.mapping(data=data[0], labels=data[1], scenario='train')


import torchvision.transforms as transforms
transform = transforms.Compose(
[transforms.ToTensor()])
minst=torchvision.datasets.mnist.MNIST(root='C:/Users/Allan/.trident/datasets/minist/', train=True, transform=transform, target_transform=None, download=True)


train_loader = DataLoader(
        minst,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )


num_epochs=50
minibatch_size=32



#raw_imgs, raw_labels = dataset.next_bach(minibatch_size)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out
#lenet_model = torch.load('lenet_model_only_pytorch_mnist.pth')
lenet_model=LeNet()
lenet_model.to(device)
#lenet_model.load_state_dict(torch.load('lenet_model_pytorch_mnist.pth'))

optimizer = optim.Adam(lenet_model.parameters(),lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
criterion=nn.CrossEntropyLoss(reduction='sum')
#
# gcd_model2 = nn.Sequential(
#     nn.Conv2d(in_channels=1,out_channels= 16, kernel_size=3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.MaxPool2d(2,padding=1),
#     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2,padding=1),
#     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#     nn.Conv2d(in_channels=64, out_channels=120, kernel_size=3, padding=1),
#     nn.AdaptiveAvgPool2d(output_size=120),
#     Classifier1d(num_classes=10,is_multiselect=False, classifier_type='dense')
# )




class GcdNet(nn.Module):
    def __init__(self):
        super(GcdNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.b1=T.GcdConv2d_Block((3, 3), 24, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        self.b2=T.GcdConv2d_Block((3, 3), 40, 2, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        self.b2_1 = T.GcdConv2d_Block((1,1), 120, 1, auto_pad=True, activation=None, divisor_rank=0)
        self.b2_2 = T.GcdConv2d_Block((3, 3),120, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        self.b2_3 = T.GcdConv2d_Block((1, 1), 40, 1, auto_pad=True, activation=None, divisor_rank=0)

        self.b3=T.GcdConv2d_Block((3, 3), 56, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        # self.b4=T.GcdConv2d_Block((3, 3), 88, 2, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        # self.b5=T.GcdConv2d_Block((3, 3), 104, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        self.b6=T.GcdConv2d_Block((3, 3), 88, 2, auto_pad=True, activation=None)
        self.pool=nn.AdaptiveAvgPool2d(output_size=88)
        self.fc2 = nn.Linear(88, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.b1(out)
        out = self.b2(out)
        out1 = self.b2_1(out)
        out1 = self.b2_2(out1)
        out1 = self.b2_3(out1)
        out=out+out1
        out = self.b3(out)
        # out = self.b4(out)
        # out = self.b5(out)
        out = self.b6(out)
        out = self.pool(out)
        out = out.view(out.size(0),out.size(1),-1 )
        out=torch.mean(out,-1,False)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


class GcdNet_1(nn.Module):
    def __init__(self):
        super(GcdNet_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.b1=T.GcdConv2d_Block_1((3, 3), 24, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        self.b2=T.GcdConv2d_Block_1((3, 3), 40, 2, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        self.b2_1 = T.GcdConv2d_Block_1((1,1), 120, 1, auto_pad=True, activation=None, divisor_rank=0)
        self.b2_2 = T.GcdConv2d_Block_1((3, 3),120, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        self.b2_3 = T.GcdConv2d_Block_1((1, 1), 40, 1, auto_pad=True, activation=None, divisor_rank=0)

        self.b3=T.GcdConv2d_Block_1((3, 3), 56, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        # self.b4=T.GcdConv2d_Block((3, 3), 88, 2, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        # self.b5=T.GcdConv2d_Block((3, 3), 104, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0)
        self.b6=T.GcdConv2d_Block_1((3, 3), 88, 2, auto_pad=True, activation=None)
        self.pool=nn.AdaptiveAvgPool2d(output_size=88)
        self.fc2 = nn.Linear(88, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.b1(out)
        out = self.b2(out)
        out1 = self.b2_1(out)
        out1 = self.b2_2(out1)
        out1 = self.b2_3(out1)
        out=out+out1
        out = self.b3(out)
        # out = self.b4(out)
        # out = self.b5(out)
        out = self.b6(out)
        out = self.pool(out)
        out = out.view(out.size(0),out.size(1),-1 )
        out=torch.mean(out,-1,False)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out

#
#
# #T.Conv2d_Block((3, 3), 16, 1, auto_pad=True, activation='leaky_relu6', normalization=None)
# gcd_model = nn.Sequential(
#     nn.Conv2d(1, 16, 3, padding=1, bias=False),
#     nn.ReLU(),
#     T.GcdConv2d_Block((3, 3), 24, 1, auto_pad=True, activation='leaky_relu6',divisor_rank=0),
#     T.GcdConv2d_Block((3, 3), 40, 2, auto_pad=True, activation=None, divisor_rank=0),
#     T.GcdConv2d_Block((3, 3), 56, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0),
#     T.GcdConv2d_Block((3, 3), 88, 2, auto_pad=True, activation=None, divisor_rank=0),
#     T.GcdConv2d_Block((3, 3), 104, 1, auto_pad=True, activation='leaky_relu6', divisor_rank=0),
#     T.GcdConv2d_Block((3, 3), 136, 1, auto_pad=True, activation=None),
#     nn.AdaptiveAvgPool2d(output_size=136),
#     Classifier1d(num_classes=10,is_multiselect=False, classifier_type='dense')
# )

gcd_model = torch.load('gcd_model_only_pytorch_mnist.pth')
#gcd_model=GcdNet_1()

gcd_model.to(device)
#gcd_model.load_state_dict(torch.load('gcd_model_pytorch_mnist.pth'))

gcd_optimizer = optim.Adam(gcd_model.parameters(),lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
gcd_mse_criterion=nn.MSELoss(reduction='mean')
gcd_ce_criterion=nn.CrossEntropyLoss(reduction='sum')
#flops_model = calculate_flops(model)
#print('flops_model:{0}'.format(flops_model))



flops_lenet_model = calculate_flops(lenet_model)
print('flops_{0}:{1}'.format('lenet_model', flops_lenet_model))

flops_gcd_model = calculate_flops(gcd_model)
print('flops_{0}:{1}'.format('gcd_model', flops_gcd_model))

#f = codecs.open('model_log_cifar100.txt', 'a', encoding='utf-8-sig')
#model = Function.load('Models/model5_cifar100.model')
#os.remove('model_log_cifar100_baseline_lr2.txt')
f = codecs.open('model_log_mnist_test3d.txt', 'a', encoding='utf-8-sig')
losses=[]
metrics=[]

gcd_losses=[]
gcd_metrics=[]
print('epoch start')
for epoch in range(num_epochs):
    mbs = 0
    while mbs<=1000:
        for  i ,(input, target) in enumerate(train_loader):
            input, target = Variable(input).to(device), Variable(target).to(device)
            lenet_output = lenet_model(input)

            lenet_loss=criterion(lenet_output,target)

            accu = 1-np.mean(np.not_equal(np.argmax(lenet_output.cpu().detach().numpy(), -1).astype(np.int64), target.cpu().detach().numpy().astype(np.int64)))
            losses.append(lenet_loss.item())
            metrics.append(accu)


            optimizer.zero_grad()
            lenet_loss.backward( retain_graph=True)
            optimizer.step()

            if mbs>=0:
                gcd_optimizer.zero_grad()
                gcd_output = gcd_model(input)
                gcd_loss =gcd_ce_criterion(gcd_output,target)+ gcd_mse_criterion(gcd_output, lenet_output)
                # if epoch>0 and mbs>100:
                #     result=gcd_output.cpu().detach().numpy()
                #     print(result)
                gcd_accu = 1 - np.mean(np.not_equal(np.argmax(gcd_output.cpu().detach().numpy(), -1).astype(np.int64),
                                                    target.cpu().detach().numpy().astype(np.int64)))
                gcd_losses.append(gcd_loss.item())
                gcd_metrics.append(gcd_accu)
                gcd_loss.backward( )
            gcd_optimizer.step()


            if mbs % 50 == 0:
                print("Baseline:     Epoch: {}/{} ".format(epoch + 1, num_epochs),
                      "Step: {} ".format(mbs),
                      "Loss: {:.4f}...".format(np.array(losses).mean()),
                      "Accuracy:{:.3%}...".format(np.array(metrics).mean()))
                f.writelines(['model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format('lanet_model', 0.01, epoch, mbs + 1, np.array(losses).mean(), np.array(metrics).mean())])


                losses=[]
                metrics=[]
                if mbs >=0:
                    print("Gcd_Conv    Epoch: {}/{} ".format(epoch + 1, num_epochs), "Step: {} ".format(mbs),
                          "Loss: {:.4f}...".format(np.array(gcd_losses).mean()),
                          "Accuracy:{:.3%}...".format(np.array(gcd_metrics).mean()))
                    f.writelines([
                        'model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format('gcd_model', 0.01, epoch, mbs + 1, np.array(gcd_losses).mean(), np.array(gcd_metrics).mean())])

                gcd_losses = []
                gcd_metrics = []
            if (mbs+1)%100==0:
                # torch.save(lenet_model.state_dict(),'lenet_model_pytorch_mnist_1.pth' )
                # torch.save(gcd_model.state_dict(), 'gcd_model_pytorch_mnist_1.pth')
                torch.save(lenet_model, 'lenet_model_only_pytorch_mnist.pth')
                torch.save(gcd_model, 'gcd_model_only_pytorch_mnist.pth')

            mbs += 1
