
dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


p_net=Sequential(
    Conv2d((3,3),10,strides=1,auto_pad=False,use_bias=True,name='conv1'),
    PRelu(),
    MaxPool2d((2,2),strides=2,auto_pad=False),
    Conv2d((3, 3), 16, strides=1, auto_pad=False,use_bias=True,name='conv2'),
    PRelu(),
    Conv2d((3,3),32,strides=1,auto_pad=False,use_bias=True,name='conv3'),
    PRelu(),
    Combine(
        Conv2d((1,1),1,strides=1,auto_pad=False,use_bias=True,activation='sigmoid',name='conv4_1'),
        Conv2d((1,1),4,strides=1,auto_pad=False,use_bias=True,name='conv4_2'),
        Conv2d((1,1),10,strides=1,auto_pad=False,use_bias=True,name='conv4_3')))
p_net.name='pnet'


r_net=Sequential(
    Conv2d((3,3),28,strides=1,auto_pad=False,use_bias=True,name='conv1'),
    PRelu(),
    MaxPool2d((3,3),strides=2,auto_pad=False),
    Conv2d((3, 3), 48, strides=1, auto_pad=False,use_bias=True,name='conv2'),
    PRelu(),
    Conv2d((3,3),64,strides=1,auto_pad=False,use_bias=True,name='conv3'),
    PRelu(),
    Flatten(),
    Dense(128,activation=None,use_bias=True,name='conv4'),
    PRelu(),
    Combine(
        Dense(1,activation='sigmoid',use_bias=True,name='conv5_1'),
        Dense(4,activation=None,use_bias=True,name='conv5_2'),
        Dense(10,activation=None,use_bias=True,name='conv5_3')))
r_net.name='rnet'


o_net=Sequential(
    Conv2d((3,3),32,strides=1,auto_pad=False,use_bias=True,name='conv1'),
    PRelu(),
    MaxPool2d((3,3),strides=2,auto_pad=False),
    Conv2d((3, 3), 64, strides=1, auto_pad=False,use_bias=True,name='conv2'),
    PRelu(),
    MaxPool2d((3,3),strides=2,auto_pad=False),
    Conv2d((3,3),64,strides=1,auto_pad=False,use_bias=True,name='conv3'),
    PRelu(),
    MaxPool2d((2, 2), strides=2,auto_pad=False),
    Conv2d((2, 2), 128, strides=1, auto_pad=False,use_bias=True,name='conv4'),
    PRelu(),
    Flatten(),
    Dense(256,activation=None,use_bias=True,name='conv5'),
    PRelu(),
    Combine(
        Dense(1,activation='sigmoid',use_bias=True,name='conv6_1'),
        Dense(4,activation=None,use_bias=True,name='conv6_2'),
        Dense(10,activation=None,use_bias=True,name='conv6_3')))
o_net.name='onet'


def Pnet(pretrained=True,
             input_shape=(3,12,12),
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3,12,12)
    pnet =ImageDetectionModel(input_shape=(3,12,12),output=p_net)
    pnet.preprocess_flow = [normalize(0, 255), image_backend_adaptive]
    if pretrained==True:
        download_file_from_google_drive('1w9ahipO8D9U1dAXMc2BewuL0UqIBYWSX',dirname,'pnet.pth')
        recovery_model=torch.load(os.path.join(dirname,'pnet.pth'))
        recovery_model.to(_device)
        pnet.model=recovery_model
    return pnet


def Rnet(pretrained=True,
             input_shape=(3,24,24),
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3,24,24)
    rnet =ImageDetectionModel(input_shape=(3,24,24),output=r_net)
    rnet.preprocess_flow = [normalize(0, 255), image_backend_adaptive]
    if pretrained==True:
        download_file_from_google_drive('1CH7z133_KrcWMx9zXAblMCV8luiQ3wph',dirname,'rnet.pth')
        recovery_model=torch.load(os.path.join(dirname,'rnet.pth'))
        recovery_model.to(_device)
        rnet.model=recovery_model
    return rnet

def Onet(pretrained=True,
             input_shape=(3,48,48),
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3,48,48)
    onet =ImageDetectionModel(input_shape=(3,24,24),output=r_net)
    onet.preprocess_flow = [normalize(0, 255), image_backend_adaptive]
    if pretrained==True:
        download_file_from_google_drive('1a1dAlSzJOAfIz77Ic38JMQJYWDG_b7-_',dirname,'onet.pth')
        recovery_model=torch.load(os.path.join(dirname,'onet.pth'))
        recovery_model.to(_device)
        onet.model=recovery_model
    return onet