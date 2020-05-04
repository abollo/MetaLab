import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
# from resnet import resnet50
from copy import deepcopy
import numpy as np
from Torch_config import *
from sp_visualize import *
import pickle



def image_transformer():
    """
    :return:  A transformer to convert a PIL image to a tensor image
              ready to feed into a neural network
    """
    return {
        'train': transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


# from layer_utils.roi_align.roi_align import CropAndResizeFunction


class SPP_Model(torch.nn.Module, ):
    def pick_models(self):
        model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))
        print(model_names)
        # pretrainedmodels   https://data.lip6.fr/cadene/pretrainedmodels/
        model_names = ['alexnet', 'bninception', 'cafferesnet101', 'densenet121', 'densenet161', 'densenet169',
                       'densenet201',
                       'dpn107', 'dpn131', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'fbresnet152',
                       'inceptionresnetv2', 'inceptionv3', 'inceptionv4', 'nasnetalarge', 'nasnetamobile',
                       'pnasnet5large',
                       'polynet',
                       'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x4d',
                       'resnext101_64x4d',
                       'se_resnet101', 'se_resnet152', 'se_resnet50', 'se_resnext101_32x4d', 'se_resnext50_32x4d',
                       'senet154', 'squeezenet1_0', 'squeezenet1_1',
                       'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'xception']

        # model_name='cafferesnet101'
        # model_name='resnet101'
        # model_name='se_resnet50'
        # model_name='vgg16_bn'
        # model_name='vgg11_bn'
        # model_name='dpn68'      #learning rate=0.0001 效果较好
        self.back_bone = 'resnet18_x'
        #self.back_bone = 'resnet34'
        # model_name='dpn92'
        # model_name='senet154'
        # model_name='densenet121'
        # model_name='alexnet'
        # model_name='senet154'

    def __init__(self, config,nFilmLayer=10):
        super(SPP_Model, self).__init__()

        self.config = config
        self.pick_models()
        
        print(f"=> creating model '{self.back_bone}'")
        #self.use_gpu = torch.cuda.is_available()
        self.nFilmLayer = nFilmLayer
        self.nMetalLayer = (int)(self.nFilmLayer/2)
        self.nMostThick = 200
        self.nMetalType = 4
        self.hybrid_metal = 0.1
        #self.activations = []
        self.activations = None

        if self.back_bone == 'resnet18_x':
            self.resNet = models.resnet18(pretrained=True)
            self.fc1 = nn.Linear(512, 512)
        else:
            self.resNet = models.resnet50(pretrained=True)
            self.fc1 = nn.Linear(2048, 512)
        self.thickness_pred = nn.Linear(512, self.nFilmLayer)

        #self.fc2 = nn.Linear(512, 512)
        self.metal_types=[]
        for i in range(self.nMetalLayer):
            metal_type = nn.Linear(512, self.nMetalType)
            self.metal_types.append(metal_type)
        self.metal_types=nn.ModuleList(self.metal_types)

        self.thickness_criterion = nn.L1Loss()
        self.thickness_criterion = nn.SmoothL1Loss()
        self.metal_criterion = nn.CrossEntropyLoss()

        if config.gpu_device is not None:
            self.cuda(config.gpu_device)
            print(next(self.parameters()).device)
            self.thickness_criterion = self.thickness_criterion.cuda()
            self.metal_criterion = self.metal_criterion.cuda()
        elif config.distributed:
            self.cuda()
            self = torch.nn.parallel.DistributedDataParallel(self)
        else:
            self = torch.nn.DataParallel(self).cuda()
        print(f"{self}\nthickness_criterion={self.thickness_criterion}\nmetal_criterion={self.metal_criterion}")

    def loss(self,thickness,thickness_true,metal_out,metal_true,cost_func=None):
        nSample = metal_out.shape[0]
        thickness_loss = self.thickness_criterion(thickness, thickness_true)
        _, t1 = metal_out.topk(1, 2, True, True)
        metal_cls = t1.view(nSample,-1)
        cost_loss = 0
        if self.hybrid_metal > 0:
            metal_loss = 0
            for i in range(nSample):
                for j in range(self.nMetalLayer):
                    t1 = metal_out[i][j].unsqueeze(0)
                    t2 = metal_true[i][j].unsqueeze(0)
                    metal_loss = metal_loss+self.metal_criterion(t1,t2)
                #cost_loss = cost_loss+cost_func(metal_cls[i], thickness[i])
            loss = metal_loss * self.hybrid_metal + thickness_loss * (1 - self.hybrid_metal)
        else:
            loss = thickness_loss
        return loss

    def save_acti(self,x,name):
        acti = x.cpu().data.numpy()
        self.activations.append({'name':name,'shape':acti.shape,'activation':acti})

    def get_resnet_all_out(self, x):
        self.activations=[]
        self.save_acti(x, "input")
        x = self.resNet.conv1(x)  # out = [N, 64, 112, 112]
        self.save_acti(x,"conv1")
        x = self.resNet.bn1(x)
        x = self.resNet.relu(x)
        x = self.resNet.maxpool(x)  # out = [N, 64, 56, 56]

        x = self.resNet.layer1(x)  # out = [N, 64, 56, 56]
        self.save_acti(x,"layer1")
        x = self.resNet.layer2(x)  # out = [N, 128, 28, 28]
        self.save_acti(x, "layer2")
        x = self.resNet.layer3(x)  # out = [N, 256, 14, 14]
        self.save_acti(x, "layer3")
        x = self.resNet.layer4(x)  # out = [N, 512, 7, 7]
        self.save_acti(x, "layer4")
        return x  # out = [N, 512, 1 ,1]

    def get_resnet_convs_out(self, x):
        """
        get outputs from convolutional layers of ResNet
        :param x: image input
        :return: middle ouput from layer2, and final ouput from layer4
        """
        x = self.resNet.conv1(x)  # out = [N, 64, 112, 112]
        x = self.resNet.bn1(x)
        x = self.resNet.relu(x)
        x = self.resNet.maxpool(x)  # out = [N, 64, 56, 56]

        x = self.resNet.layer1(x)  # out = [N, 64, 56, 56]
        x = self.resNet.layer2(x)  # out = [N, 128, 28, 28]
        x = self.resNet.layer3(x)  # out = [N, 256, 14, 14]
        x = self.resNet.layer4(x)  # out = [N, 512, 7, 7]

        return x  # out = [N, 512, 1 ,1]

    def get_device(self, last_conv_out):
        nSample = last_conv_out.shape[0]
        last_conv_out = self.resNet.avgpool(last_conv_out)
        last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

        fc1_out = F.relu(self.fc1(last_conv_out))
        thickness = self.thickness_pred(fc1_out)
        # thickness = F.softmax(self.thickness_pred(thickness), 1)

        #fc2_out = F.relu(self.fc2(last_conv_out))
        if self.hybrid_metal > 0:
            metal_cls = torch.zeros((nSample,self.nMetalLayer,self.nMetalType))
            if self.config.gpu_device is not None:
                metal_cls = metal_cls.cuda(self.config.gpu_device, non_blocking=True)
            for i in range(self.nMetalLayer):
                cur_cls = self.metal_types[i](fc1_out)
                metal_cls[:,i,:] = cur_cls
        else:
            metal_cls = None

        return thickness, metal_cls

    def forward(self, x, return_atten_info=False):
        if self.activations is not None:
            last1 = self.get_resnet_all_out(x)
            with open(pkl_acti_visual, 'wb') as f:
                pickle.dump(self.activations, f, protocol=-1)
        else:
            last1 = self.get_resnet_convs_out(x)
        thickness, metal_cls = self.get_device(last1)
        return thickness, metal_cls

    def evaluate(self, faces):
        preds = []
        weigh = np.linspace(1, self.nFilmLayer, self.nFilmLayer)

        for face in faces:
            face = Variable(torch.unsqueeze(face, 0))
            thickness, metal_cls = self.forward(face)

            metal_cls = F.softmax(metal_cls, 1)
            metal_prob, metal_pred = torch.max(metal_cls, 1)

            metal_pred = metal_pred.cpu().data.numpy()[0]
            metal_prob = metal_prob.cpu().data.numpy()[0]

            thickness_probs = thickness.cpu().data.numpy()
            thickness_probs.resize((self.nFilmLayer,))

            # expectation and variance
            thickness_pred = sum(thickness_probs * weigh)
            thickness_var = np.square(np.mean(thickness_probs * np.square(weigh - thickness_pred)))

            preds.append([metal_pred, metal_prob, thickness_pred, thickness_var])
        return preds


if __name__ == "__main__":
    if False:       #only for debug
        t0 = torch.tensor([0.3]).cuda()
        a = t0.item()
        t0 = torch.zeros((16, 5, 4))
        f = nn.CrossEntropyLoss()
        t1 = t0[1,1]
        t1 = t1.unsqueeze(0)
        x = torch.tensor([[0.0000,  0.0000,  0.1527]])
        a = f( x,torch.tensor([2]))
        a = f(t1,torch.tensor(2))
        cur_cls = torch.ones((16,4))
        t0[:,1,:] = cur_cls
    config = TORCH_config(None)

    a = SPP_Model(config,nFilmLayer=10)
    print(f"SPP_Model={a}")
    pass

