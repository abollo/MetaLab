import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
# from resnet import resnet50
from copy import deepcopy
import numpy as np


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
        self.back_bone = 'resnet18'
        # model_name='dpn92'
        # model_name='senet154'
        # model_name='densenet121'
        # model_name='alexnet'
        # model_name='senet154'

    def __init__(self, config):
        super(SPP_Model, self).__init__()

        self.config = config
        self.pick_models()
        self.resNet = models.resnet18(pretrained=True)
        print(f"=> creating model '{self.back_bone}'")
        #self.use_gpu = torch.cuda.is_available()
        self.nLayer = 10
        self.nMostThick = 200
        self.nMetalType = 4
        self.hybrid_metal = 0

        self.fc1 = nn.Linear(512, 512)
        self.thickness_pred = nn.Linear(512, self.nLayer)

        self.fc2 = nn.Linear(512, 512)
        self.metal_cls_pred = nn.Linear(512, self.nMetalType)

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

    def loss(self,thickness,thickness_true,metal_out,metal_true):
        thickness_loss = self.thickness_criterion(thickness, thickness_true)
        if self.hybrid_metal > 0:
            metal_loss = self.metal_criterion(metal_out, metal_true)
            loss = metal_loss * self.hybrid_metal + thickness_loss * (1 - self.hybrid_metal)
        else:
            loss = thickness_loss
        return loss

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
        last_conv_out = self.resNet.avgpool(last_conv_out)
        last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

        thickness = F.relu(self.fc1(last_conv_out))
        thickness = self.thickness_pred(thickness)
        # thickness = F.softmax(self.thickness_pred(thickness), 1)

        metal_cls = F.relu(self.fc2(last_conv_out))
        if self.hybrid_metal > 0:
            metal_cls = self.metal_cls_pred(metal_cls)
        else:
            metal_cls = None

        return thickness, metal_cls

    def forward(self, x, return_atten_info=False):
        last1 = self.get_resnet_convs_out(x)
        thickness, metal_cls = self.get_device(last1)
        return thickness, metal_cls

    def evaluate(self, faces):
        preds = []
        weigh = np.linspace(1, self.nLayer, self.nLayer)

        for face in faces:
            face = Variable(torch.unsqueeze(face, 0))
            thickness, metal_cls = self.forward(face)

            metal_cls = F.softmax(metal_cls, 1)
            metal_prob, metal_pred = torch.max(metal_cls, 1)

            metal_pred = metal_pred.cpu().data.numpy()[0]
            metal_prob = metal_prob.cpu().data.numpy()[0]

            thickness_probs = thickness.cpu().data.numpy()
            thickness_probs.resize((self.nLayer,))

            # expectation and variance
            thickness_pred = sum(thickness_probs * weigh)
            thickness_var = np.square(np.mean(thickness_probs * np.square(weigh - thickness_pred)))

            preds.append([metal_pred, metal_prob, thickness_pred, thickness_var])
        return preds


if __name__ == "__main__":
    a = SPP_Model()
    print(f"SPP_Model={a}")
    pass
