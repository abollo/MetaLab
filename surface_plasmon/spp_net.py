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

"""
0 ResNet(
  0 (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  1(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  2 (relu): ReLU(inplace)
  3 (maxpool): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
  4 (stack1): Sequential 
  5 (stack2): Sequential 
  6 (stack3): Sequential 
  7 (stack4): Sequential 
  8 (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
  9 (fc): Linear(in_features=2048, out_features=2000, bias=True)
)
1 Linear(in_features=2048, out_features=512, bias=True)
2 Linear(in_features=512, out_features=99, bias=True)
3 Linear(in_features=2048, out_features=512, bias=True)
4 Linear(in_features=512, out_features=2, bias=True)
"""

class SPP_Model(torch.nn.Module, ):
  def __init__(self,config):
    super(SPP_Model, self).__init__()
    self.config = config
    self.resNet = models.resnet18(pretrained=True)

    self.use_gpu = torch.cuda.is_available()
    self.nLayer = 10
    self.nMetalType = 4

    self.fc1          = nn.Linear(512, 512)
    self.thickness_pred = nn.Linear(512, self.nLayer)

    self.fc2          = nn.Linear(512, 512)
    self.metal_cls_pred = nn.Linear(512, self.nMetalType)

  def get_resnet_convs_out(self, x):
    """
    get outputs from convolutional layers of ResNet
    :param x: image input
    :return: middle ouput from layer2, and final ouput from layer4
    """
    x = self.resNet.conv1(x)    # out = [N, 64, 112, 112]
    x = self.resNet.bn1(x)
    x = self.resNet.relu(x)
    x = self.resNet.maxpool(x)  # out = [N, 64, 56, 56]

    x = self.resNet.layer1(x)   # out = [N, 64, 56, 56]
    x = self.resNet.layer2(x)   # out = [N, 128, 28, 28]
    x = self.resNet.layer3(x)   # out = [N, 256, 14, 14]
    x = self.resNet.layer4(x)   # out = [N, 512, 7, 7]

    return x # out = [N, 512, 1 ,1]



  def get_device(self, last_conv_out):
    last_conv_out = self.resNet.avgpool(last_conv_out)
    last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

    thickness = F.relu(self.fc1(last_conv_out))
    thickness = F.softmax(self.thickness_pred(thickness), 1)

    metal_cls = F.relu(self.fc2(last_conv_out))
    metal_cls = self.metal_cls_pred(metal_cls)

    return thickness, metal_cls


  def forward(self, x, return_atten_info = False):
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
