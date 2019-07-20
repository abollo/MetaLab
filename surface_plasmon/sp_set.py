# python -m visdom.server

import torch
import os
import random
import pandas as pd
import time
import cv2
import gc
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from random import shuffle
import numpy as np
from torchvision import  transforms as TRANS
from torchvision.transforms import functional as tvF
from PIL import Image
from PIL import ImageEnhance
#from utils.visualize import *
from utils.plot_tensor import *
import re

'''
    from albumentations import (
        HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
        IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
        IAASharpen, IAAEmboss, Flip, OneOf, Compose
    )
'''

class AUG_white_0(object):
    def __init__(self,isTRain, p=.5):
        if isTRain:
            self.aug = Compose([
                CLAHE(),
                #RandomRotate90(),
                Transpose(),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=45, p=.75),
                Blur(blur_limit=3),
                OpticalDistortion(),
                GridDistortion(),
                HueSaturationValue()
            ], p=p)
        else:
            self.aug = Compose([CLAHE()], p=1)

    def __call__(self,img):
        cv2_im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image = self.aug(image=cv2_im)['image']
        pil_im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return pil_im


#参见 F:/Downloads/pytorch-best-practice-master/data/dataset.py

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def check_dir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    return dir_

def get_total_files(class_names, path=None):
    sum_total = 0
    for subdir in class_names:
        files = os.listdir(path + subdir)
        n_files = len(files)
        sum_total += n_files
    return sum_total


def list_dir(dir,extensions):
    files_0,files = os.listdir(dir),[]
    for file in files_0:
        name, extension = os.path.splitext(file)
        if extension not in extensions:
            continue
        files.append(file)
    return files

dict_metal={
    'au':0, 'ag':1, 'al':2, 'cu':3
}
class device_info:
    #   '188nm [ag(6)_31_cu(9)_20_al(6)_26_cu(8)_31_au(6)_45_]_'
    def __init__(self, info,path,id_=0):
        self.path = path

        info = info.replace('_', ' ').replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
        #tokens = re.split(" _\'('\')'\'['\']'", info)
        tokens = info.split( )
        self.ID = id_
        self.metal_types = []       #only for some layers
        self.thickness = []
        nLayer = (int)((len(tokens)-1)/3)
        for i in range(nLayer):
            self.metal_types.append(tokens[3*i+1])
            h1,h2=(float)(tokens[3 * i + 2]),(float)(tokens[3 * i + 3])
            self.thickness.append(h1)
            self.thickness.append(h2)

    def metal_labels(self,x=0):
        labels=[]
        for metal in self.metal_types:
            type = dict_metal[metal]
            labels.append(type)
        return labels

    def __repr__(self):
        return "class_info"

    def __str__(self):
        #return str(self.__class__) + ": " + str(self.__dict__)
        return  "\n\t" + str(self.__dict__)

    def nz(self):
        nItem = len(self.items)
        return nItem

class SPP_TRANS(object):
    def __init__(self,params,tte):
        self.params = params
        self.shape = params.input_shape
        self.input_dim= self.shape[1]
        self.channel = self.shape[0]
        self.tte = tte

        self.rotate_alg = "none"
        self.isHisto = True
        self.normal_alg = params.normal
        normalize = TRANS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if False:
            if self.tte != 'train':
                self.rotate_alg = "none"
            else:
                self.rotate_alg = "random"


        print("====== SPP_TRANS input_dim={}".format(self.input_dim))
    def Crop(self,img):
        cv2_im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w, channels = cv2_im.shape
        zoom_0 = h * 1.0 / self.hei
        if self.hei != h or self.wth != w:
            assert zoom_0 > 1
            zoom_0 = zoom_0 * 1.1
            s = random.uniform(1.0 / zoom_0, 1)
            h1, w1 = min(h, self.hei * s), min(w, self.wth * s)
            h1, w1 = int(h1), int(w1)
            y = h - h1;
            x = w - w1
            assert (y >= 0 and x >= 0)
            y = y * random.uniform(0, 1);
            x = x * random.uniform(0, 1)
            x, y = int(x), int(y)
            cv2_im = cv2_im[y:y + h1, x:x + w1]
            cv2_im = cv2.resize(cv2_im, (self.hei, self.wth))

    def __call__(self,img):
        #if self.tte != 'train':
        #   return self.transforms(img)
        cv2_im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2_im = cv2.resize(cv2_im, (self.input_dim, self.input_dim), interpolation=cv2.INTER_CUBIC)
        h, w, channels = cv2_im.shape
        #cv2.imwrite("fastNlMeansDenoising_1.jpg",cv2_im)
        #cv2.fastNlMeansDenoising(cv2_im, cv2_im, 5);       #h 值高可以很好的去除噪声,但也会把图像的细节抹去
        #cv2_im = cv2.bilateralFilter(cv2_im, -11, 17, 17)

        tensor = tvF.to_tensor(cv2_im)
        tensor = tvF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        #plot_batch_grid(tensor,"./dump/{}".format('Red_TRANS_1'),"train",0,0)
        return tensor


class surfae_plasmon_set(data.Dataset):
    def __init__(self, params,tte='train', user_trans=None ):
        self.random_pick = None
        self.classes = []
        self.devices = []

        self.Y = []
        self.key_map = {}
        self.tte = tte
        self.transform = user_trans

        normalize = TRANS.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        self.transforms=SPP_TRANS(params,self.tte)
        return

    def AdaptiveSample(self,nMaxCls):
        pass

    def scan_folders(self, root_path, params, adptive=False,pkl_path=None):
        self.root = root_path
        self.pkl_path = pkl_path

        nAllItem=0
        id,self.devices= 0,[]
        extensions=['.jpg']
        nz,nMinFile,nMaxFile = 0,0 ,params.nMostCls
        files = os.listdir(self.root)
        for file in files:
            nz = nz + 1
            name, extension = os.path.splitext(file)
            if extension not in extensions:            continue
            try:
                device = device_info(name,f"{self.root}/{file}")
            except:
                print(f"Failed to load device@{self.root}/{file}")
                continue
            self.devices.append(device)
            if len(self.devices) > nMaxFile:
                break
        #nAllItem = nAllItem+cls.nz()

        self.tLoad = 0
        print( "====== scan_folders@ \"{}\" nItem={}\n ".format(self.root,self.__len__()) )


    def gen_rand(self,seed):
        random.seed(seed)
        self.random_pick=[random.uniform(0, 1) for _ in range(self.__len__())]

    def get_imbalance(self, config_, rPosi=0.5):  # Get random selection of data for batch GD. Upsample positive classes to make it balanced in the training batch
        batch_size = config_.batch_size
        nPosi = int(np.round(batch_size * rPosi));
        assert nPosi > 0 and nPosi < batch_size
        set_0=[i for i, x in enumerate(self.Y) if x == 0];  set_1=[i for i, x in enumerate(self.Y) if x == 1]
        pos_idx = random.sample( set_0, nPosi)
        neg_idx = random.sample( set_1, batch_size-nPosi)
        idx = np.concatenate([pos_idx, neg_idx])
        tY_ = np.zeros((batch_size))
        tX_ = np.zeros((batch_size, 1,self.nFilter, self.nSegment), dtype=np.float32)
        nz=0
        for id in idx:
            t = (int)(self.Y[id])
            tY_[nz] = t
            s0, s1, melgram = self.get_melgram(id)
            tX_[nz,0,:,0:s1-s0] = melgram[:, s0:s1]
            nz=nz+1
        tX, tY=torch.Tensor(tX_), torch.LongTensor(tY_)
        return tX,tY

    def pick_slice_T(self,path,nCol,split,melgram,thrsh):
        loop=1
        while True:
            s0 = int((nCol - self.nSegment)*split)
            s1 = s0 + self.nSegment
            X = melgram[:, s0:s1]
            mn, mx = X.min(), X.max()
            loop = loop+1
            if abs(mx - mn) > thrsh or loop>40:
                break
            split = random.uniform(0, 1)
        if abs(mx-mn)<=thrsh:
            print("Strange@{} mx={},mn={},thrsh={}".format(path,mx,mn,thrsh))
            assert 0
        return s0

    def __getitem__(self, index):
        t0=time.time()
        nItem=self.__len__()
        assert self.random_pick is None or len(self.random_pick)==nItem
        assert (index >= 0 and index < nItem)
        device = self.devices[index]
        img_path,metal_labels = device.path,device.metal_labels()
        data = Image.open(img_path)
        data = self.transforms(data)
        #metal_labels = metal_labels[0]
        metal_labels = np.asarray(metal_labels).astype(np.int64)
        thickness = np.asarray(device.thickness).astype(np.float32)
        return data, metal_labels,thickness
        #return tX_, tY_

    def __len__(self):
        nItem = len(self.devices)
        return nItem



if __name__ == '__main__':
    pass

