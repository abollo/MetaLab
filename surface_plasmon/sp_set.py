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

class White_Normalize(object):
    def __init__(self,hei,wth,rotate_alg,normal_alg="CLAE_r"):
        self.hei = hei
        self.wth = wth
        self.T_crop = 0.3
        self.rotate_alg = rotate_alg
        self.isHisto = True
        self.normal_alg = normal_alg
        #self.crop_alg = crop_alg
        print("====== White_Normalize hei={},wth={},rotate_alg={},normal_alg={}".format(hei,wth,rotate_alg,self.normal_alg))

    def __call__(self,img):
        #return self.contrast_pil(img,self.degree)
        return self.normal_cv2(img)

    def normal_cv2(self,img):
        cv2_im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w, channels = cv2_im.shape
        if h!=self.hei or w!=self.wth:
            cv2_im = cv2.resize(cv2_im, (self.hei, self.wth))
        cv2_im = cell_normalize(cv2_im, "CLAE", self.rotate_alg)
        h, w, channels = cv2_im.shape
        gray = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        #kernel
        kernel = gray[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        kernel = clahe.apply(kernel)
        kernel = cv2.resize(kernel, (self.hei, self.wth))

        channels = [gray, cl,kernel]
        m_img = cv2.merge(channels)
        pil_im = Image.fromarray(cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB))
        #pil_im.save('normal_0.jpg')
        return pil_im

class Red_Normalize(object):
    def __init__(self,hei,wth,rotate_alg,crop_alg="random",normal_alg="CLAE_r"):
        self.hei = hei
        self.wth = wth
        self.T_crop = 0.3
        self.rotate_alg = rotate_alg
        self.isHisto = True
        self.normal_alg = normal_alg
        print("====== Red_Normalize hei={},wth={},rotate_alg={},normal_alg={}".format(hei,wth,rotate_alg,self.normal_alg))

    def __call__(self,img):
        #return self.contrast_pil(img,self.degree)
        return self.normal_cv2(img)

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

    def circle_normalize(self,img):
        h, w, channels = img.shape
        r_0=min(h,w)*0.4
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=(int)(r_0*0.8), maxRadius=(int)(r_0*1))
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('detected circles', img)
        cv2.waitKey(0)
        return img

    def normal_cv2(self,img):
        #img.save('normal_0.jpg')
        cv2_im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w, channels = cv2_im.shape
        #cv2_im=self.circle_normalize(cv2_im)
        #cv2_im = contour_normalize(cv2_im)
        cv2_im = cell_normalize(cv2_im,"CLAE",self.rotate_alg)
        if self.normal_alg=='CLAE_r_mix':
            cv2_im = cell_expand(cv2_im)
        pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
        #pil_im.save('normal_1.jpg')
        return pil_im

class Contrast(object):
    def __init__(self,degree):
        self.degree = degree
        self.isRotate = False
        self.isHisto = True

    def __call__(self,img):
        #return self.contrast_pil(img,self.degree)
        return self.contrast_cv2(img, self.degree)

    def contrast_pil(self,img,degree):
        #img.save('Contrast_0.jpg')
        angle = random.random() * 360
        if True:
            img = img.rotate(angle)
        else:
            opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
            center = tuple(np.array(img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        enh_contrast = ImageEnhance.Contrast(img)
        enh_contrast.enhance(degree)
        #img.save('Contrast_1.jpg')
        return img

    def contrast_cv2(self,img,degree):
        #img.save('Contrast_0.jpg')
        cv2_im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2_im = cell_normalize(cv2_im)
        pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
        #pil_im.save('Contrast_1.jpg')
        return pil_im

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


def path2df_(root_path,  extensions,class_name=None,rTest=0.5, tile=False):
    #check_dir(path + 'train_npz/')
    #check_dir(path + 'test_npz/')
    class_names = set()
    folds = [x[0] for x in os.walk(root_path)]
    if False:
        total_load = get_total_files(class_names, path=path)
        print(" ROOT@@@{}\tclass_names = {},total_files={}".format(path,class_names, total_load))
        nb_classes = len(class_names)
    tic_0 = time.time()
    for fold in folds:
        if fold==root_path:     continue
        files = os.listdir(fold)
        shuffle(files)
        nz,nTrain = 0,len(files)*(1-rTest)
        for file in files:
            nz=nz+1
            name, extension = os.path.splitext(file)
            if extension not in extensions:            continue
            tokens = name.split('_')
            cls = tokens[0]
            class_names.add(cls)

    return class_names

def list_dir(dir,extensions):
    files_0,files = os.listdir(dir),[]
    for file in files_0:
        name, extension = os.path.splitext(file)
        if extension not in extensions:
            continue
        files.append(file)
    return files

class class_info:
    def __init__(self, nam,id_=0):
        self.name = nam
        self.ID = id_
        self.items = []

    def __repr__(self):
        return "class_info"

    def __str__(self):
        #return str(self.__class__) + ": " + str(self.__dict__)
        return  "\n\t" + str(self.__dict__)

    def nz(self):
        nItem = len(self.items)
        return nItem

class CELL_TRANS(object):
    def __init__(self,params,tte):
        self.shape = params['input_shape']
        self.input_dim= self.shape[1]
        self.channel = self.shape[0]
        self.tte = tte
        self.cell=params['cell_type']
        self.params = params
        self.rotate_alg = "none"
        self.isHisto = True
        self.normal_alg = params['normal']
        normalize = TRANS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.cell=='white':
            if self.tte != 'train':
                self.rotate_alg = "none"            #"lift"
            else:
                self.rotate_alg = "random"          #"lift"
        elif self.cell=='red':
            if self.tte != 'train':
                self.rotate_alg = "none"
            else:
                self.rotate_alg = "random"


        print("====== Red_TRANS input_dim={}".format(self.input_dim))
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

        # cv2_im=self.circle_normalize(cv2_im)
        # cv2_im = contour_normalize(cv2_im)
        cv2_im = cell_normalize(cv2_im, "CLAE", self.rotate_alg)
        #cv2.imwrite("fastNlMeansDenoising_2.jpg", cv2_im)
        if True:  # self.normal_alg=='CLAE_r_mix':
            cv2_im = cell_expand(cv2_im,self.shape)
        #pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
        #tensor = tvF.to_tensor(pil_im)
        tensor = tvF.to_tensor(cv2_im)
        tensor = tvF.normalize(tensor, mean=[0.485, 0.456, 0.406, 0.406, 0.406, 0.406, 0.406, 0.406, 0.406],
                               std=[0.229, 0.224, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225])
        #plot_batch_grid(tensor,"G:/Beion/dump/{}".format('Red_TRANS_1'),"train",0,0)
        return tensor


class surfae_plasmon_set(data.Dataset):
    def __init__(self, params,tte='train', user_trans=None ):
        self.random_pick = None
        self.classes = []
        self.items = []

        self.Y = []
        self.key_map = {}
        self.tte = tte
        self.transform = user_trans

        normalize = TRANS.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        #self.transforms=CELL_TRANS(params,self.tte)
        return

    def AdaptiveSample(self,nMaxCls):
        self.items=[]
        nz0, nz1=1000000,0
        rand_path=""
        for cls in self.cls_infos:
            nz = len(self.items)
            if cls.nz()<=nMaxCls:
                self.items.extend(cls.items)
                if False:   #简单复制没啥用，令人吃惊       66.4%(EPOCH=3)  65.7%(EPOCH=4)
                    nLoop = (int)(nMaxCls / len(cls.items))
                    for i in range(nLoop-1):
                        self.items.extend(cls.items)
            else:
                random.shuffle(cls.items)
                if rand_path == "":
                    rand_path = cls.items[0]
                self.items.extend(cls.items[0:nMaxCls])
            nz0 = min(nz0,len(self.items)-nz)
            nz1 = max(nz1, len(self.items)-nz)

            #print("\t{}\tname=\"{}\"\t\tnz={}".format(cls.ID,cls.name,cls.nz()))
        print("AdaptiveSample ...nSample={} nz=[{}-{}] nMaxCls={} nCls={}".format(self.__len__(),nz0,nz1,nMaxCls,len(self.cls_infos)))
        print("AdaptiveSample ...rand_path={}".format(rand_path))

    def scan_folders(self, root_path, params, adptive=False,pkl_path=None):
        self.root = root_path    #'F:/AudioSet/audioData-gc/'
        self.pkl_path = pkl_path
        return
        # self.params = params
        # self.classes = classes
        nAllItem=0
        id,self.cls_infos= 0,[]
        for pose in  params['blood_cells']:
            self.cls_infos.append( class_info(pose, id) )
            id = id+1
        extensions=['.jpg']
        nz,nMinFile,nMaxFile = 0,0 ,params['nMostCls']
        folds = [x[0] for x in os.walk(root_path)]
        if False:
            total_load = get_total_files(class_names, path=path)
            print(" ROOT@@@{}\tclass_names = {},total_files={}".format(path, class_names, total_load))
            nb_classes = len(class_names)
        tic_0 = time.time()
        for fold in folds:
            if fold == root_path:     continue
            tokens = fold.split('/')
            cls = None
            for cls_info in self.cls_infos:
                if tokens[-1] == cls_info.name:
                    cls = cls_info
                    break
            if cls is None:
                print("\r{} is not in ANY CLASS!!!".format(fold + '/' + file), end="")
                continue
            files = os.listdir(fold)
            for file in files:
                nz = nz + 1
                name, extension = os.path.splitext(file)
                if extension not in extensions:            continue
                cls.items.append([fold + '/' + file, cls.ID])
                if not adptive:
                    self.items.append([fold+'/'+file, cls.ID])
                if cls.nz() > nMaxFile:
                    break
            nAllItem = nAllItem+cls.nz()

        self.tLoad = 0
        info = "".join([str(x) for x in self.cls_infos] )
        print( "====== scan_folders@ \"{}\" nAllItem={} self.items={}\n class_info=\n".format(self.root,nAllItem,self.__len__()) )
        for cls in self.cls_infos:
            print("\t{}\tname=\"{}\"\t\tnz={}".format(cls.ID,cls.name,cls.nz()))
        # self.scaler_(X_)
        # print("======ROOT={} CLASS={} N={}\tY_={}".format(root, self.class_names, self.nItem, self.Y_.shape))

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
        item = self.items[index]
        img_path,img_label = item[0],item[1]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, img_label
        #return tX_, tY_

    def __len__(self):
        nItem = len(self.items)
        return nItem



if __name__ == '__main__':
    if True:    #更专业的https://github.com/gveres/donateacry-corpus
        path = "D:\VideoP\ObjectTracking\BrushPose"
        classes = path2df_(path,['.jpg'],['Front','Top']  )

