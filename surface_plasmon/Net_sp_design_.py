import argparse
import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
MODEL_DIR = os.path.abspath("../models/")
sys.path.append(MODEL_DIR)  # To find local version of the library

import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from utils.plot_tensor import *
from Torch_config import *
from sp_set import surfae_plasmon_set,spp_film,guided_metal_cost
from spp_net import *
from sp_spectrum import *
import visdom
import sys
import cv2
from utils.visualize import *

import pickle
import subprocess

if False:
    import torchvision.models as models
else:
    import cadene_detector.cadene_models as models

#print(subprocess.check_output(['python -m visdom.server']))


def InitParser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',help='path to dataset')
    #parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', help='model architecture: ' +' | '.join(model_names) + ' (default: resnet34)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=300, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    print(parser)
    return parser

def GetTarget(dat_loader):
    target_list=[]
    for i, (input, target) in enumerate(dat_loader):
        t_1=target.cpu().detach().numpy()
        target_list.append(t_1)
    target = np.concatenate( target_list, axis=0 )      #https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
    return target

def GetGBDT_featrues(model_path,model,vis_title, train_loader,val_loader, criterion,opt):
    trainX, trainY=None,None

    checkpoint = torch.load(model_path)
    args.start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    model.gbdt_features = []
    acc_test,predicts,_ = validate(vis_title, val_loader, model,  0,opt)
    assert (acc_test == best_acc1)
    testX = np.concatenate(model.gbdt_features)
    testY = GetTarget(val_loader)
    #assert(len(predicts)==len(list(testY)))

    model.gbdt_features = []
    acc_train,_= validate(vis_title, train_loader, model,  0,opt)
    trainX = np.concatenate(model.gbdt_features)
    trainY = GetTarget(train_loader)

    assert (testX.shape[0] == testY.shape[0])
    pkl_path = "C:/CellLab/data/train={}_test={}_{}_.pickle".format(trainX.shape,testX.shape[0],vis_title)
    with open(pkl_path, "wb") as fp:  # Pickling
        pickle.dump([trainX,trainY,testX,testY], fp)

    return acc_test

class SPP_Torch(object):
    """ SPP by torch """

    def __init__(self, config,module):
        self.config = config
        self.model = module
        self.config.normal = "normal"
        self.check_model = None
        self.guided_cost_fun = guided_metal_cost
        self.best_acc1 = 0
        print(f"====== Parameters ={config.__dict__}\n")
        self.LoadData()
        pass

    def guided_(self,devices,metal_out, metal_true, P_thickness, thickness_true,cost_func):
        batch_size = metal_true.size(0)
        _, t1 = metal_out.topk(1, 2, True, True)
        P_metals = t1.view(batch_size, -1)
        cost_all_0,cost_all_1 = 0.,0.
        for i in range(batch_size):
            device = devices[i]
            t1 = np.asarray(device.thickness).astype(np.float32)
            t2 = thickness_true[i].cpu().numpy()
            isEqual = np.array_equal(t1,t2)
            assert(isEqual)
            P_metal = P_metals[i]
            P_thick = P_thickness[i]
            P_thick = P_thick.detach().cpu().numpy()
            isUpdate,cost_0,cost_1 = device.guided_update(self.config,P_metal.cpu().numpy(),P_thick,cost_func)
            cost_all_0+=cost_0;         cost_all_1+=cost_1
        return cost_all_0,cost_all_1


    
    def LoadData(self):
        """ Data loading code   """
        traindir = os.path.join(self.config.data, 'train/')
        valdir = os.path.join(self.config.data, 'test/')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
        self.train_dataset, val_dataset = surfae_plasmon_set(self.config, tte='train'), surfae_plasmon_set(self.config, tte='eval',nMaxFile=1000)
        self.train_dataset.scan_folders(traindir, self.config, adptive=True)
        if self.config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
    
        val_dataset.scan_folders(valdir, self.config, adptive=False)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False,
                                                 num_workers=self.config.workers, pin_memory=True)


    def Train_(self,check_model=None):
        #SGD确实偏慢
        #optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay, nesterov=True)
        optimizer = torch.optim.Adam(self.model.parameters(), args.lr,weight_decay=args.weight_decay)
        self.info = 'Huber_Adam'
        # optionally resume from a checkpoint
        if check_model is not None and os.path.isfile(check_model):
            print("=> loading checkpoint '{}'".format(check_model))
            self.check_model=check_model
            checkpoint = torch.load(check_model)
            args.start_epoch = checkpoint['epoch']
            self.best_acc1 = checkpoint['best_acc1']
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> best_acc1='{self.best_acc1}' (epoch {checkpoint['epoch']})")

        cudnn.benchmark = True
        if False and args.evaluate:
            self.validate(self.val_loader, self.model, args.start_epoch,args)
            return
        model_name = self.model.back_bone
        self.vis_title = "{}[{}]_most={}_lr={}".format(model_name,self.info,self.config.nMostCls,self.config.lr)
        self.dump_dir = "E:/MetaLab/dump/"
        vis = visdom.Visdom(env=self.vis_title)
        vis_cost_train = visdom.Visdom(env=self.vis_title+f"cost@train")
        vis_cost_test = visdom.Visdom(env=self.vis_title + f"cost@test")
    
        for epoch in range(self.config.start_epoch, self.config.epochs):
            if self.config.distributed:
                self.train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            if self.check_model is not None:
                pass
            else:
                self.train_dataset.AdaptiveSample(self.config.nMostCls)
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,num_workers=self.config.workers)
                self.train_core(self.train_loader, self.model, optimizer, epoch,vis_cost_train)
    
            # evaluate on validation set
            acc1,_,cost_val = self.validate(self.val_loader, self.model, epoch,self.config)
            vis_plot(self.config, vis, epoch, acc1,"SPP_net")

            vis_plot(self.config, vis_cost_test, epoch, cost_val, "cost@test")

            # remember best acc@1 and save checkpoint
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.model.back_bone,
                'state_dict': self.model.state_dict(),
                'best_acc1': self.best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


    def train_core(self,train_loader, model, optimizer, epoch,vis):
        args = self.config
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc_thick = AverageMeter()
        acc_metal = AverageMeter()
        trainset = train_loader.dataset
        nSamp_0,nSamp = trainset.__len__(),0
        # switch to train mode
        model.train()

        end = time.time()
        cost_all_0,cost_all_1,c0,c1=0,0,0,0,
        for i, (input, metal_true,thickness_true,indexs) in enumerate(train_loader):
            indexs = indexs.cpu().numpy().tolist()
            devices = [trainset.devices[i] for i in indexs]
            # measure data loading time
            data_time.update(time.time() - end)
            plot_batch_grid(input,self.dump_dir,"train",epoch,i)

            if args.gpu_device is not None:
                input = input.cuda(args.gpu_device, non_blocking=True)
            metal_true = metal_true.cuda(args.gpu_device, non_blocking=True)
            thickness_true = thickness_true.cuda(args.gpu_device, non_blocking=True)

            # compute output
            thickness, metal_out = model(input)
            loss = model.loss(thickness, thickness_true, metal_out, metal_true)
            if False:
                _, metal_max = torch.max(metal_out, 1)
                _, thickness_max = torch.max(thickness, 1)
                thickness_loss = model.thickness_criterion(thickness, thickness_true)
                if model.hybrid_metal>0:
                    metal_loss = args.metal_criterion(metal_out, metal_true)
                    loss = metal_loss * model.hybrid_metal + thickness_loss * (1 - model.hybrid_metal)
                else:
                    loss =thickness_loss


            # measure accuracy and record loss
            _thick,_metal,metal_pred = accuracy(metal_out, metal_true,thickness, thickness_true)
            losses.update(loss.item(), input.size(0))
            acc_thick.update(_thick, input.size(0))
            acc_metal.update(_metal, input.size(0))
            if self.guided_cost_fun is not None:
                c0,c1 = self.guided_(devices,metal_out, metal_true,thickness, thickness_true,self.guided_cost_fun)
                cost_all_0+=c0;      cost_all_1+=c1
                nSamp += input.shape[0]


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@thickness {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@metal type {top2.val:.3f} ({top2.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=acc_thick, top2=acc_metal))

            break
        if nSamp>0:
            vis_plot(self.config, vis, epoch, cost_all_0/nSamp, "cost_0@train")
            #vis_plot(self.config, vis, epoch, cost_all_1/nSamp, "cost_1@train")

#plot输出太累了
    def Plot_Compare(self,devices,epoch,batch,thickness_out, metal_out):
        #plot_batch_grid(input, self.dump_dir, "train", epoch, i)
        nSamp=metal_out.shape[0]
        #t_size=(256,256)
        #assert(len(input_imags)==nSamp)
        _,metal_out = metal_out.topk(1, 2, True, True)
        metal_out=metal_out.view(nSamp, -1)
        materials = ['au', 'ag', 'al', 'cu']
        #fig, ax = plt.subplots(nSamp,2)         #  figsize=(10, 5 * nSamp))
        #plot_batch_grid(input, self.dump_dir, "valid", epoch, batch)
        for j in range(nSamp):
            img_path, metal_labels = devices[j].path, devices[j].metal_labels()
            img_0 = cv2.imread(img_path)
            cv2.imwrite(f"{args.dump_dir}/{batch*100+j}_user_{devices[j].info}.jpg",img_0)
            #img_0 = cv2.resize(img_0, t_size)
            #ax[j,0].imshow(cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB))            ax[j,0].set_title(devices[j].info)

            if True:
                thickness=thickness_out[j,:]
                metal_nos=metal_out[j,:]
                metals=[]
                for type in metal_nos:
                    assert(type>=0 and type<4)
                    metals.append(materials[type])
                device = SP_device(thickness.cpu().numpy(), metals, 1, "", args)
                img_1 = device.HeatMap()
                if False:
                    cv2.imwrite(f"{args.dump_dir}/{batch * 100 + j}_pred_{device.title}.jpg", img_1)
                else:
                    img_1 = cv2.resize(img_1,(img_0.shape[1],img_0.shape[0]), interpolation=cv2.INTER_CUBIC)
                    #img_off =np.abs(img_1-img_0)
                    image_all = np.concatenate((img_0, img_1), axis=1)
                    cv2.imshow("", image_all);                    cv2.waitKey(0)
                    cv2.imwrite(f"{args.dump_dir}/{batch * 100 + j}_off_{device.title}.jpg", img_off)

                print("")
            else:
                img_1 = img_0
            #ax[j,1].imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))            ax[j,1].set_title(f"Predict")
            #cv2.imwrite(f"{args.dump_dir}/{j}_1.jpg",img_1)
            #break
        #path = f"{args.dump_dir}/Plot_Compare_{nSamp}_{epoch}_.jpg"        plt.savefig(path)
        print("")
            #cv2.imwrite(f"{args.dump_dir}/{j}_0.jpg", img)

    def Plot_Compare_1(self,devices,epoch,batch,thickness_out, metal_out):
        nSamp=metal_out.shape[0]
        _,metal_out = metal_out.topk(1, 2, True, True)
        metal_out=metal_out.view(nSamp, -1)
        materials = ['au', 'ag', 'al', 'cu']
        for j in range(nSamp):
            #if not(batch == 17 and j==8):   continue
            cmp_title = f"{batch * 100 + j}"
            thickness = thickness_out[j, :]
            metal_nos = metal_out[j, :]
            img_path, metal_labels = devices[j].path, devices[j].metal_labels()
            if self.guided_cost_fun is not None:
                P_metals,cur_metals = metal_out[j,:],devices[j].metal_labels()
                cost_1 = self.guided_cost_fun(cur_metals, devices[j].thickness)
                cost_2 = self.guided_cost_fun(P_metals.cpu().numpy(), thickness.cpu().numpy())
                delta = cost_1-cost_2
                cmp_title=cmp_title+f"_[{delta:.4g}]"
                print(f"\t{batch}-{j}\tcost=[{cost_1:.4g}-{cost_2:.4g}]\tdelta_cost={delta:.4g}/{delta*100/cost_1:.3g}% @{batch}:{j}")
            device_0 = SP_device(devices[j].thickness, devices[j].metal_types, 1, "", args)
            #img_0 = cv2.imread(img_path)

            metals=[]
            for type in metal_nos:
                assert(type>=0 and type<4)
                metals.append(materials[type])
            device = SP_device(thickness.cpu().numpy(), metals, 1, "", args)
            #path = f"{args.compare_dir}/{batch * 100 + j}_off_{device.title}.jpg"

            SPP_compare(device_0, device,cmp_title,"", args)
            del device_0,device
            gc.collect()
        print("")

    def validate(self,val_loader, model, epoch,opt,gbdt_features=None):
        valset = self.val_loader.dataset
        nVal = valset.__len__()
        args = self.config
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc_thick = AverageMeter()
        acc_metal = AverageMeter()
        cost_val=0

        # switch to evaluate mode
        model.eval()
        predicts=[]
        nClass=4
        accu_cls_=np.zeros(nClass)
        accu_cls_1 = np.zeros(nClass)
        with torch.no_grad():
            end = time.time()

            valset.isSaveItem = True
            for batch, (input, metal_true,thickness_true,indexs) in enumerate(val_loader):
                indexs = indexs.cpu().numpy().tolist()
                devices = [valset.devices[i] for i in indexs]
                if args.gpu_device is not None:
                    input = input.cuda(args.gpu_device, non_blocking=True)
                    metal_true = metal_true.cuda(args.gpu_device, non_blocking=True)
                    thickness_true = thickness_true.cuda(args.gpu_device, non_blocking=True)
                # compute output
                thickness, metal_out = model(input)
                loss = model.loss(thickness,thickness_true,metal_out,metal_true)
                if self.check_model is not None :
                    self.Plot_Compare_1(devices,epoch,batch,thickness, metal_out)
                    #self.Plot_Compare(devices, epoch, batch, thickness, metal_out)

                # measure accuracy and record loss
                _thick, _metal,metal_pred = accuracy(metal_out, metal_true,thickness, thickness_true)
                if False:        #each class by cys
                    for i in range(len(pred)):
                        cls = target[i]
                        accu_cls_[cls]=accu_cls_[cls]+1
                        if(pred[i]==cls):
                            accu_cls_1[cls] = accu_cls_1[cls] + 1
                losses.update(loss.item(), input.size(0))
                acc_thick.update(_thick, input.size(0))
                acc_metal.update(_metal, input.size(0))
                batch_size = metal_true.size(0)
                for i in range(batch_size):
                    c0 = self.guided_cost_fun(metal_pred[i], thickness[i])
                    cost_val+=c0
                valset.after_batch()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                break

            print(' * Acc@Thickness {top1.avg:.3f} Acc@Metal type {top5.avg:.3f}'.format(top1=acc_thick, top5=acc_metal))
        for i in range(nClass):
            cls=['au', 'ag', 'al', 'cu'][i]
            nz=(int)(accu_cls_[i])
            #print("{}-{}-{:.3g}".format(cls,nz,accu_cls_1[i]/nz),end=" ")
        print(f"cost_val={cost_val/nVal}")
        return acc_thick.avg,predicts,cost_val/nVal


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(metal_out, metal_true,thickness_out, thickness_true):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    p1 = None
    with torch.no_grad():
        batch_size = metal_true.size(0)
        thickness = thickness_out.cpu().data.numpy()
        thickness_true = thickness_true.cpu().data.numpy()
        thickness_accu = np.mean(np.abs(thickness - thickness_true))

        if metal_out is None:
            metal_accu=0
        else:
        #topk   A namedtuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor.
            _, t1 = metal_out.topk(1, 2, True, True)
            pred = t1.view(batch_size,-1)
            correct = pred.eq(metal_true)
            nEle  = correct.numel()
            correct_k = correct.view(-1).float().sum(0, keepdim=True)
            metal_accu = correct_k.mul_(100.0 / nEle).item()

            if False:    #each class accuracy by cys     5_1_2019
                _, pred_1 = metal_out.topk(1, 1, True, True)
                p1,t1=pred_1.t().cpu().numpy().squeeze(),metal_true.cpu().numpy()
                assert(p1.shape==t1.shape)

        return thickness_accu,metal_accu,pred
        #return res,p1


if __name__ == '__main__':
    parser=InitParser()
    args = parser.parse_args()
    args = ArgsOnSpectrum(args)

    if False:
        thickness=[6,19,6,76,8,8,5,25,5,9]
        device = SP_device(thickness,['ag', 'ag', 'au', 'cu','au'],1,"",args)
        device.HeatMap()
    config = TORCH_config(args)
    module = SPP_Model(config,nFilmLayer=10)
    learn = SPP_Torch(config, module)
    pretrained_model=None
    #pretrained_model = "E:/MetaLab/models/spp/spp_7_21.pth.tar"
    #pretrained_model = "E:/MetaLab/models/spp/spp_guided_8_16.pth.tar"
    #args.evaluate = True
    learn.Train_(check_model=pretrained_model)
    learn.Evaluate_()
