import argparse
import os
import random
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
from sp_set import surfae_plasmon_set
import visdom
import sys
from utils.visualize import *
from utils.pytorch_env import *
import pickle

if False:
    import torchvision.models as models
else:
    import cadene_detector.cadene_models as models

[   'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
     'squeezenet1_0', 'squeezenet1_1',
     'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
#print(models.__dict__)
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print(model_names)
#pretrainedmodels   https://data.lip6.fr/cadene/pretrainedmodels/
model_names=['alexnet', 'bninception', 'cafferesnet101', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
             'dpn107', 'dpn131', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'fbresnet152',
             'inceptionresnetv2', 'inceptionv3', 'inceptionv4', 'nasnetalarge', 'nasnetamobile', 'pnasnet5large', 'polynet',
             'resnet101', 'resnet152', 'resnet18','resnet34', 'resnet50', 'resnext101_32x4d', 'resnext101_64x4d',
             'se_resnet101', 'se_resnet152', 'se_resnet50', 'se_resnext101_32x4d', 'se_resnext50_32x4d', 'senet154', 'squeezenet1_0', 'squeezenet1_1',
             'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'xception']

#model_name='cafferesnet101'
#model_name='resnet101'
#model_name='se_resnet50'
#model_name='vgg16_bn'
#model_name='vgg11_bn'
model_name='dpn68'      #learning rate=0.0001 效果较好
#model_name='cys_dpn'
#model_name='dpn92'
#model_name='senet154'
#model_name='densenet121'
#model_name='alexnet'
#model_name='senet154'
if True:
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default=model_name,
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet34)')
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
best_acc1 = 0

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
    acc_test,predicts = validate(vis_title, val_loader, model, criterion, 0,opt)
    assert (acc_test == best_acc1)
    testX = np.concatenate(model.gbdt_features)
    testY = GetTarget(val_loader)
    #assert(len(predicts)==len(list(testY)))

    model.gbdt_features = []
    acc_train,_= validate(vis_title, train_loader, model, criterion, 0,opt)
    trainX = np.concatenate(model.gbdt_features)
    trainY = GetTarget(train_loader)

    assert (testX.shape[0] == testY.shape[0])
    pkl_path = "C:/CellLab/data/train={}_test={}_{}_.pickle".format(trainX.shape,testX.shape[0],vis_title)
    with open(pkl_path, "wb") as fp:  # Pickling
        pickle.dump([trainX,trainY,testX,testY], fp)

    return acc_test

def main():
    global args, best_acc1
    args = parser.parse_args()
    args.nMostCls = 5000
    args.input_shape = [3, 224, 224]
    args.normal = 'gray_2_{}'.format(args.input_shape)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    args.pretrained = True
    if args.arch.startswith('cys'):
        if args.arch=='cys_dpn':
            args.pretrained=False
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=nClass,input_shape=args.input_shape,pretrained='')
            args.lr = 0.0005
    # create model
    elif args.pretrained:
        args.lr = 0.0001
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    args.gpu = 0
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train/')
    valdir = os.path.join(args.data, 'test/')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    args.gpu_device = pytorch_env(42)
    print("====== Parameters ={}".format(args.__dict__))

    train_dataset, val_dataset = surfae_plasmon_set(args, tte='train'), surfae_plasmon_set(args, tte='eval')
    # train_data.scan_folders('F:/AudioSet/train_npz/',classes, opt, opt.pkl_path + ".train", train=True)
    train_dataset.scan_folders(traindir, args,  adptive=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    val_dataset.scan_folders(valdir, args, adptive=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion,args)
        return

    vis_title = "{}[{}]_most={}_lr={}".format(model_name,args.normal,args.nMostCls,args.lr)
    vis = visdom.Visdom(env=vis_title)
    if False:
        train_dataset.AdaptiveSample(args.nMostCls)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
        acc1 = GetGBDT_featrues('C:/CellLab/model_best_(89.8）.pth.tar',model,vis_title,train_loader,val_loader,criterion,opt)
        #acc1 = GetGBDT_featrues('C:/CellLab/model_best(粒细胞).pth.tar', model, vis_title, train_loader, val_loader,criterion, opt)
        os._exit(-10)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_dataset.AdaptiveSample(args.nMostCls)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
        train(vis_title,train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1,_ = validate(vis_title,val_loader, model, criterion, epoch,args)
        vis_plot(args, vis, epoch, acc1,"SPP_net")
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(vis_title,train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        plot_batch_grid(input,"./dump/{}".format(vis_title),"train",epoch,i)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        [acc1, acc2],_ = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc2[0], input.size(0))

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
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top2=top5))



def validate(vis_title,val_loader, model, criterion, epoch,opt,gbdt_features=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    predicts=[]
    nClass=4
    accu_cls_=np.zeros(nClass)
    accu_cls_1 = np.zeros(nClass)
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            plot_batch_grid(input, "./dump/{}".format(vis_title), "valid", epoch, i)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            [acc1, acc5],pred = accuracy(output, target, topk=(1, 5))
            if True:        #each class by cys
                for i in range(len(pred)):
                    cls = target[i]
                    accu_cls_[cls]=accu_cls_[cls]+1
                    if(pred[i]==cls):
                        accu_cls_1[cls] = accu_cls_1[cls] + 1
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    for i in range(nClass):
        cls=['au', 'ag', 'al', 'cu'][i]
        nz=(int)(accu_cls_[i])
        print("{}-{}-{:.3g}".format(cls,nz,accu_cls_1[i]/nz),end=" ")
    print("err=".format(0))
    return top1.avg,predicts


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    p1 = None
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
    #topk   A namedtuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor.
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        t1 = target.view(1, -1).expand_as(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        if True:    #each class accuracy by cys     5_1_2019
            _, pred_1 = output.topk(1, 1, True, True)
            p1,t1=pred_1.t().cpu().numpy().squeeze(),target.cpu().numpy()
            assert(p1.shape==t1.shape)
        return res,p1


if __name__ == '__main__':
    main()
