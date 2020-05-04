import os
import numpy as np
from sp_spectrum import *
from sp_set import *
import cv2

'''
metal cost at https://www.bloomberg.com/markets/commodities/futures/metals 5/3/2020
    材料        彭博社报价(5/3/2020)                    密度                系数
    Au  1,700.90-USD/oz	    60 美元/克              19.32克/立方厘米        1159.2
    Ag  14.94-USD/oz        0.53 美元/克            10.53克/立方厘米        5.58
    Cu  5,110.00-USD/MT	    0.005 美元/克           8.9克/立方厘米	        0.045 
    Al	1,487.00-USD/MT	    0.0015 美元/克          2.7克/立方厘米	        0.004
    
online code to get the cost     https://repl.it/repls/PriceyStandardChords
    pau, pag, pal, pcu=300.0, 3.5, 0.05, 0.1        #不考虑密度
    pau, pag, pal, pcu=1159.2, 5.58, 0.045 , 0.004  #考虑密度
    au, ag, al, cu=0,0,0,0
    au+=5+5
    ag+=0
    al+=0
    cu+=5+5+5
    print(pau*au+ ag*ag + pal*al + pcu*cu)
'''
def guided_metal_cost(metals, thickness):
    price = [300.0, 3.5, 0.05, 0.1]         #不考虑密度
    price = [1159.2, 5.58, 0.045, 0.004]    #考虑密度       $/cm3
    cost = 0
    i = 0
    for metal in metals:
        thick = thickness[2*i]
        cost = cost+price[metal]*thick
        i = i+1

    return cost

guided_xitas=np.arange(30, 90, 1)
guided_xitas=np.concatenate((np.arange(30, 35, 1),np.arange(35, 50, 0.2), np.arange(50, 90, 1)))
class spp_film:
    #   '188nm [ag(6)_31_cu(9)_20_al(6)_26_cu(8)_31_au(6)_45_]_'
    def __init__(self,args, info,path,id_=0):
        self.args = args
        self.path = path
        self.info_history = []

        self.info = info
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
        self.thickness_base = sum(self.thickness)
        self.init_feat_roi(args)



    def init_feat_roi(self,args):
        feat_roi_path = f'{args.dump_dir}/feat_roi/{self.info}_.npy'
        if os.path.isfile(feat_roi_path):
            self.feat_roi = np.load(feat_roi_path)
        else:
            device = SP_device(self.thickness, self.metal_types, args.polarisation, "", args,roi_xitas=guided_xitas)
            self.feat_roi = device.R.astype(np.float16)
            np.save(feat_roi_path, self.feat_roi)
            del device
            gc.collect()
        cost_0 = guided_metal_cost(self.metal_labels(), self.thickness)
        self.info_history.append((self.info,cost_0))

    def guided_update(self,args,batch_no,no,P_metal_,P_thickness_,cost_func):        
        T_delta_off=0.1
        P_sum = P_thickness_.sum()
        cost_1 = cost_func(P_metal_, P_thickness_)
        cur_metal = self.metal_labels()
        cost_2 = cost_func(cur_metal, self.thickness)
        if True:
            if P_sum < self.thickness_base / 1.5 or P_sum > self.thickness_base * 1.5:
                return False,cost_2,cost_1
            cost_1 = cost_func(P_metal_, P_thickness_)
            cur_metal = self.metal_labels()
            cost_2 = cost_func(cur_metal, self.thickness)
            if cost_1>=cost_2:
                return False,cost_2,cost_1
            nMetal = len(cur_metal)
            nDiff=0
            for i in range(nMetal):
                if(P_metal_[i] != cur_metal[i]):    nDiff=nDiff+1
            if nDiff>1:
                return False,cost_2,cost_1
            t0=time.time()
            device = SP_device(P_thickness_, P_metal_, args.polarisation, "", args,roi_xitas=guided_xitas,feat_roi=self.feat_roi)
            if device.R is None:
                print(f"------{batch_no}::{no}",end="")
                return False,cost_2,cost_1
            assert (self.feat_roi.shape == device.R.shape)
            nCol=(int)(device.R.shape[1]/2)
            f0 = np.linalg.norm(self.feat_roi[:,0:nCol])
            delta_off = np.linalg.norm(device.R[:,0:nCol] - self.feat_roi[:,0:nCol]) / f0
            #print(f"------{batch_no}::{no}\t R_2D R={device.R.shape} nCol={nCol} time={time.time()-t0:.3f} \tdelta_off={delta_off:.2f} f0={f0:.2f}")
            
            if delta_off>T_delta_off:            
                del device
                return False,cost_2,cost_1

            del device
        self.metal_types = []
        for metal in P_metal_:
            self.metal_types.append(self.args.materials[metal])
        self.thickness=P_thickness_
        title = f"{len(self.info_history)}_{self.ID}__{self.info}"
        device = SP_device(self.thickness, self.metal_types, args.polarisation, "", args)
        _,self.path = device.HeatMap(title=title)
        self.info = device.title
        self.info_history.append((self.info,(float)(cost_1)))
        nInfo = len(self.info_history)
        print(f"guided_update {self.info_history[nInfo-2]}=>{self.info_history[nInfo-1]}")
        return True,cost_2,cost_1

    def metal_labels(self,x=0):
        labels=[]
        for metal in self.metal_types:
            type = self.args.dict_metal[metal]
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