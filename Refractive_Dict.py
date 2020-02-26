from some_libs import *
import warnings
from jreftran_rt import *
from config import DefaultConfig
from Polynomial import *
from spec_gram import *
from numpy.linalg import norm

#refractive index
class Refractive_Dict(object):
    map2 = {}

    def __init__(self,config):
        self.dicts = {}
        self.config = config
        return

    def InitMap2(self, maters, lendas):
        for mater in maters:
            for lenda in lendas:
                self.map2[mater, lenda] = self.Get(mater, lenda, isInterPolate=True)
    #比较复杂，需要重新设计
    def Load2(self,material, path_R,path_I, scale=1):
        fb1 = np.loadtxt(path_R)
        fb2 = np.loadtxt(path_I)
        n_au = fb1[:, 1] + 1j * fb2[:, 1]
        print("{}@@@{} shape={}\n{}".format(material, path, df.shape, df.head()))

    def Load(self, material, path, scale=1):
        df = pd.read_csv(path, delimiter="\t", header=None, names=['lenda', 're', 'im'], na_values=0).fillna(0)
        if scale is not 1:
            df['lenda'] = df['lenda'] * scale
            df['lenda'] = df['lenda'].astype(int)
        self.dicts[material] = df
        rows, columns = df.shape
        # if columns==3:
        print("{}@@@{} shape={}\n{}".format(material, path, df.shape, df.head()))

    def Get(self, material, lenda, isInterPolate=False):
        # lenda = 1547
        n = 1 + 0j
        if material == "air":
            return n
        if material == "Si3N4":
            if self.config.model=='v1':
                return 2. + 0j
            else:
                return 2.46 + 0j
            #return 2.0
        assert self.dicts.get(material) is not None
        df = self.dicts[material]
        assert df is not None
        pos = df[df['lenda'] == lenda].index.tolist()
        # assert len(pos)>=1
        if len(pos) == 0:
            if isInterPolate:
                A = df.as_matrix(columns=['lenda'])  # df.as_matrix(columns=['lenda'])
                idx = (np.abs(A - lenda)).argmin()
                if idx == 0:
                    lenda_1, re_1, im_1 = df['lenda'].loc[idx], df['re'].loc[idx], df['im'].loc[idx]
                else:
                    lenda_1, re_1, im_1 = df['lenda'].loc[idx - 1], df['re'].loc[idx - 1], df['im'].loc[idx - 1]
                lenda_2, re_2, im_2 = df['lenda'].loc[idx], df['re'].loc[idx], df['im'].loc[idx]
                re = np.interp(lenda, [lenda_1, lenda_2], [re_1, re_2])
                im = np.interp(lenda, [lenda_1, lenda_2], [im_1, im_2])
            else:
                return None
        elif len(pos) > 1:
            re, im = df['re'].loc[pos[0]], df['im'].loc[pos[0]]
        else:
            re, im = df['re'].loc[pos[0]], df['im'].loc[pos[0]]
        n = re + im * 1j
        return n