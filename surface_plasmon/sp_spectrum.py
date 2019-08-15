import sys
import os
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from some_libs import *
from jreftran_rt import *
import cmath
import argparse
import pickle
from datetime import datetime
import gzip
import cv2
import scipy
from numba import jit
import numbers


def fig2data(fig):
    fig.canvas.draw()
    if True:  # https://stackoverflow.com/questions/42603161/convert-an-image-shown-in-python-into-an-opencv-image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf

def SPP_compare(device_0,device_1,title,path,args):
    p=2 #minkowski distance (p=2 for Euclidean distance):
    #dist = scipy.spatial.distance.cdist(device_0.R, device_1.R, 'minkowski', p)
    nrm_0 = np.linalg.norm(device_0.R)
    dist = np.linalg.norm(device_0.R-device_1.R)/nrm_0
    if path is None or len(path) == 0:
        path = f"{args.compare_dir}/[{dist*100:.2g}]_{title}_.jpg"
    img_0,_ = device_0.HeatMap(noAxis=False,title=f"Target\n{device_0.title}",cbar=False)
    img_1,_ = device_1.HeatMap(noAxis=False,title=f"Prediction\n{device_1.title}",cbar=False)
    diff = np.abs(device_1.R - device_0.R)
    relative_diff=np.where(device_0.R==0, 0, diff/device_0.R) #diff/device_0.R
    max=diff.max()
    avg=diff.mean()
    desc=f"Difference\nmean={diff.mean():.2g} median={np.median(diff):.2g} max={diff.max():.2g}"
    img_diff,_ = device_0.HeatMap(R_=diff,noAxis=False,title=desc,cbar=True)
    image_all = np.concatenate((img_0, img_1,img_diff), axis=1)
    #cv2.imshow("", image_all);    cv2.waitKey(0)
    cv2.imwrite(path,image_all)
    del image_all,img_0, img_1,img_diff


def N_maters_dict(args,materials = ['au', 'ag', 'al', 'cu']):
    dict = {}
    for mater in materials:
        N_r_path, N_i_path = '{}/{}_n1.txt'.format(args.mater_file_dir,mater), '{}/{}_k1.txt'.format(args.mater_file_dir,mater)
        fb1 = np.loadtxt(N_r_path)
        fb2 = np.loadtxt(N_i_path)
        n_au = fb1[:, 1] + 1j * fb2[:, 1]
        print(N_r_path, N_i_path)
        dict[mater]=n_au
    return dict

class SP_device(object):
    def SaveR(self):
        if True:
            path = '{}/{}/{}_.npz'.format(args.dump_dir, mType, title)
            np.savez_compressed(path, a=R, b=mType, c=title, d=args)
            if False:
                f = gzip.open(path, 'wb')
                pickle.dump([R, mType, title], f)
                f.close()
                with open(path, 'wb') as f:
                    pickle.dump([R, mType, title], f, protocol=-1)
                with open(path, 'rb') as f:
                    R, mType, title = pickle.load(f)
            del R;
            gc.collect()
            loaded = np.load(path)
            R, mType, title, args = loaded['a'], loaded['b'], loaded['c'], loaded['d']
            args = args[0]

    '''
        sns.heatmap 很难用，需用自定义，参见https://stackoverflow.com/questions/53248186/custom-ticks-for-seaborn-heatmap
    '''
    def HeatMap(self, R_=None,title="", noAxis=True,cbar=True):
        sns.set(font_scale=2)
        args,data = self.args,self.R
        if R_ is not None:
            assert(R_.shape == data.shape)
            data = R_
        #xitas = args.xitas
        ticks = np.linspace(0, 1, 10)
        xlabels = [int(i) for i in np.linspace(300, 2000, 10)]
        x0, x1 = self.xitas.min(), self.xitas.max()
        ylabels = ["{:.3g}".format(i) for i in np.linspace(x0, x1, 10)]

        s = max(data.shape[1] / args.dpi, data.shape[0] / args.dpi)

        #fig.set_size_inches(18.5, 10.5)
        cmap = 'coolwarm'  # "plasma"  #https://matplotlib.org/examples/color/colormaps_reference.html
        #cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
        if noAxis:          # tight samples for training(No text!!!)
            figsize = (s , s )
            fig, ax = plt.subplots(figsize=figsize, dpi=args.dpi)
            ax = sns.heatmap(data, ax=ax, cmap=cmap,cbar=False, xticklabels=False, yticklabels=False)
            path = '{}/all/{}_.jpg'.format(args.dump_dir,title)
            fig.savefig(path,bbox_inches='tight', pad_inches = 0)
            image = cv2.imread(path)
            #image = fig2data(ax.get_figure())      #会放大尺寸，难以理解
            assert(image.shape==(693,697,3))        #必须固定一个尺寸
            #cv2.imshow("",image);       cv2.waitKey(0)
            plt.close("all")
            return image,path
        else:       # for paper
            figsize = (s * 1.1, s * 1.1)
            fig, ax = plt.subplots(figsize=figsize, dpi=args.dpi)  # more concise than plt.figure:
            if title is None or len(title)==0:
                ax.set_title('Reflectance\n{}'.format(self.title))
            else:
                ax.set_title(title)
            #cbar_kws={'label': 'Reflex', 'orientation': 'horizontal'}
            # sns.set(font_scale=0.2)
            #  cbar_kws={'label': 'Reflex', 'orientation': 'horizontal'} , center=0.6
            #ax = sns.heatmap(data, ax=ax, cmap=cmap,yticklabels=ylabels[::-1],xticklabels=xlabels)
            #cbar_kws = dict(ticks=np.linspace(0, 1, 10))
            ax = sns.heatmap(data, ax=ax, cmap=cmap,vmin=0, vmax=1,cbar=cbar)
            plt.ylabel('Incident Angle');       plt.xlabel('Wavelength(nm)')

            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels[::-1])
            y_limit = ax.get_ylim();
            x_limit = ax.get_xlim()
            ax.set_yticks(ticks * y_limit[0])
            ax.set_xticks(ticks * x_limit[1])
            if False:
                path = '{}/{}/{}_[{}].jpg'.format(args.dump_dir,mType,title,data.shape)
                plt.show(block=True)
               #plt.savefig(path,bbox_inches='tight')

            image = fig2data(ax.get_figure())
            plt.close("all")
            return image,""
    plt.close("all")
        #plt.show()


#attenuated total reflection (ATR) spectrum
    def R_2D(self,args,polarisation):
        d, n_data = self.thicks, self.n_data
        nXita = len(self.xitas)
        nLenda = len(args.lendas)
        self.R = np.zeros((nXita - 1, nLenda))
        M0 = np.zeros((2, 2, d.shape[0]), dtype=complex)
        M_t0 = np.identity(2, dtype=complex)
        print("")
        for i in range(nLenda):  # wavelength's number
            for j in range(nXita - 1):  # angle
                M = M0;
                M_t = M_t0
                row = (int)(nXita - 2 - j);
                col = (int)(i)  # 数据格式原因
                r = jreftran_R(args.lendas[i], d, n_data[i, :], self.xitas[j], polarisation, M, M_t)
                if np.isnan(r):
                    print("nan@[{},{}]".format(row, col))
                self.R[row, col] = r
        del M0,M_t0
        gc.collect()

    def __del__(self):
        del self.n_data,
        gc.collect()

    def __init__(self,thickness,metals,polarisation,title,args,roi_xitas=None):
        assert( 'N_dict' in args.__dict__)
        N_dict = args.N_dict
        self.args = args
        self.polarisation = polarisation
        nm = 1.0e-9
        epsilon0 = 1.0 / (36 * np.pi) * nm                      # dielectric constant of the free space
        # lendas = np.arange(300,2010,10)*nm               #300:10:2000[nm]
        nLenda =len(args.lendas)
        #t_0 = np.arange(0,90,0.2)

        #xitas = np.arange(0, 90, 0.1)
        #xitas = np.arange(30, 60, 0.001)
        self.xitas = args.xitas_base if roi_xitas is None else roi_xitas
        nXita = len(self.xitas)
        N = len(thickness)         #N_al + N_au
        k = 0
        self.thicks=np.zeros(N+2)
        self.maters=['prism']
        self.n_data=np.zeros((nLenda,N+2),dtype=np.complex64)
        d, n_data = self.thicks, self.n_data
        d[0] = np.nan
        n_data[:, 0] = 1.77  # Al2O3
        for i in range(N): #layer
            if i%2 == 0:    #metals
                d[i + 1] = thickness[i] * nm         #% [m]
                if metals is not None:
                    mater= metals[(int)(i/2)]
                    if isinstance(mater, numbers.Integral):
                        mater = args.materials[mater]
                    self.maters.append(mater)
                    #type = (int)(np.random.uniform(0, len(materials)))
                    n_au = N_dict[mater]
                    title=title+"{}({})_".format(mater,(int)(thickness[i]))
                n_data[:, i + 1] = n_au[:]
            else:       #   dielectric
                d[i + 1] = thickness[i]  * nm         #% [m]
                n_data[:, i + 1] = 1.46         #SiO2
                title = title + "{}_".format((int)(thickness[i]))
                k = k + 1;
        #self.title="{}nm [{}]".format(np.sum(d_au)+np.sum(d_al),title)
        thick=(int)(np.sum(thickness))
        self.title = "{}nm [{}]".format(thick, title)

        self.maters.append('air')
        d[N+1]=np.nan
        n_data[:,N+1] = 1       #air
        #print("lenda={}\nd={}\nn_data={}\nxitas={}".format(lendas,d,n_data,xitas))

        if True:
            self.R_2D(args,polarisation)
        else:
            self.R=np.zeros((nXita-1,nLenda))
            M0 = np.zeros((2, 2, d.shape[0]), dtype=complex)
            M_t0 = np.identity(2, dtype=complex)
            print("")
            for i in range(nLenda):                   #wavelength's number
                for j in range(nXita-1):              #angle
                    M=M0;      M_t=M_t0
                    row = (int)(nXita-1-j);  col=(int)(i)   #数据格式原因
                    r = jreftran_R(lendas[i],d,n_data[i,:],args.xitas[j],polarisation,M,M_t)
                    if np.isnan(r):
                        print("nan@[{},{}]".format(row,col))
                    self.R[row,col] = r
            del M,M_t;        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser('Metlab Hyperbolic')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='model/bdcn_pretrained_on_bsds500.pth',#'params/bdcn_final.pth',
        help='the model to test')
    parser.add_argument('--dump-dir', type=str, default='E:/MetaLab/hyperbolic/',
        help='the dir to store result')
    parser.add_argument('-layers', type=int, default=5,help='the number of layers')
    return parser.parse_args()

def ArgsOnSpectrum(args):
    assert(args is not  None)

    args.dpi = 100
    args.delta_angle = 0.1
    args.xitas_base = np.arange(0, 90 + args.delta_angle, args.delta_angle)
    if False:  # 6.21   尝试
        args.delta_angle = 0.001
        args.xitas_base = np.arange(40, 55. + args.delta_angle, args.delta_angle)
    nm = 1.0e-9
    args.lendas = np.arange(300, 2010, 10) * nm
    args.mater_file_dir = 'E:/MetaLab/hyperbolic'
    args.N_dict = N_maters_dict(args)
    args.polarisation = 1  # TM

    args.dump_dir='E:/MetaLab/dump/'
    args.compare_dir = 'E:/MetaLab/dump/compare/'
    args.materials = ['au', 'ag', 'al', 'cu']
    return args

if __name__ == '__main__':
    args = parse_args()
    args = ArgsOnSpectrum(args)
    args.random_seed = datetime.now()

    N_case,case = 5000,0
    mType = 'random'        #'al'
    #ud_au = 5  # 金属厚度[nm]
    N_dict = N_maters_dict(args)
    N_di = args.layers # 模型介质层数
    N_al, N_au = N_di, N_di
    d_al_TM = None  # randi([50, 1680], 1, N_al). / 10; % 168 =（300 - 25） / 1.44
    d_au = None


    # N_r_path,N_i_path = './hyperbolic/cu_n1.txt','./hyperbolic/cu_k1.txt'
    #d_au = ud_au * np.ones(N_au)
    # d_al_TM = np.array([52.2,16.1,18.8,16.1,71.8])
    d_al_TM = []

    random.seed(args.random_seed)
    print("================= SurfacePlasmon v0.1 ================= \n\targs={}".format(args.__dict__))
    #random.seed(42)
    while case <N_case:
        d_al_TM = None  #randi([50, 1680], 1, N_al). / 10; % 168 =（300 - 25） / 1.44
        d_au = None
        t0=time.time()
        #N_r_path,N_i_path = './hyperbolic/cu_n1.txt','./hyperbolic/cu_k1.txt'
        #d_au = ud_au * np.ones(N_au)
        #d_al_TM = np.array([52.2,16.1,18.8,16.1,71.8])
        d_au,d_al_TM=[],[]
        sum = 0
        metals = []
        for no in range(N_au):
            h=random.uniform(5, 10)
            d_au.append((int)(h))
            sum = sum+h
            #materials = ['au', 'ag', 'al', 'cu']
            metals.append(random.choice(args.materials))
            #metals.applend['au']   固定为某种金属
        for no in range(N_di):
            #hSi = random.uniform(5, 168)
            hSi = random.uniform(5, 300-sum)
            d_al_TM.append((int)(hSi))
            sum = sum+hSi
            if sum>=300-5:
                break

        title=""    #""{}_{}".format(mType,d_al_TM)
        #if sum * 1.46 < (300 - 25):
        if sum * 1.46 < (300):
            thickness = [None] * (N_au + N_di)
            thickness[::2] = d_au;            thickness[1::2] = d_al_TM
            device = SP_device(thickness, metals, args.polarisation, title, args)
            #SPP_compare(device,device,"off","", args)
            device.HeatMap()
            del device;     gc.collect()
            print("{}:\t{:.4g} sum={} d={},{}".format(case,time.time()-t0,sum,d_au,d_al_TM))
            case = case+1
            #break

