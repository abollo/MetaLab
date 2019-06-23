from SurfacePlasmon import *

def Case_6(N_al,N_au,d_al,d_au,polarisation,mType,title,N_dict,args):
    nm = 1.0e-9
    epsilon0 = 1.0 / (36 * np.pi) * nm                      # dielectric constant of the free space
    lendas = np.arange(300,2010,10)*nm               #300:10:2000[nm]
    if mType != 'random':
        #fb1 = np.loadtxt(N_r_path)        fb2 = np.loadtxt(N_i_path)        n_au = fb1[:,1]+1j*fb2[:,1]
        n_au = N_dict[mType]
    else:
        title=""
    nLenda =len(lendas)
    #t_0 = np.arange(0,90,0.2)

    #xitas = np.arange(0, 90, 0.1)
    #xitas = np.arange(30, 60, 0.001)
    M = len(args.xitas)
    N = 2
    k = 0
    d=np.zeros(N+2)
    d[0] = np.nan
    n_data=np.zeros((nLenda,N+2),dtype=np.complex64)
    d=np.array( [np.inf, 5* nm, 30* nm, np.inf] )
    for j in range(nLenda):
        n_data[j, 1] = 3.719+4.362j
        n_data[j, 2] = 0.130+3.162j
    title="{}nm [{}]".format(np.sum(d),"spp")

    n_data[:, 0] = 1.517     #棱镜
    n_data[:,N+1] = 1       #air
    #print("lenda={}\nd={}\nn_data={}\nxitas={}".format(lendas,d,n_data,xitas))
    r = np.zeros((M,nLenda),dtype=np.complex64);
    t = np.zeros((M,nLenda),dtype=np.complex64);
    R=np.zeros((M,nLenda))
    T = np.zeros((M,nLenda));
    A= np.zeros((M,nLenda))
    if False:
        for i in range(nLenda):                      #wavelength's number
            n_ = n_data[i,:]
            r, t, R, T, A = jreftran_rt_vector(lendas[i],d,n_data[i,:],xitas,polarisation)
    else:
        for i in range(nLenda):                   #wavelength's number
            for j in range(M):              #angle
                #the refractive index of the each layer
                #Complex refractive index for eatch layer
                row = (int)(M-1-j);  col=(int)(i)   #数据格式原因
                r[row,col], t[row,col], R[row,col], T[row,col], A[row,col] = jreftran_rt(lendas[i],d,n_data[i,:],args.xitas[j],polarisation)
    #print("")
    HeatMap(R,mType,title,args,lendas)

def parse_args():
    parser = argparse.ArgumentParser('Metlab Hyperbolic')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='model/bdcn_pretrained_on_bsds500.pth',#'params/bdcn_final.pth',
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='result',
        help='the dir to store result')
    parser.add_argument('-layers', type=int, default=5,help='the number of layers')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.xitas= np.arange(30, 60, 0.1)
    args.dpi=100
    if False:   #6.21   尝试
        args.delta_angle=0.001
        args.xitas = np.arange(40, 55.+args.delta_angle, args.delta_angle)

    N_case,case = 1000,0
    mType = 'random'        #'al'
    #ud_au = 5  # 金属厚度[nm]
    N_dict = N_maters_dict()
    N_di = args.layers # 模型介质层数
    N_al, N_au = N_di, N_di
    d_al_TM = None  # randi([50, 1680], 1, N_al). / 10; % 168 =（300 - 25） / 1.44
    d_au = None
    polarisation = 1  # TM

    # N_r_path,N_i_path = './hyperbolic/cu_n1.txt','./hyperbolic/cu_k1.txt'
    #d_au = ud_au * np.ones(N_au)
    # d_al_TM = np.array([52.2,16.1,18.8,16.1,71.8])
    d_al_TM = []
    random.seed(42)
    while case <N_case:
        d_al_TM = None  #randi([50, 1680], 1, N_al). / 10; % 168 =（300 - 25） / 1.44
        d_au = None
        t0=time.time()
        #N_r_path,N_i_path = './hyperbolic/cu_n1.txt','./hyperbolic/cu_k1.txt'
        #d_au = ud_au * np.ones(N_au)
        #d_al_TM = np.array([52.2,16.1,18.8,16.1,71.8])
        d_au,d_al_TM=[],[]
        sum = 0
        for no in range(N_au):
            h=random.uniform(5, 10)
            d_au.append((int)(h))
            sum = sum+h
        for no in range(N_di):
            #hSi = random.uniform(5, 168)
            hSi = random.uniform(5, 300-sum)
            d_al_TM.append((int)(hSi))
            sum = sum+hSi
            if sum>=300-5:
                break
        title="{}_{}".format(mType,d_al_TM)
        #if sum * 1.46 < (300 - 25):
        if sum * 1.46 < (300):
            Case_6(N_al, N_au, d_al_TM, d_au, polarisation, mType, title, N_dict,args)
            print("{}:\t{:.4g} sum={} d={},{}".format(case,time.time()-t0,sum,d_au,d_al_TM))
            case = case+1
            break