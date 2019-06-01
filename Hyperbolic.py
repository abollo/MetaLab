from some_libs import *
from jreftran_rt import *

def HeatMap(R):
    #plt.imshow(R);      plt.show()
    #fig = plt.figure()      #figsize=(6, 3.2)
    #ax = fig.add_subplot(111)
    fig, ax = plt.subplots(figsize=(2, 1))

    ax.set_title('Hyperbolic - R\n')
    cmap = 'coolwarm'  # "plasma"  #https://matplotlib.org/examples/color/colormaps_reference.html
    #cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    if True:
        ticks = np.linspace(0, 1, 10)
        ylabels = [int(i) for i in np.linspace(0, 90, 10)]
        xlabels = [int(i) for i in np.linspace(300, 2000, 10)]
        #cbar_kws={'label': 'Reflex', 'orientation': 'horizontal'}
        sns.set(font_scale=0.8)
        sns.heatmap(R, square=True, ax=ax, cmap=cmap, yticklabels=ylabels, center=0.6, cbar_kws={'label': 'Reflex', 'orientation': 'horizontal'})
        plt.ylabel('Xita')
        y_limit = ax.get_ylim();
        x_limit = ax.get_xlim()
        ax.set_yticks(ticks * y_limit[0])
        ax.set_xticks(ticks * x_limit[1])
        # ax.set_xticks(np.arange(1, 256, 1))
    else:
        plt.imshow(R)
        ax.set_aspect(256 / 90.0)
        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
    plt.savefig('F:\Project\MetaLab\hyperbolic\sample\{}.jpg'.format('metlab_0'))
    plt.show()

def Hyperbolic_metamaterial_au(N_al,N_au,d_al,d_au,polarisation,pathway1,pathway2,str1,str2):
    nm = 1.0e-9
    epsilon0 = 1.0 / (36 * np.pi) * nm                      # dielectric constant of the free space
    I = np.arange(300,2010,10)*nm               #300:10:2000[nm]
    fb1 = np.loadtxt(str1)
    fb2 = np.loadtxt(str2)
    n_au = fb1[:,1]+1j*fb2[:,1]
    L =len(I)
    t_0 = np.arange(0,90,1)
    M = len(t_0)
    N = N_al + N_au
    k = 0
    d=np.zeros(N+2)
    d[0] = np.nan
    n_data=np.zeros((L,N+2),dtype=np.complex64)
    for i in range(N): #layer
        if i%2 == 0:
            d[i + 1] = d_au[k] * nm         #% [m]
            n_data[:, i + 1] = n_au[:]
        else:
            d[i + 1] = d_al[k] * nm         #% [m]
            n_data[:, i + 1] = 1.46         #SiO2
            k = k + 1;

    n_data[:, 0] = 1.77     #Al2O3
    d[N+1]=np.nan
    n_data[:,N+1] = 1       #air

    r = np.zeros((M,L),dtype=np.complex64);
    t = np.zeros((M,L),dtype=np.complex64);
    R=np.zeros((M,L))
    T = np.zeros((M,L));
    A=np.zeros((M,L))
    for i in range(L):                   #wavelength's number
        for j in range(M):              #angle
            #the refractive index of the each layer
            #Complex refractive index for eatch layer
            row = (int)(M-1-j);  col=(int)(i)   #数据格式原因
            r[row,col], t[row,col], R[row,col], T[row,col], A[row,col] = jreftran_rt(I[i],d,n_data[i,:],t_0[j],polarisation)
            #[r(i,j),t(i,j),R(i,j),T(i,j),A(i,j),Y_tot(i,j)]=
    print("")
    HeatMap(R)


if __name__ == '__main__':
    ud_au = 5       #金属厚度[nm]

    N_di = 5    # 模型介质层数
    N_al,N_au = N_di,N_di
    d_al_TM = None  #randi([50, 1680], 1, N_al). / 10; % 168 =（300 - 25） / 1.44
    d_au = None
    polarisation=1 #TM
    pathway1 = None #['H:\for_DrCys\hyperbolic_metamaterial\随机厚度\TM\Cu1\',num2str(numfig_TM),'.png'];
    pathway2 = None #['H:\for_DrCys\hyperbolic_metamaterial\随机厚度\TM\Cu1\',num2str(numfig_TM),'.txt'];
    str1,str2 = './hyperbolic/cu_n1.txt','./hyperbolic/cu_k1.txt'
    d_au = ud_au * np.ones(N_au)
    d_al_TM = np.array([52.2,16.1,18.8,16.1,71.8])
    Hyperbolic_metamaterial_au(N_al, N_au, d_al_TM, d_au, polarisation, pathway1, pathway2, str1, str2)