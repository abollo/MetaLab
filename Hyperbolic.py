from some_libs import *
from jreftran_rt import *

def Hyperbolic_metamaterial_au(N_al,N_au,d_al,d_au,polarisation,pathway1,pathway2,str1,str2):
    nm = 1.0e-9
    epsilon0 = 1.0 / (36 * np.pi) * nm                      # dielectric constant of the free space
    I = np.arange(300,2010,10)*nm               #300:10:2000[nm]
    fb1 = np.loadtxt(str1)
    fb2 = np.loadtxt(str2)
    n_au = fb1[:,1]+1j*fb2[:,1]
    L =len(I)
    t_0 = np.arange(0,89.9,0.1)
    M = len(t_0)
    N = N_al + N_au
    k = 0
    d=np.zeros(N+2)
    d[0] = np.nan
    n_data=np.zeros((L,N+2))
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

    r = np.zeros((L,M));        t = np.zeros((L,M));    R=np.zeros((L,M))
    T = np.zeros((L, M));       A=np.zeros((L,M))
    for i in range(L):                   #wavelength's number
        for j in range(M):              #angle
            #the refractive index of the each layer
            #Complex refractive index for eatch layer
            r[i,j], t[i,j], R[i,j], T[i,j], A[i,j] = jreftran_rt(I[i],d,n_data[i,:],t_0[j],polarisation)
            #[r(i,j),t(i,j),R(i,j),T(i,j),A(i,j),Y_tot(i,j)]=
    print("")
    plt.imshow(r);      plt.show()
    plt.imshow(t);      plt.show()
    plt.imshow(R);      plt.show()
    plt.imshow(T);      plt.show()
    plt.imshow(A);      plt.show()


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