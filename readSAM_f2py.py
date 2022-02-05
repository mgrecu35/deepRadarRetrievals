from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
import combAlg as sdsu
#from bisectm import *


sdsu.mainfortpy()
sdsu.initp2()


import glob
import numpy as np

fs=sorted(glob.glob("restart_3/cm1out*nc"))
nt=0
zKuL=[]
R=287
h=(250/2.+np.arange(76)*250)/1e3
h1=(0.+np.arange(77)*250)/1e3

graupCoeff=np.polyfit(np.log10(sdsu.tablep2.gwc[:272]),\
                      sdsu.tablep2.zkag[:272],1)



umu=np.cos(0.0/180*np.pi)
fisot=2.7
tbL=[]
tb_L=[]
sfcRainL=[]
xL=[]
jacobL=[]
import numpy as np

pRateL=[]
cfadZ=np.zeros((76,60),float)

from numba import jit
@jit(nopython=True)
def build_cfad(zKu,cfadZ):
    nt,nz=zKu.shape
    for i in range(nt):
        for k in range(nz):
            if zKu[i,k]>0:
                i0=int(zKu[i,k])
                if i0>=0 and i0<60:
                    cfadZ[k,i0]+=1
zKuL=[]
attKuL=[]
zKu_cL=[]
rwcL=[]
swcL=[]
xL=[]
attKuL=[]
from hb_2 import *
zku_1=[]
zku_2=[]
alpha_1=alpha[::-1]
beta=0.7713
maxHL=[]
zKu_msL=[]
f1L=[]
f2L=[]
from simLoop import simLoop
ijL=[]
zKu_nubf_msL=[]
zKa_nubf_msL=[]
iL=[-1, -1, -1, 0, 0, 0, 1, 1, 1]
jL=[-1, 0, 1, -1, 0, 1, -1, 0, 1]

piaL=[]
piaKaL=[]
dnL=[]
zKaL,zKa_msL,zKa_cL,attKaL=[],[],[],[]
for f in fs[:]:
    for it in range(4):
        ijL=[]
        zku_3d=np.zeros((76,200,200),float)-99
        zku_3d_ms=np.zeros((76,200,200),float)-99
        zku_3d_true=np.zeros((76,200,200),float)-99
        zku_3d,zku_3d_ms,pia2d,\
            zka_3d,zka_3d_ms,pia2d_ka=\
                simLoop(f,zKuL,zKu_msL,xL,beta,maxHL,zKu_cL,sdsu,zku_3d,\
                        zku_3d_ms,zku_3d_true,pRateL,attKuL,rwcL,swcL,\
                        f1L,f2L,ijL,
                        dnL,zKaL,zKa_msL,zKa_cL,attKaL)
        
        for i_12 in ijL:
            i1=i_12[0]
            i2=i_12[1]
            z1=np.zeros((76),float)
            z1_ka=np.zeros((76),float)
            pia_nubf=0
            pia_ka_nubf=0
            for di,dj in zip(iL,jL):
                z1+=10.0**(0.1*zku_3d_ms[:,i1+di,i2+dj])/9.0
                pia_nubf+=10**(-0.1*pia2d[i1+di,i2+dj])/9.0
                z1_ka+=10.0**(0.1*zka_3d_ms[:,i1+di,i2+dj])/9.0
                pia_ka_nubf+=10**(-0.1*pia2d_ka[i1+di,i2+dj])/9.0
            pia_nubf=-10*np.log10(pia_nubf)
            pia_ka_nubf=-10*np.log10(pia_ka_nubf)
            piaL.append([pia2d[i1,i2],pia_nubf])
            zKu_nubf_msL.append(np.log10(z1)*10)
            piaKaL.append([pia2d_ka[i1,i2],pia_ka_nubf])
            zKa_nubf_msL.append(np.log10(z1_ka)*10)
                

plt.figure()
plt.plot(np.array(zKuL).mean(axis=0),h)
plt.plot(np.array(zKu_msL).mean(axis=0),h)
plt.plot(np.array(zKu_nubf_msL).mean(axis=0),h)
plt.plot(np.array(zKaL).mean(axis=0),h)
plt.plot(np.array(zKa_msL).mean(axis=0),h)
plt.plot(np.array(zKa_nubf_msL).mean(axis=0),h)
plt.ylim(0,15)
plt.grid()
plt.savefig('randomZ.png')
#stop
plt.figure()
zka_3dm=np.ma.array(zka_3d_ms,mask=zka_3d_ms<10)
plt.pcolormesh(zka_3dm[:,:,150],cmap='jet',vmax=35)
plt.colorbar()
zKu_msL=np.array(zKu_msL)
build_cfad(zKu_msL,cfadZ)
plt.figure()
plt.pcolormesh(np.arange(60),np.arange(54)*0.25,cfadZ[:54,:],cmap='jet',\
               norm=matplotlib.colors.LogNorm())
plt.xlim(10,60)
plt.ylim(0,12)
plt.colorbar()
plt.savefig('cfad_qv_1.0.png')


#plt.figure()
#plt.plot((np.array(attKuL)/10**(0.1*np.array(zKuL)*0.7712)).mean(axis=0),h)
#import matplotlib.pyplot as plt

#x, y = np.random.multivariate_normal(mean, cov, 5000).T


#zKa_ms=xr.DataArray(zKa_msL)
#pRate=xr.DataArray(pRateL)
#attKa=xr.DataArray(attKaL)
#tb35=xr.DataArray(tbL)
#piaKa=xr.DataArray(piaKaL)
#xL=xr.DataArray(xL,dims=['dim_0','dim_11'])
#jacob=xr.DataArray(jacobL,dims=['dim_0','dim_1','dim_11'])
#d=xr.Dataset({"zKa_obs":zKa_obs,"zKa_true":zKa_true,"zKa_ms":zKa_ms,\
#              "pRate":pRate,"attKa":attKa,"tb35":tb35, "piaKa":piaKa,"xL":xL,\
#              "jacob":jacob})
#d.to_netcdf("simulatedObs_SAM.nc")

from minisom import MiniSom
n1=10
n2=1
nz=54
som = MiniSom(n1,n2,nz,sigma=2.5,learning_rate=0.5, random_seed=0)
#np.random.seed(seed=10)
som.random_weights_init(zKu_msL[:,:54])
som.train_random(zKu_msL[:,:54],500) # training with 100 iterations

zKuClass=np.zeros((n1,nz),float)
pRateClass=np.zeros((n1),float)
countSClass=np.zeros((n1),float)
winL=np.zeros((zKu_msL.shape[0]),int)
for it,z1 in enumerate(zKu_msL):
    win=som.winner(z1[:54])
    winL[it]=win[0]

pRateL=np.array(pRateL)
plt.figure()
for i in range(n1):
    a=np.nonzero(winL==i)
    zKuClass[i]=zKu_msL[a[0],:54].mean(axis=0)
    pRateClass[i]=pRateL[a[0]].mean()
    countSClass[i]=len(a[0])
    plt.plot(zKuClass[i,:],h[:54]-h[17])

plt.savefig("cm1_zKu_class.png")
plt.figure()
plt.hist2d(np.array(maxHL)[:,0],np.array(maxHL)[:,1],bins=(35+np.arange(20),range(12)),norm=matplotlib.colors.LogNorm(),cmap='jet')

piaMap=np.zeros((20,12),float)
cpiaMap=np.zeros((20,12),float)

for maxH in maxHL:
    i0=int(maxH[0]-35)
    j0=int(maxH[1])
    if i0>=0 and i0<20 and j0>=0 and j0<12:
        piaMap[i0,j0]+=maxH[2]
        cpiaMap[i0,j0]+=1

piaMap/=cpiaMap
plt.figure()
c=plt.pcolormesh(35+np.arange(20),range(12),piaMap.T,cmap='jet')
plt.contour(35+np.arange(20),range(12),cpiaMap.T,levels=[10])
plt.colorbar(c)

#plt.figure()
#plt.pcolormesh(35+np.arange(20),range(12),cpiaMap.T,cmap='jet')

import xarray as xr
zKu_L=xr.DataArray(zKuL,dims=['np','nz'])
zKu_cL=xr.DataArray(zKu_cL,dims=['np','nz'])
zKu_msL=xr.DataArray(zKu_msL,dims=['np','nz'])
zKa_L=xr.DataArray(zKaL,dims=['np','nz'])
zKa_cL=xr.DataArray(zKa_cL,dims=['np','nz'])
zKa_msL=xr.DataArray(zKa_msL,dims=['np','nz'])
rwcL=xr.DataArray(rwcL,dims=['np','nz'])
swcL=xr.DataArray(swcL,dims=['np','nz'])
pRateL=xr.DataArray(pRateL,dims=['np','nz'])
attKuL=xr.DataArray(attKuL,dims=['np','nz'])
attKaL=xr.DataArray(attKaL,dims=['np','nz'])
f1L=xr.DataArray(f1L,dims=['np','nz'])
f2L=xr.DataArray(f2L,dims=['np','nz'])
piaL=xr.DataArray(piaL,dims=['np','n2'])
piaKaL=xr.DataArray(piaKaL,dims=['np','n2'])
dnL=xr.DataArray(dnL,dims=['np','nz'])
zKu_nubf_ms=xr.DataArray(zKu_nubf_msL,dims=['np','nz'])
zKa_nubf_ms=xr.DataArray(zKa_nubf_msL,dims=['np','nz'])
ds=xr.Dataset({"zKu":zKu_L,"zKu_ms":zKu_msL,"zKu_true":zKu_cL,\
               "zKa":zKu_L,"zKa_ms":zKu_msL,"zKa_true":zKa_cL,\
               "rwc":rwcL,"swc":swcL,"attKu":attKuL,\
               "attKa":attKaL,\
               "f1":f1L,"f2":f2L, "zKu_nubf_ms":zKu_nubf_ms,"piaKu":piaL,\
               "zKa_nubf_ms":zKa_nubf_ms,"piaKa":piaL,\
               "pRate":pRateL, "dn":dnL})
comp = dict(zlib=True, complevel=5)

encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf("sim_obs_CM1_dn_sloped.nc", encoding=encoding)

plt.figure()
plt.plot(np.array(zKuL).mean(axis=0),h)
plt.ylim(0,15)
plt.grid()
plt.savefig('randomZ.png')
