import matplotlib.colors as col
import matplotlib
#import tensorflow as tf
#from tensorflow.keras.layers import *
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.optimizers import Adam, RMSprop
import combAlg as sdsu
import numpy as np
from netCDF4 import Dataset
sdsu.mainfortpy()
sdsu.initp2()
alpha_ice=0.0001273
beta=0.771
alpha_rain=0.00041273
import glob
fs=glob.glob("convProfs/kuka_zprofs_Conv_*nc")
cfadZ=np.zeros((80,40),float)
def readGPM(f,cfadZ):
    fh=Dataset(f)
    zKu=fh["zKu"][:]
    zKa=fh["zKa"][:]
    bzd=fh["bzd"][:]
    zKuC=fh["zKuCorrected"][:]

    sfcPrecip=fh["sfcPrecip"][:]
    bcf=fh["bcf"][:]
    bsfc=fh["binSfc"][:]
    eps=fh["eps"][:]
    relFlag=fh["relFact"][:]
    try:
        srtPIA=fh["srtPIA"][:]
    except:
        print(fh)
        stop
    srtPIAL=[]
    dprPIAL=[]
    ic=0
    zKuL=[]
    zKaL=[]
    zKucL=[]
    sfcPrecipL=[]
    layerT=[]
    slayerT=[]
    epsL=[]
    #print(fh)
    #stop
    for i,zKu1 in enumerate(zKu):
        if zKa[i,:bzd[i]].max()<30 or bzd[i]>160 or bcf[i]<168 or \
           bzd[i]+30>bcf[i] or relFlag[i]>=2 or relFlag[i]<1:
            continue
        a=np.nonzero(zKu1[:bzd[i]+16]>10)
        ic+=1
        zKu1[zKu1<0]=0
        zKuC[i,zKuC[i,:]<0]=0
        zKuL.append(zKu1[bzd[i]+16-80:bzd[i]+30:2])
        zKaL.append(zKa[i,bzd[i]+16-80:bzd[i]+30:2])
        zKucL.append(zKuC[i,bzd[i]+16-80:bzd[i]+30:2])
        dprPIAL.append(zKuC[i,bcf[i]-1]-zKu1[bcf[i]-1])
        srtPIAL.append(srtPIA[i])
        sfcPrecipL.append(sfcPrecip[i])
        layerT.append(bsfc[i]-bzd[i])
        slayerT.append(bsfc[i]-bzd[i]-30),
        epsL.append(eps[i])
        #print(bsfc[i]-bzd[i]-30)
        for j in a[0]:
            if j+71-8-bzd[i]>=0 and j+71-8-bzd[i]<80:
                k0=int(zKu1[j]-10)
                if k0<40:
                    cfadZ[j+71-8-bzd[i],k0]+=1
    return ic,zKuL,sfcPrecipL,zKucL,srtPIAL,layerT,dprPIAL,slayerT,zKaL,epsL

ic=0
zKuL=[]
zKaL=[]
zKucL=[]
sfcPrecipL=[]
srtPIAL=[]
layerTL=[]
slayerTL=[]
dprPIAL=[]
slayerTL=[]
epsL=[]
for f in sorted(fs)[:180]:
    dic,zKu1,sfcPrecip1,zKuc1,srtPIA1,lt1,\
        dpr_pia,sl1,zKa1,eps1=readGPM(f,cfadZ)
    zKuL.extend(zKu1)
    zKucL.extend(zKuc1)
    zKaL.extend(zKa1)
    epsL.extend(eps1)
    srtPIAL.extend(srtPIA1)

zKuL=np.array(zKuL)
zKucL=np.array(zKucL)
zKaL=np.array(zKaL)
zKuL[zKuL<0]=0
zKucL[zKucL<0]=0
zKaL[zKaL<0]=0
#srtPIAL=
srtPIAL=np.array(srtPIAL)
from minisom import MiniSom
n1=49
n2=1
nz=zKu1[0].shape[0]
som = MiniSom(n1,n2,nz,sigma=2.5,learning_rate=0.5, random_seed=0)
som.random_weights_init(zKuL)

from minisom import MiniSom
nt=zKuL.shape[0]
winL=np.zeros((nt),int)
it=0
for z1 in zKuL:
    win=som.winner(z1)
    winL[it]=win[0]
    it+=1

slopeL=[]

n12=int(n1**0.5)
import matplotlib.pyplot as plt

srtPIAc=[]

def hb_2(zKum,alphaS,alphaR,beta,dr,srt_pia,iz):
    q=0.2*np.log(10)
    alpha1d=np.interp(range(nz),[0,iz,iz+1,nz],[alphaS,alphaS,alphaR,alphaR])
    dalpha=np.interp(range(nz),[0,iz,iz+1,nz],[1,1,1,0.4])
    zeta=q*beta*dalpha*alpha1d*10**(0.1*zKum*beta)*dr
    eps=0.9999*(1-10**(-0.1*beta*srt_pia))/zeta.cumsum()[-1]
    zeta=zeta
    corrc=eps*zeta.cumsum()
    zc=zKum-10/beta*np.log10(1-corrc)
    return zc,eps,corrc,zeta

dr=0.25
alphaS=0.000128
alphaR=0.00041
eta=0.7713
epsL_hb=[]
epsL_dpr=[]

for i in range(n1):
    i0=int(i/n12)
    j0=i-i0*n12
    #plt.subplot(n12,n12,i+1)
    plt.figure()
    plt.subplot(121)
    ind1=np.nonzero(winL==i)
    #plt.plot(np.array(zSimL[i]).mean(axis=0),range(47))
    zm=np.array(zKuL[ind1[0],:]).mean(axis=0)
    zmc=np.array(zKucL[ind1[0],:]).mean(axis=0)
    zm_ka=np.array(zKaL[ind1[0],:]).mean(axis=0)
    ind=np.argmax(zm)
    zcL=[]
    
    piaHBL=[]
    for i1 in ind1[0]:
        iz=min(30,ind)
        srtPIAL[i1]
        piaHB=zKucL[i1][-1]-zKuL[i1][-1]
        if piaHB<0.1:
            piaHB=0.8*srtPIAL[i1]
        zc1,eps1,c1,zeta1=hb_2(zKuL[i1],alphaS,alphaR,beta,dr,\
                               piaHB,iz)
        epsL_hb.append(eps1)
        epsL_dpr.append(epsL[i1])
        zcL.append(zc1)
        a1=np.nonzero(zc1!=zc1)
        #print(len(a1[0]))
        #stop
    #print(epsL)
    print(np.corrcoef(epsL_hb,epsL_dpr))
    plt.plot(zm,range(47))
    plt.plot(zmc,range(47))
    plt.plot(np.array(zcL).mean(axis=0),range(47))
    plt.plot(zm[:ind],range(ind),'*')
    plt.plot(zm_ka,range(47))
    plt.plot(0.5*(zm-zm_ka),range(47))
    plt.ylim(46,0)
    plt.subplot(122)
    plt.plot(np.gradient(0.5*(zm-zm_ka)),range(47))
    srtPIAc.append([srtPIAL[ind1].mean(axis=0),zmc[-1]-zm[-1]])
    plt.ylim(46,0)
    plt.grid()
             
