import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import combAlg as sdsu

sdsu.mainfortpy()
sdsu.initp2()
alpha_ice=0.0001273
beta=0.771
alpha_rain=0.00041273

def hb(zKum,alpha_ice,ibzd,alpha_rain,beta,dr):
    q=0.2*np.log(10)
    zeta=np.zeros((zKum.shape),float)
    zeta[:ibzd]=alpha_ice*10**(0.1*zKum[:ibzd]*beta)*q*beta*dr
    zeta[ibzd:]=alpha_rain*10**(0.1*zKum[ibzd:]*beta)*q*beta*dr
    srt_piaKu=max(5,55-zKum[:ibzd+4].max())
    zetamax=1.-10**(-srt_piaKu/10.*beta)
    if zeta.cumsum()[-1]>zetamax:
        eps=0.9999*zetamax/zeta.cumsum()[-1]
        zeta=eps*zeta
    else:
        eps=1.0
    corrc=eps*zeta.cumsum()
    zc=zKum-10/beta*np.log10(1-corrc)
    return zc,eps,-10/beta*np.log10(1-corrc[-1])

def hb_backwards(zKum,alpha_ice,ibzd,alpha_rain,beta,dr,srt_pia):
    q=0.2*np.log(10)
    zeta=np.zeros((zKum.shape),float)
    zeta[:ibzd]=alpha_ice*10**(0.1*zKum[:ibzd]*beta)*q*beta*dr
    zeta[ibzd:]=alpha_rain*10**(0.1*zKum[ibzd:]*beta)*q*beta*dr
    srt_piaKu=max(5,55-zKum[:ibzd+4].max())
    zetamax=1.-10**(-srt_piaKu/10.*beta)
    As=10**(-0.1*beta*srt_pia)
    zeta_sum=zeta[::-1].cumsum()[::-1]
    A=zeta_sum+As
    A[A>1]=1
    zc=zKum-10/beta*np.log10(A)
    return zc

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
    zKucL=[]
    sfcPrecipL=[]
    layerT=[]
    slayerT=[]
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
        zKucL.append(zKuC[i,bzd[i]+16-80:bzd[i]+30:2])
        dprPIAL.append(zKuC[i,bcf[i]-1]-zKu1[bcf[i]-1])
        srtPIAL.append(srtPIA[i])
        sfcPrecipL.append(sfcPrecip[i])
        layerT.append(bsfc[i]-bzd[i])
        slayerT.append(bsfc[i]-bzd[i]-30)
        #print(bsfc[i]-bzd[i]-30)
        for j in a[0]:
            if j+71-8-bzd[i]>=0 and j+71-8-bzd[i]<80:
                k0=int(zKu1[j]-10)
                if k0<40:
                    cfadZ[j+71-8-bzd[i],k0]+=1
    return ic,zKuL,sfcPrecipL,zKucL,srtPIAL,layerT,dprPIAL,slayerT
ic=0
zKuL=[]
zKucL=[]
sfcPrecipL=[]
srtPIAL=[]
layerTL=[]
slayerTL=[]
dprPIAL=[]
slayerTL=[]
for f in sorted(fs)[:60]:
    dic,zKu1,sfcPrecip1,zKuc1,srtPIA1,lt1,dpr_pia,sl1=readGPM(f,cfadZ)
    slayerTL.extend(sl1)
    srtPIAL.extend(srtPIA1)
    layerTL.extend(lt1)
    ic+=dic
    zKuL.extend(zKu1)
    zKucL.extend(zKuc1)
    sfcPrecipL.extend(sfcPrecip1)
    dprPIAL.extend(dpr_pia)
zKuc1L=[]
piaHBL=[]
sfcPrecipL2=[]
pRate2L=[]
grateCoeff=np.array([ 0.06438296, -1.72211385])
grateCoeff=np.array([ 0.0738764 , -1.56192592])
grateCoeff=np.array([ 0.0738764 , -1.69254395])
grateCoeff=np.array([ 0.0738764 , -1.75785293]) #dn_snow=-0.75
graupRateL=[]
for i,zku1 in enumerate(zKuL):
    ibzd=32
    dr=0.25
    #print()
    srt_pia=srtPIAL[i]*(1-slayerTL[i]/layerTL[1])
    zkuc1= hb_backwards(zku1,alpha_ice,ibzd,alpha_rain,beta,dr,srt_pia)
    #zkuc1,eps,piaHB=hb(zku1,alpha_ice,ibzd,alpha_rain,beta,dr)
    zKuc1L.append(zkuc1)
    sfcPrecipL2.append((10**(0.1*zkuc1[-1])/300.)**(1/1.3))
    pRate2L.append((10**(0.1*zkuc1[:])/300.)**(1/1.3))
    graupRateL.append(10**(grateCoeff[0]*zkuc1+grateCoeff[1]))
    #piaHBL.append(piaHB)

plt.plot(np.array(pRate2L).mean(axis=0))
fint=np.interp(range(47),[0,20,36,39,47],[1,1,1,0,0])
mixPrecip=fint*np.array(graupRateL).mean(axis=0)+(1-fint)*np.array(pRate2L).mean(axis=0)
plt.plot(mixPrecip)
plt.plot(np.array(graupRateL).mean(axis=0),'*')
stop   
plt.plot(np.array(zKuL).mean(axis=0),range(47)[::-1])
plt.plot(np.array(zKucL).mean(axis=0),range(47)[::-1])
plt.plot(np.array(zKuc1L).mean(axis=0),ranrgge(47)[::-1])
zKuL=np.array(zKuL)
zKucL=np.array(zKucL)
#plt.figure()
#plt.scatter(zKuL[:,-16],zKucL[:,-16]-zKuL[:,-16])
stop
fname="sim_obs_CM1.nc"
zKu,zKu_true,zKa,zKa_true,pRate,piaKu,piaKa=[],[],[],[],[],[],[]
def readDataset(fname):
    fh=Dataset(fname)
    pRate=fh["pRate"][:]
    a=np.nonzero(pRate[:,0]<20)
    zKu=fh["zKu_nubf_ms"][:][a]
    zKu_true=fh["zKu_true"][:][a]
    zKa=fh["zKa_nubf_ms"][:][a]
    zKa_true=fh["zKa_true"][:][a]
    pRate=fh["pRate"][:][a]
    piaKu=fh["piaKu"][:][a]
    piaKa=fh["piaKa"][:][a]
  
    return zKu,zKu_true,zKa,zKa_true,pRate,piaKu,piaKa

import glob
fL=glob.glob("sim_obs_CM1_dn_sl_*.nc")
for f in sorted(fL)[-1:]:
    zKu1,zKu_true1,zKa1,zKa_true1,pRate1,piaKu1,piaKa1=readDataset(f)
    zKu.extend(zKu1)
    zKu_true.extend(zKu_true1)
    zKa.extend(zKa1)
    zKa_true.extend(zKa_true1)
    pRate.extend(pRate1)
    piaKu.extend(piaKu1)
    piaKa.extend(piaKa1)


#plt.plot(np.array(zKu).mean(axis=0)[10:50],np.arange(76)[10:50]-10)

zKu=np.array(zKu)
zKu_true=np.array(zKu_true)
zKa=np.array(zKa)
zKa_true=np.array(zKa_true)
pRate=np.array(pRate)
piaKu=np.array(piaKu)
piaKa=np.array(piaKa)

from sklearn.preprocessing import StandardScaler
scalerZKu = StandardScaler()
scalerZKa = StandardScaler()
scalerPrec=StandardScaler()



scalerZKu.fit(zKu_true[:,:])
scalerZKa.fit(zKa_true[:,:])
scalerPrec.fit(pRate[:,:])
zKu_sc=scalerZKu.transform(zKu)[:,10:50][:,::-1]
zKu_true_sc=scalerZKu.transform(zKu_true)[:,10:50][:,::-1]
zKa_sc=scalerZKa.transform(zKa)[:,10:50][:,::-1]
zKa_true_sc=scalerZKa.transform(zKa_true)[:,10:50][:,::-1]
pRate_sc=scalerPrec.transform(pRate)[:,10:50][:,::-1]

from sklearn.model_selection import train_test_split

ind=range(zKu.shape[0])


from sklearn.cluster import KMeans
import matplotlib
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=70, \
                        algorithm='ball_tree').fit(zKu[:,10:50][:,::-1])


rms,ind=nbrs.kneighbors(np.array(zKuL))
zdiffL=[]
sfcPrecipL2=[]
for i,zku in enumerate(zKuL):
    zdiffL.append(zku-zKu[ind[i][0],10:50][::-1])
    sfcPrecipL2.append(pRate[ind[i],0].mean(axis=0))
    
plt.plot(np.array(zdiffL).mean(axis=0),range(40))
plt.plot(np.array(zdiffL).std(axis=0),range(40))
plt.ylim(40,0)
