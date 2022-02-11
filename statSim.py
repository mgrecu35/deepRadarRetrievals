import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib
#import tensorflow as tf
#from tensorflow.keras.layers import *
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.optimizers import Adam, RMSprop
import combAlg as sdsu

sdsu.mainfortpy()
sdsu.initp2()
alpha_ice=0.0001273
beta=0.771
alpha_rain=0.00041273


from forwardModelZ import *
#forward_model_ku_st(graup_rate,rain_rate,dNw,dr,di,sdsu,iz):

fh=Dataset("convDataBase2.nc")
rain1D=fh["rain1D"][:]
snow1D=fh["snow1D"][:]
zKu_obs=fh["zKu"][:]
dr=0.25
iz=32
piaKu=fh["srtPIA"][:]
from backwardsProf import *
precipRateL=[]
ztrueL=[]
dprSfcP=fh["sfcPrecip"][:]
dprSfcPL=[]
for i in range(10000):
    for ip in range(5):
        srtPIA=piaKu[i]*0.8+np.random.randn()*2
        if(srtPIA>0):
            precipRate,piaKu_top,ztrue=backwards_kuret(dr,sdsu,iz,zKu_obs[i,:],piaKu[i]*0.8)
            precipRateL.append(precipRate)
            ztrueL.append(ztrue)
            dprSfcPL.append(dprSfcP[i])
        
plt.subplot(121)
plt.plot(np.array(precipRateL).mean(axis=0),range(47)[::-1])
plt.subplot(122)
plt.plot(np.array(ztrueL).mean(axis=0),range(47))
plt.plot(np.array(zKu_obs).mean(axis=0),range(47))
plt.ylim(46,0)
stop

dnw_coeff=np.array([-0.23830612,  0.61780454])
dr=0.25
zKuL=[]
rain1m=np.log(rain1D[:]+1e-3).mean(axis=0)
rainCov=np.cov(np.log(rain1D+1e-3).T)
rain1D_f=np.random.multivariate_normal(rain1m,rainCov,15000)
snow1m=np.log(snow1D[:]+1e-3).mean(axis=0)
snowCov=np.cov(np.log(snow1D.T)+1e-3)

snow1D_f=np.random.multivariate_normal(snow1m,snowCov,15000)
from scipy.ndimage import gaussian_filter

for i,rain_rate in enumerate(rain1D_f[:10,:15000]):
    di=4
    iz=31
    rain_rate=np.exp(rain_rate)
    dNw=np.polyval(dnw_coeff,np.log10(rain_rate+0*1e-3))
    dNw[rain_rate<1e-1]=0
    graup_rate=np.exp(snow1D_f[i,:])
    f=graup_rate[32]/rain_rate[33]
    f1=np.random.randn(47)
    f1=np.exp(gaussian_filter(f1,sigma=4))
    graup_rate[:33]*=((0.7+0.2*np.random.random())/f)
    graup_rate*=f1
    rain_rate*=f1
    zKu_m,piaKu_m=forward_model_ku_st(graup_rate,rain_rate,dNw,dr,di,sdsu,iz)
    zKuL.append(zKu_m)

import random
pop_ind=range(rain1D.shape[0])
for i in range(1500):
    i1,i2=random.sample(pop_ind,2)
    w1=np.random.random()
    rain_rate=w1*rain1D[i1,:]+(1-w1)*rain1D[i2,:]
    dNw=np.polyval(dnw_coeff,np.log10(rain_rate+0*1e-3))
    dNw[rain_rate<1e-1]=0
    graup_rate=w1*snow1D[i1,:]+(1-w1)*snow1D[i2,:]
    zKu_m,piaKu_m=forward_model_ku_st(graup_rate,rain_rate,dNw,dr,di,sdsu,iz)
    zKuL.append(zKu_m)
    
plt.plot(np.array(zKuL).mean(axis=0),np.arange(47)[::-1])
plt.plot(np.array(zKu_obs).mean(axis=0),np.arange(47)[::-1])


# = np.random.multivariate_normal(mean, cov, 5000)




from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
n1=9
nc=n1*n1
zKu=np.array(zKuL)
kmeans = KMeans(n_clusters=nc, random_state=10)
kmeans.fit(zKu)
obs_c=kmeans.predict(zKu_obs.data.astype(float))
ic=0
#precipC=[]
#piaC=[]
y1=[]
y2=[]
import random
for i in range(n1):
    for j in range(n1):
        ic+=1
        plt.subplot(n1,n1,ic)
        a=np.nonzero(kmeans.labels_==ic-1)
        plt.plot(zKu[a[0],:].mean(axis=0),range(47))
        zstd=zKu[a[0],:].std(axis=0)
        zm=zKu[a[0],:].mean(axis=0)
        plt.fill_betweenx(range(47),zm-zstd,zm+zstd,alpha=0.2)
        plt.ylim(46,0)
        a=np.nonzero(obs_c==ic-1)
        if len(a[0])>0:
            plt.plot(zKu_obs[a[0],:].mean(axis=0),range(47))
            plt.legend(["%2.2i"%len(a[0])])
        #n1,n2=random.sample(list(a[0]), 2)
        #precipC.append(sfcPrecip[a].mean())
        #piaC.append(srtPIA[a].mean())
        #y1.append(sfcPrecip_m[n1])
        #y2.append(sfcPrecip_m[n2])
