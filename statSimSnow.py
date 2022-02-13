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
zKuSimL=[]
piaKuL=[]
for i in range(10000):
    precipRate,piaKu=forward_kuret(dr,sdsu,zKu_obs[i,:])
    piaKuL.append(piaKu)
    precipRateL.append(precipRate)
    dprSfcPL.append(dprSfcP[i])

plt.plot(np.array(precipRateL).mean(axis=0),range(47)[::-1])

stop
#plt.subplot(122)
#plt.plot(np.array(ztrueL).mean(axis=0),range(47))
#plt.plot(np.array(zKuSimL).mean(axis=0),range(47))
#plt.plot(np.array(zKu_obs).mean(axis=0),range(47))

plt.ylim(46,0)




from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
n1=5
nc=n1*n1
zKu=np.array(zKuSimL)
kmeans = KMeans(n_clusters=nc, random_state=10)
kmeans.fit(zKu)
obs_c=kmeans.predict(zKu_obs.data.astype(float))
ic=0
#precipC=[]
#piaC=[]
y1=[]
y2=[]
import random
ic=0
for i in range(n1):
    for j in range(n1):
        ic+=1
        plt.subplot(n1,n1,ic)
        a=np.nonzero(kmeans.labels_==ic-1)
        plt.plot(zKu[a[0],:].mean(axis=0),range(47))
        zstd=zKu[a[0],:].std(axis=0)
        zm=zKu[a[0],:].mean(axis=0)
        plt.fill_betweenx(range(47),zm-zstd,zm+zstd,alpha=0.2)
        precipRm=np.array(precipRateL)[a[0],:].mean(axis=0)
        plt.plot(precipRm,range(47))
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
