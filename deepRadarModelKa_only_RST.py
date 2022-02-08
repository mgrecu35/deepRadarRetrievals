import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop

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
for f in fL:
    zKu1,zKu_true1,zKa1,zKa_true1,pRate1,piaKu1,piaKa1=readDataset(f)
    zKu.extend(zKu1)
    zKu_true.extend(zKu_true1)
    zKa.extend(zKa1)
    zKa_true.extend(zKa_true1)
    pRate.extend(pRate1)
    piaKu.extend(piaKu1)
    piaKa.extend(piaKa1)


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
zKu_sc=scalerZKu.transform(zKu)[:,::-1]
zKu_true_sc=scalerZKu.transform(zKu_true)[:,::-1]
zKa_sc=scalerZKa.transform(zKa)[:,::-1]
zKa_true_sc=scalerZKa.transform(zKa_true)[:,::-1]
pRate_sc=scalerPrec.transform(pRate)[:,::-1]

from sklearn.model_selection import train_test_split

ind=range(zKu.shape[0])

nt,nz=zKu_sc.shape
ind_train, ind_test,\
    y_train, y_test = train_test_split(range(nt), pRate_sc,\
                                       test_size=0.5, random_state=42)


from sklearn.cluster import KMeans
import matplotlib


import scipy
import scipy.optimize
from scipy.optimize import minimize as minimize

from sklearn.neighbors import NearestNeighbors
import numpy as np
X=np.swapaxes(np.array([zKu_sc,zKa_sc]).T,0,1)
x_train=X[ind_train,:,:]
x_test=X[ind_test,:,:]
piaKa_train=piaKa[ind_train,:]
piaKa_test=piaKa[ind_test,:]

nbrs = NearestNeighbors(n_neighbors=70, \
                        algorithm='ball_tree').fit(x_train[:,-40:,1])


n1=x_test.shape[0]
r1=np.random.random(n1)
a1=np.nonzero(r1<0.1)
rms,ind=nbrs.kneighbors(x_test[a1[0],-40:,1])
sfcPrecip_test=[]
sfcPrecip_0=[]

for i, i1 in enumerate(a1[0]):
    w1=np.exp(-0.5*(piaKa_test[i1,1]-piaKa_train[ind[i],1])**2/9)
    y1=(y_train[ind[i],-1]*w1).sum()/w1.sum()
    y2=y_test[i1,-1]
    sfcPrecip_test.append(scalerPrec.scale_[0]*y2+scalerPrec.mean_[0])
    sfcPrecip_0.append(scalerPrec.scale_[0]*y1+scalerPrec.mean_[0])

sfcPrecip_test=np.array(sfcPrecip_test)
sfcPrecip_0=np.array(sfcPrecip_0)
interVs=[[1,5],[5,10],[10,15],[15,20]]
c1,c2=0,0
for r1,r2 in zip(sfcPrecip_test,sfcPrecip_0):
    if r1<1:
        continue
    c1+=1
    if abs(r1-r2)<r1:
        c2+=1

print(c2/c1)

for int1 in interVs:
    a=np.nonzero((sfcPrecip_test-int1[0])*(sfcPrecip_test-int1[1])<0)
    diff=np.abs((sfcPrecip_test[a[0]]-sfcPrecip_0[a[0]]))
    #plt.hist(diff/sfcPrecip_test[a[0]]*100)
    b=np.nonzero(diff>1)
    print(int1, len(b[0])/len(a[0]))

ax=plt.subplot(111)
plt.hist2d(sfcPrecip_test,sfcPrecip_0,bins=np.arange(25)*0.75,norm=matplotlib.colors.LogNorm(),cmap='jet')
ax.set_aspect('equal')

stop
def lstm_model(ndims=2):
    ntimes=None
    inp = tf.keras.layers.Input(shape=(ntimes,ndims,))
    out1 = tf.keras.layers.LSTM(12, return_sequences=True)(inp)
    out1 = tf.keras.layers.LSTM(12, return_sequences=True)(out1)
    out = tf.keras.layers.LSTM(1, recurrent_activation=None, \
                               return_sequences=True)(out1)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


itrain=1
#stop
model=lstm_model(1)

if itrain==1:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss='mse',\
        metrics=[tf.keras.metrics.MeanSquaredError()])
    history = model.fit(x_train[:,-40:,:1], y_train[:,-40:], \
                        batch_size=32,epochs=2,
                        validation_data=(x_test[:,-40:,:1], \
                                         y_test[:,-40:]))
else:
    model=tf.keras.models.load_model("radarProfilingKuFreq.h5")

model.save("radarProfilingKuFreq.h5")

yp=model(x_test[:,-40:,:1])

sfcPrecip_0=scalerPrec.scale_[0]*yp[:,-1,0].numpy()+scalerPrec.mean_[0]
sfcPrecip_test=pRate[:,0][ind_test]
interVs=[[1,10],[10,20],[20,30],[30,40],[40,50]]
for int1 in interVs:
    a=np.nonzero((sfcPrecip_test-int1[0])*(sfcPrecip_test-int1[1])<0)
    diff=np.abs((sfcPrecip_test[a[0]]-sfcPrecip_0[a[0]]))
    #plt.hist(diff/sfcPrecip_test[a[0]]*100)
    b=np.nonzero(diff>1)
    print(int1, len(b[0])/len(a[0]))
