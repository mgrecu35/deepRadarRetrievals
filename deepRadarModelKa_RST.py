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
zKu,zKu_true,zKa,zKa_true,pRate=[],[],[],[],[]
def readDataset(fname):
    fh=Dataset(fname)
    zKu=fh["zKu_nubf_ms"][:]
    zKu_true=fh["zKu_true"][:]
    zKa=fh["zKa_nubf_ms"][:]
    zKa_true=fh["zKa_true"][:]
    pRate=fh["pRate"][:]
    return zKu,zKu_true,zKa,zKa_true,pRate

import glob
fL=glob.glob("sim_obs_CM1_dn_sl_*.nc")
for f in fL:
    zKu1,zKu_true1,zKa1,zKa_true1,pRate1=readDataset(f)
    zKu.extend(zKu1)
    zKu_true.extend(zKu_true1)
    zKa.extend(zKa1)
    zKa_true.extend(zKa_true1)
    pRate.extend(pRate1)

zKu=np.array(zKu)
zKu_true=np.array(zKu_true)
zKa=np.array(zKa)
zKa_true=np.array(zKa_true)
pRate=np.array(pRate)
    
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
model=lstm_model(2)
X=np.swapaxes(np.array([zKu_sc,zKa_sc]).T,0,1)
x_train=X[ind_train,:,:]
x_test=X[ind_test,:,:]

if itrain==1:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss='mse',\
        metrics=[tf.keras.metrics.MeanSquaredError()])
    history = model.fit(x_train[:,-40:,:], y_train[:,-40:], \
                        batch_size=32,epochs=2,
                        validation_data=(x_test[:,-40:,:], \
                                         y_test[:,-40:]))
else:
    model=tf.keras.models.load_model("radarProfilingDualFreq.h5")

model.save("radarProfilingDualFreq.h5")

yp=model(x_test[:,-40:])
sfcPrecip_0=scalerPrec.scale_[0]*yp[:,-1,0].numpy()+scalerPrec.mean_[0]
sfcPrecip_test=pRate[:,0][ind_test]
a=np.nonzero((sfcPrecip_test-40)*(sfcPrecip_test-50)<0)
diff=np.abs((sfcPrecip_test[a[0]]-sfcPrecip_0[a[0]]))
plt.hist(diff/sfcPrecip_test[a[0]]*100)
