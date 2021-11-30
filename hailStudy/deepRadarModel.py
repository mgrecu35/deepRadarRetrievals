import pickle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np 
from netCDF4 import Dataset

from sklearn.preprocessing import StandardScaler
scalerZ = StandardScaler()
scalerZHB = StandardScaler()
scalerT = StandardScaler()
scalerR = StandardScaler()

fh=Dataset("zKuPrecip_Dataset.nc","r")
zKu=fh["zKu"][:]
t2c=fh["t2c"][:]
pRate=fh["pRate"][:]
fh.close()
zKu[zKu<0]=0
#stop
zKus=scalerZ.fit_transform(zKu[:,::-1])
t2cs=scalerT.fit_transform(t2c[:,::-1])
pRates=scalerR.fit_transform(pRate[:,::-1])
nt=pRate.shape[0]
X=np.zeros((nt,60,2),float)
y=np.zeros((nt,60,1),float)
X[:,:,0]=zKus
X[:,:,1]=t2cs
y[:,:,0]=pRates

def getData(fname,scalerZ,scalerT,scalerR):
    fh=Dataset(fname,"r")
    zKu=fh["zKu"][:]
    t2c=fh["t2c"][:]
    pRate=fh["pRate"][:]
    fh.close()
    zKu[zKu<0]=0
    zKus=scalerZ.transform(zKu[:,::-1])
    t2cs=scalerT.transform(t2c[:,::-1])
    pRates=scalerR.transform(pRate[:,::-1])
    nt=pRate.shape[0]
    X=np.zeros((nt,60,2),float)
    y=np.zeros((nt,60,1),float)
    X[:,:,0]=zKus
    X[:,:,1]=t2cs
    y[:,:,0]=pRates
    return X,y

X_val_IPHEX,y_IPHEX=getData("zKuPrecip_Dataset_Validation_IPHEX.nc",scalerZ,scalerT,scalerR)
X_val_ColStorm,y_ColStorm=getData("zKuPrecip_Dataset_Validation_ColStorm.nc",scalerZ,scalerT,scalerR)

def lstm_model(ndims=2):
    ntimes=None
    inp = tf.keras.layers.Input(shape=(ntimes,ndims,))
    out1 = tf.keras.layers.LSTM(12, return_sequences=True)(inp)
    out1 = tf.keras.layers.LSTM(12, recurrent_activation='sigmoid',return_sequences=True)(out1)
    out = tf.keras.layers.LSTM(1, recurrent_activation=None, \
                               return_sequences=True)(out1)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

itrain=0
if itrain==1:
    model=lstm_model(2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss='mse',\
        metrics=[tf.keras.metrics.MeanSquaredError()])
    
    
    history = model.fit(X, y, batch_size=64,epochs=120,
                        validation_data=(X_val_IPHEX, y_IPHEX))
else:
    model=tf.keras.models.load_model("radarProfiling.h5")

