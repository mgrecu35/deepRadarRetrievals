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

fh=Dataset("zKuPrecip_NUBF_Dataset.nc","r")
zKu=fh["zKu_MS_NUBF"][:]
t2c=fh["t2c"][:]
pRate=fh["pRate"][:]
zKuHB=fh["zKuHB"][:]
zKu[zKu<0]=0
zKuHB[zKuHB<0]=0
#stop
zKus=scalerZ.fit_transform(zKu)[:,::-1][:,-30:]
zKuHBs=scalerZHB.fit_transform(zKuHB)[:,::-1][:,-30:]
t2cs=scalerT.fit_transform(t2c)[:,::-1][:,-30:]
pRates=scalerT.fit_transform(pRate)[:,::-1][:,-30:]
nt=pRate.shape[0]
X=np.zeros((nt,30,3),float)
y=np.zeros((nt,30,1),float)
X[:,:,0]=zKus
X[:,:,1]=zKuHBs
X[:,:,2]=t2cs
y[:,:,0]=pRates

def lstm_model(ndims=2):
    ntimes=None
    inp = tf.keras.layers.Input(shape=(ntimes,ndims,))
    out1 = tf.keras.layers.LSTM(12, return_sequences=True)(inp)
    out1 = tf.keras.layers.LSTM(12, recurrent_activation='sigmoid',return_sequences=True)(out1)
    out = tf.keras.layers.LSTM(1, recurrent_activation=None, \
                               return_sequences=True)(out1)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

model=lstm_model(3)


model.compile(
    optimizer=tf.keras.optimizers.Adam(),  \
    loss='mse',\
    metrics=[tf.keras.metrics.MeanSquaredError()])


history = model.fit(X, y, batch_size=32,epochs=50,
                    validation_data=(X, y))
