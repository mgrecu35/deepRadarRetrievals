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

import pickle
[zKu,zKa,pRate]=pickle.load(open("nnDataSet.pklz","rb"))


#fh=Dataset("zKuPrecip_Dataset.nc","r")
#zKu=fh["zKu"][:]
#t2c=fh["t2c"][:]
#pRate=fh["pRate"][:]
#fh.close()
#zKu[zKu<0]=0
#stop
zKus=scalerZ.fit_transform(zKu[:,:])
zKas=scalerT.fit_transform(zKa[:,:])
pRates=scalerR.fit_transform(pRate[:,:])
nt=pRate.shape[0]
X=np.zeros((nt,75,1),float)
y=np.zeros((nt,75,1),float)
#X[:,:,0]=zKus
X[:,:,0]=zKas
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

itrain=1
if itrain==1:
    model=lstm_model(1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss='mse',\
        metrics=[tf.keras.metrics.MeanSquaredError()])
    
    
    history = model.fit(X, y, batch_size=32,epochs=20,
                        validation_data=(X, y))
else:
    model=tf.keras.models.load_model("radarProfilingDWR.h5")


#y_OK_=model.predict(X_val_OK)
#y_IPHEX_=model.predict(X_val_IPHEX)
#y_ColStorm_=model.predict(X_val_ColStorm)
