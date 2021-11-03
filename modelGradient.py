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

model=tf.keras.models.load_model('lstm_model.h5')


x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
with tf.GradientTape() as t:
    t.watch(x_tensor)
    output = model(x_tensor)[:,-10,0]

result = output
gradients = t.gradient(output, x_tensor)
