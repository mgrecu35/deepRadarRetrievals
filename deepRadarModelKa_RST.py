import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop



fh=Dataset("sim_obs_CM1.nc")

from sklearn.preprocessing import StandardScaler
scalerZ = StandardScaler()

zKu=fh["zKu_ms"][:]
zKu_true=fh["zKu_true"][:]
fh.close()
scalerZ.fit(zKu_true[:,:])
zKu_sc=scalerZ.transform(zKu)[:,::-1]
zKu_true_sc=scalerZ.transform(zKu_true)[:,::-1]

from sklearn.model_selection import train_test_split

ind=range(zKu.shape[0])

x_train, x_test,\
    y_train, y_test = train_test_split(zKu_sc, zKu_true_sc,\
                                       test_size=0.25, random_state=42)

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
model=lstm_model(1)

if itrain==1:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss='mse',\
        metrics=[tf.keras.metrics.MeanSquaredError()])
    history = model.fit(x_train[:,-40:], y_train[:,-40:], \
                        batch_size=32,epochs=2,
                        validation_data=(x_test[:,-40:], \
                                         y_test[:,-40:]))
else:
    model=tf.keras.models.load_model("radarProfilingKa.h5")

model.save("radarProfilingKu.h5")
