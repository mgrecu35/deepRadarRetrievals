import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np 

matplotlib.rcParams.update({'font.size': 12})

d=pickle.load(open("convProf_pct_parall.pklz","rb"))

zKu=d["zKu"]
zKa=d["zKa"]
precipRate=d["precipRate"]
sfcPrecip=d["sfcPrecip"]
a=np.nonzero(sfcPrecip>0)
zKu_t=d["zKu_t"]
zKa_t=d["zKa_t"]
precipRate_t_=d["precipRate_t"]
sfcPrecip_t=d["sfcPrecip_"]
a_=np.nonzero(sfcPrecip_t>0)

stop
def lstm_model(ndims=2):
    ntimes=None
    inp = tf.keras.layers.Input(shape=(ntimes,ndims,))
    out1 = tf.keras.layers.LSTM(12, return_sequences=True)(inp)
    out1 = tf.keras.layers.LSTM(12, recurrent_activation='sigmoid',\
                                return_sequences=True)(out1)
    out = tf.keras.layers.LSTM(1, recurrent_activation=None, \
                               return_sequences=False)(out1)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

from sklearn.preprocessing import StandardScaler
scalerZ = StandardScaler()
scalerR = StandardScaler()
zKas=scalerZ.fit_transform(zKa[a[0],:])
sfcPrecipS=scalerR.fit_transform(sfcPrecip[a[0],np.newaxis])

zKas_t=scalerZ.transform(zKa_t[a_[0],:])
sfcPrecipS_t=scalerR.transform(sfcPrecip_t[a_[0],np.newaxis])

model=lstm_model(1)
itrain=1
if itrain==1:
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss='binary_crossentropy',\
        metrics=[tf.keras.metrics.MeanSquaredError()])
    
    
    history = model.fit(zKas, sfcPrecipS, batch_size=64,epochs=50,
                        validation_data=(zKas_t, sfcPrecipS_t))
    model.save("radarProfilingKa_sfc.h5")

    
    

    
