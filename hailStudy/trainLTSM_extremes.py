import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np 
from tensorflow.keras import datasets, layers, models

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

def cnn_model(ndims):
    inp = tf.keras.layers.Input(shape=(ndims,1))
    out = layers.Conv1D(32, 3, activation='relu',input_shape=(ndims,1))(inp)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = layers.Conv1D(32, 3, activation='relu',input_shape=(ndims,1))(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = layers.Conv1D(16, 3, activation='relu',input_shape=(ndims,1))(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = layers.Conv1D(16, 3, activation='relu',input_shape=(ndims,1))(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = layers.Conv1D(16, 3, activation='relu',input_shape=(ndims,1))(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = tf.keras.layers.Dense(8)(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = tf.keras.layers.Dense(1,activation='relu')(out)
    #model.add(layers.MaxPooling1D(2))
    #model.add(layers.Conv1D(16, 3, activation='relu'))
    #model.add(layers.MaxPooling1D(2))
    #model.add(layers.Conv2D(16, 3, activation='relu'))
    return tf.keras.Model(inputs=inp, outputs=out)

def cnn_model_sfc(ndims):
    inp = tf.keras.layers.Input(shape=(ndims,1))
    out = layers.Conv1D(16, 3, activation='relu',input_shape=(ndims,1))(inp)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = layers.Conv1D(15, 3, activation='relu',input_shape=(ndims,1))(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = layers.Conv1D(16, 3, activation='relu',input_shape=(ndims,1))(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = layers.Conv1D(16, 3, activation='relu',input_shape=(ndims,1))(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = layers.Conv1D(16, 3, activation='relu',input_shape=(ndims,1))(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = tf.keras.layers.Dense(8)(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = tf.keras.layers.Dense(1)(out)
    #model.add(layers.MaxPooling1D(2))
    #model.add(layers.Conv1D(16, 3, activation='relu'))
    #model.add(layers.MaxPooling1D(2))
    #model.add(layers.Conv2D(16, 3, activation='relu'))
    return tf.keras.Model(inputs=inp, outputs=out)

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
y=sfcPrecip[a[0]]*0
y[sfcPrecip[a[0]]>45]=1

zKas_t=scalerZ.transform(zKa_t[a_[0],:])
sfcPrecipS_t=scalerR.transform(sfcPrecip_t[a_[0],np.newaxis])
y_t=sfcPrecip_t[a_[0]]*0
y_t[sfcPrecip_t[a_[0]]>45]=1

model=lstm_model(1)
itrain=1
if itrain==1:
    model=cnn_model_sfc(100)
    #model=lstm_model(1)
    #model.compile(
    #    optimizer=tf.keras.optimizers.Adam(),  \
    #    loss=tf.keras.losses.BinaryCrossentropy(),\
    #    metrics=['accuracy'])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss=tf.keras.losses.MeanSquaredError(),\
        metrics=['mse'])
    
    
    history = model.fit(zKas[:,:],sfcPrecipS, batch_size=32,epochs=50,
                        validation_data=(zKas_t[:,:], sfcPrecipS_t))
    model.save("radarProfilingKa.h5")

stop
def score(y,yp_):
    a1=np.nonzero(y>0.9)
    a2=np.nonzero(yp_[a1[0],0]>0.9)
    print(len(a2[0])/len(a1[0]))
    
xL=[]
yL=[]
for i in a[0]:
    z1=zKa[i,:]
    b=np.nonzero(z1[0:75]>30)
    x1=[z1[0:75].max()/35.]
    ik=np.argmax(z1)
    x1.append((10**(0.1*z1[0:75])).sum()/5e3)
    a1=np.polyfit(range(25),z1[75:],1)
    x1.extend([len(b[0])/5,a1[0]/0.3,a1[1]/27,ik/70])
    #xL.append(x1)
    xL.append(z1)
    if sfcPrecip[i]>45:
        yL.append(1+0.*sfcPrecip[i])
    else:
        yL.append(0+0*sfcPrecip[i])

from sklearn.decomposition import PCA
scaler = StandardScaler()
xL=np.array(xL)
yL=np.array(yL)
xL = scaler.fit_transform(xL)
pca = PCA(n_components=22)
pca.fit(xL)

xL=pca.transform(xL)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(xL, yL, random_state=0)
#clf = DecisionTreeRegressor(max_leaf_nodes=5, random_state=0)
clf = RandomForestClassifier(max_leaf_nodes=5, class_weight={0:1,1:2.5},random_state=0)
clf.fit(X_train, y_train)

stop
