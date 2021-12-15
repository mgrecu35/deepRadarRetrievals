from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

import glob
fs=glob.glob("Conv/ku*Conv*paral*nc")

zKuL=[]
zKaL=[]
zKuCL=[]
bcfL=[]
bzdL=[]
sfcPrecipL=[]
precipRateL=[]
pctL=[]
#zKuL.extend(fh["zKu"][:])
#zKuCL.extend(fh["zKuCorrected"][:])
#zKaL.extend(fh["zKa"][:])
#bcfL.extend(fh["bcf"][:])
#bzdL.extend(fh["bzd"][:])
#fh.close()

for f in sorted(fs):
    #if "parall" not in f:
    #    continue
    fh=Dataset(f)
    zKuL.extend(fh["zKu"][:])
    zKuCL.extend(fh["zKuCorrected"][:])
    zKaL.extend(fh["zKa"][:])
    bcfL.extend(fh["bcf"][:])
    bzdL.extend(fh["bzd"][:])
    precipRateL.extend(fh["precipRate"][:])
    sfcPrecipL.extend(fh["sfcPrecip"][:])
    pctL.extend(fh["PCT"][:])
    fh.close()

icount=0
zKutL=[]
zKuCtL=[]
zKatL=[]
zsfcL=[]
sfcPreciptL=[]
pct_tL=[]
precipRatetL=[]
zKutL_=[]
zKuCtL_=[]
zKatL_=[]
zsfcL_=[]
sfcPreciptL_=[]
precipRatetL_=[]
pct_tL_=[]
for i in range(len(zKuL)):
    dn=bcfL[i]-bzdL[i]
    if dn>=25 and sfcPrecipL[i]==sfcPrecipL[i]:
        a1=np.nonzero(zKuL[i][bzdL[i]-50:bzdL[i]-4]>20)
        if len(a1[0])>3 and pctL[i][1]<300:
            if np.random.rand()<0.5:
                icount+=1
                zKutL.append(zKuL[i][bzdL[i]-75:bzdL[i]+25])
                zKuCtL.append(zKuCL[i][bzdL[i]-75:bzdL[i]+25])
                zKatL.append(zKaL[i][bzdL[i]-75:bzdL[i]+25])
                precipRatetL.append(precipRateL[i][bzdL[i]-75:bzdL[i]+25])
                sfcPreciptL.append(precipRateL[i][bzdL[i]+24])
                pct_tL.append(pctL[i])
            else:
                zKutL_.append(zKuL[i][bzdL[i]-75:bzdL[i]+25])
                zKuCtL_.append(zKuCL[i][bzdL[i]-75:bzdL[i]+25])
                zKatL_.append(zKaL[i][bzdL[i]-75:bzdL[i]+25])
                precipRatetL_.append(precipRateL[i][bzdL[i]-75:bzdL[i]+25])
                sfcPreciptL_.append(precipRateL[i][bzdL[i]+24])
                pct_tL_.append(pctL[i])
                #stop


zKutL=np.array(zKutL)
zKuCtL=np.array(zKuCtL)
zKatL=np.array(zKatL)
zKatL[zKatL<10]=0
zKutL[zKutL<10]=0
zKuCtL[zKuCtL<10]=0
precipRatetL=np.array(precipRatetL)
sfcPreciptL=np.array(sfcPreciptL)
pct_tL=np.array(pct_tL)
pct_tL_=np.array(pct_tL_)

zKutL_=np.array(zKutL_)
zKuCtL_=np.array(zKuCtL_)
zKatL_=np.array(zKatL_)
zKatL_[zKatL_<10]=0
zKutL_[zKutL_<10]=0
zKuCtL_[zKuCtL_<10]=0
precipRatetL_=np.array(precipRatetL_)
sfcPreciptL_=np.array(sfcPreciptL_)

import pickle

pickle.dump({"zKu":zKutL,"zKa":zKatL,"precipRate":precipRatetL,"sfcPrecip":sfcPreciptL,\
             "zKu_t":zKutL_,"zKa_t":zKatL_,"precipRate_t":precipRatetL_,"sfcPrecip_":sfcPreciptL_,"pct":pct_tL,"pct_t":pct_tL_},\
            open("convProf_pct_parall.pklz","wb"))

stop
from minisom import MiniSom
#from minisom import MiniSom

n1=50
n2=1
nz=100
som = MiniSom(n1,n2,nz,sigma=2.5,learning_rate=0.5, random_seed=0)
#np.random.seed(seed=10)
som.random_weights_init(zKutL)
som.train_random(zKutL,500) # training with 100 iterations
nt=zKutL.shape[0]
winL=np.zeros((nt),int)
it=0
zKuClass=np.zeros((n1,nz),float)
zKaClass=np.zeros((n1,nz),float)
pRateClass=np.zeros((n1),float)
pRate1DClass=np.zeros((n1,nz),float)
countSClass=np.zeros((n1),float)
for it,z1 in enumerate(zKutL):
    win=som.winner(z1)
    winL[it]=win[0]

for i in range(n1):
    a=np.nonzero(winL==i)
    zKuClass[i]=zKutL[a[0],:].mean(axis=0)
    zKaClass[i]=zKatL[a[0],:].mean(axis=0)
    pRateClass[i]=sfcPreciptL[a[0]].mean()
    pRate1DClass[i]=precipRatetL[a[0],:].mean(axis=0)
    countSClass[i]=len(a[0])
    
plt.figure()
plt.pcolormesh(range(n1),np.arange(nz)-75,zKuClass.T,cmap='jet',vmin=0)
plt.ylim(24,-75)
plt.colorbar()
plt.figure()
plt.pcolormesh(range(n1),np.arange(nz)-75,zKaClass.T,cmap='jet',vmin=0)
plt.ylim(24,-75)
plt.colorbar()
plt.figure()
plt.subplot(211)
plt.plot(range(n1),pRateClass)
plt.subplot(212)
plt.plot(range(n1),countSClass/countSClass.sum()*100)
plt.figure()
plt.pcolormesh(range(n1),np.arange(nz)-75,pRate1DClass.T,cmap='jet',vmin=0)
plt.ylim(24,-75)
plt.colorbar()

cumSum=np.zeros((100),float)
for i in range(100):
    a=np.nonzero(sfcPreciptL<i+0.5)
    cumSum[i]=sfcPreciptL[a].sum()/sfcPreciptL.sum()*100

    #plt.plot(zKutL.mean(axis=0))
#plt.plot(zKuCtL.mean(axis=0))
#plt.plot(zKatL.mean(axis=0))
#sfcPreciptL=np.array(sfcPreciptL)
#stop
#from sklearn.neighbors import KNeighborsRegressor
#knn = KNeighborsRegressor(30, weights='distance')
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(zKatL[:,:], sfcPreciptL[:], test_size=0.33, random_state=42)
#knn.fit(X_train,y_train)
#y_=knn.predict(X_test)

import pickle
pickle.dump([zKutL,zKatL,precipRatetL],open("nnDataSet.pklz","wb"))


cumSum=np.zeros((100),float)
for i in range(100):
    a=np.nonzero(r<i+0.5)
    cumSum[i]=r[a].sum()/r.sum()*100
