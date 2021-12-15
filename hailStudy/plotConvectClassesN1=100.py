import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 12})

d=pickle.load(open("convProf_pct.pklz","rb"))

zKu=d["zKu"]
zKa=d["zKa"]
precipRate=d["precipRate"]
sfcPrecip=d["sfcPrecip"]
pct=d["pct"]
zKu_=d["zKu_t"]
zKa_=d["zKa_t"]
precipRate_=d["precipRate_t"]
sfcPrecip_=d["sfcPrecip_"]
pct_=d["pct_t"]

from minisom import MiniSom

n1=50
n2=1
nz=100
som = MiniSom(n1,n2,nz,sigma=2.5,learning_rate=0.5, random_seed=0)
#np.random.seed(seed=10)
som.random_weights_init(zKu)
som.train_random(zKu,500) # training with 100 iterations
nt=zKu.shape[0]
winL=np.zeros((nt),int)
it=0
zKuClass=np.zeros((n1,nz),float)
zKaClass=np.zeros((n1,nz),float)
pRateClass=np.zeros((n1),float)
pRate1DClass=np.zeros((n1,nz),float)
countSClass=np.zeros((n1),float)
pctClass=np.zeros((n1,2),float)
for it,z1 in enumerate(zKu):
    win=som.winner(z1)
    winL[it]=win[0]

for i in range(n1):
    a=np.nonzero(winL==i)
    zKuClass[i]=zKu[a[0],:].mean(axis=0)
    zKaClass[i]=zKa[a[0],:].mean(axis=0)
    pRateClass[i]=sfcPrecip[a[0]].mean()
    pRate1DClass[i]=precipRate[a[0],:].mean(axis=0)
    countSClass[i]=len(a[0])
    pctClass[i,:]=pct[a[0],:].mean(axis=0)
plt.figure()
plt.pcolormesh(range(n1),np.arange(nz)-75,zKuClass.T,cmap='jet',vmin=0)
plt.ylim(24,-75)
plt.grid()
plt.title("Ku-band")
plt.xlabel("Class #")
plt.ylabel("Relative range")
c=plt.colorbar()
c.ax.set_title('dBZ')
plt.savefig('class_zKu_pct_dist.png')

plt.figure()
plt.pcolormesh(range(n1),np.arange(nz)-75,zKaClass.T,cmap='jet',vmin=0)
plt.ylim(24,-75)
plt.xlabel("Class #")
plt.ylabel("Relative range")
plt.title("Ka-band")
plt.grid()
c=plt.colorbar()
c.ax.set_title('dBZ')
plt.savefig('class_zKa_pct_dist.png')

plt.figure()
f=plt.subplot(211)
plt.plot(range(n1),pRateClass)
plt.title("Mean surface precipitation by class")
plt.ylabel("[mm/h]")
#f.axes.xaxis.set_visible=False
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
plt.xlim(0,n1-1)
plt.grid()
plt.subplot(212)
plt.plot(range(n1),pctClass[:,0])
plt.plot(range(n1),pctClass[:,1])
plt.xlabel("Class #")
plt.title("PCT")
plt.ylabel("K")
plt.grid()
plt.xlim(0,n1-1)
plt.savefig('classStats_pct.png')

#plt.figure()
#plt.pcolormesh(range(n1),np.arange(nz)-75,pRate1DClass.T,cmap='jet',vmin=0)
#plt.ylim(24,-75)
#plt.colorbar()

cumSum=np.zeros((100),float)
for i in range(100):
    a=np.nonzero(sfcPrecip<i+0.5)
    cumSum[i]=sfcPrecip[a].sum()/sfcPrecip.sum()*100

plt.figure()
plt.plot(np.arange(100)+0.5,cumSum)
plt.xlabel('Precipitation rate (mm/h)')
plt.ylabel('Percentage of lighter precipitation [%]')
plt.grid()
plt.xlim(0,100)
plt.ylim(0,100)
plt.savefig('precipRatePDF.png')


#stop
n1ka=1000
somka = MiniSom(n1ka,n2,nz,sigma=4.5,learning_rate=0.25, random_seed=0)
#np.random.seed(seed=10)
somka.random_weights_init(zKa)
somka.train_random(zKa,500) # training with 100 iterations
nt=zKu.shape[0]
winkaL=np.zeros((nt),int)
it=0
pRateKaClass=np.zeros((n1ka),float)
for it,z1 in enumerate(zKa):
    winka=somka.winner(z1)
    winkaL[it]=winka[0]

for i in range(n1ka):
    a=np.nonzero(winkaL==i)
    if(len(a[0])>0):
        pRateKaClass[i]=sfcPrecip[a].mean()

rainAv1=np.zeros((n1),float)
rainAv2=np.zeros((n1),float)
rainAv3=np.zeros((n1),float)
ccount=np.zeros((n1),float)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaler_pct = StandardScaler()
#zKa[:,0:2]=pct
#zKa_[:,0:2]=pct_

zKas = scaler.fit_transform(zKa)
pcts = scaler_pct.fit_transform(pct)
zKas_= scaler.transform(zKa_)
pcts_ = scaler_pct.fit_transform(pct_)
pca2 = PCA(n_components=22)
pca2.fit(zKas)

xKa = pca2.transform(zKas)
xKa_ = pca2.transform(zKas_)
xKa[:,-2:]=pcts
xKa_[:,-2:]=pcts_
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(30, weights='distance')

knn.fit(xKa,sfcPrecip)
y_=knn.predict(xKa_)
y2_=y_.copy()*0
for i,z1 in enumerate(zKa_):
    winka=somka.winner(z1)
    win=som.winner(zKu_[i])
    rainAv1[win[0]]+=pRateKaClass[winka[0]]
    y2_[i]=pRateKaClass[winka[0]]
    rainAv3[win[0]]+=y_[i]
    rainAv2[win[0]]+=sfcPrecip_[i]
    ccount[win[0]]+=1

plt.figure()
#plt.plot(rainAv1/ccount)
#plt.plot(rainAv3/ccount)
#plt.plot(rainAv2/ccount)
ax=plt.subplot(111)
plt.scatter(rainAv2/ccount,rainAv3/ccount)
plt.title('KuKa vs Ka aggregated by class')
plt.xlabel('KuKa surface precipitation rate [mm/h]')
plt.ylabel('Ka surface precipitation rate [mm/h]')
ax.set_aspect('equal')
plt.xlim(0,30)
plt.ylim(0,30)
plt.grid()
plt.savefig('scatterByClass_pct.png')

cumSum2=np.zeros((100),float)
cumSum3=np.zeros((100),float)
for i in range(100):
    a=np.nonzero(y_<i+0.5)
    cumSum2[i]=y_[a].sum()/y_.sum()*100
    cumSum3[i]=y2_[a].sum()/y2_.sum()*100

plt.figure()
plt.plot(np.arange(100)+0.5,cumSum)
plt.plot(np.arange(100)+0.5,cumSum2,color='red')
plt.plot(np.arange(100)+0.5,cumSum3,color='green')
plt.xlabel('Precipitation rate (mm/h)')
plt.ylabel('Percentage of lighter precipitation [%]')
plt.legend(['KuKa','Ka','Ka-2'])
plt.grid()
plt.xlim(0,100)
plt.ylim(0,100)
plt.savefig('precipRatePDF_pct.png')
