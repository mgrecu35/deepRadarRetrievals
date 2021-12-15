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
#zKuCL.xtend(fh["zKuCorrected"][:])
#zKaL.extend(fh["zKa"][:])
#bcfL.extend(fh["bcf"][:])
#bzdL.extend(fh["bzd"][:])
#fh.close()
from wrf_io import *
fname='../LowLevel/wrfout_d03_2018-06-25_03:36:00'
from wrf_io import read_wrf_nohyd
import combAlg as cAlg
rho,z,T,tsk,w_10,qv,press=read_wrf_nohyd(fname,0)
ix=200
iy=100
freq85=85
ireturn=1
import combAlg as sdsu
sdsu.mainfortpy()
sdsu.initp2()

kextAtmL=[]
tmL=[]
zmL=[]

grauprate=sdsu.tablep2.grauprate[:253]
kextg=sdsu.tablep2.kexts2[:253,:]
salbg=sdsu.tablep2.salbs2[:253,:]
asymg=sdsu.tablep2.asyms2[:253,:]

hailrate=sdsu.tablep2.hailrate[:270]
kexth=sdsu.tablep2.kexth[:270,:]
salbh=sdsu.tablep2.salbh[:270,:]
asymh=sdsu.tablep2.asymh[:270,:]

for k in range(59):
    absair,abswv = sdsu.gasabsr98(freq85,T[k,iy,ix],qv[k,iy,ix]*rho[k,iy,ix],\
                                 press[k,iy,ix],ireturn)
    kextAtmL.append(absair+abswv)
    zmL.append(z[k:k+2,iy,ix].mean())
    tmL.append(T[k,iy,ix])

kextAtm1d=np.interp(0.125/2+np.arange(176)*0.125,zmL,kextAtmL)
T1d=np.interp(np.arange(177)*0.125,zmL,tmL)
#stop
    
scattCoeff_ice=np.array([[-2.849, 1.277, -0.050],\
                         [-0.015, 0.003, -0.0004],\
                         [-3.69,  0.97,   0.016],\
                         [0.388, -0.0004, 0.042]])

scattCoeff_ice=np.array([[-3.449, 1.277, -0.050],\
                         [-0.015, 0.003, -0.0004],\
                         [-3.69,  0.97,   0.016],\
                         [0.388, -0.0004, 0.042]])

scattCoeff_w=np.array([[-1.237, 0.856, -0.024],\
                       [-0.844, 0.081, -0.006],\
                       [-2.08,  0.90,  -0.034],\
                       [0.128,  0.00016, 0.050]])

import bisectm as b
def calcTb(prate1,bzd,bcf,kextAtm1d,T1d,scattCoeff_w,scattCoeff_ice,\
           hailrate,kexth,salbh,asymh,\
           grauprate,kextg,salbg,asymg):
    kext1d=kextAtm1d.copy()
    salb1d=kextAtm1d.copy()*0
    asym1d=kextAtm1d.copy()*0
    kext1d_h=kextAtm1d.copy()
    salb1d_h=kextAtm1d.copy()*0
    asym1d_h=kextAtm1d.copy()*0
    prate1[prate1>128]=128
    for k in range(0,bzd):
        if prate1[k]>0.01:
            kext=np.exp(scattCoeff_ice[0,0]+scattCoeff_ice[0,1]*\
                        np.log(prate1[k])+\
                        scattCoeff_ice[0,2]*np.log(prate1[k])**2)
            salb=np.exp(scattCoeff_ice[1,0]+scattCoeff_ice[1,1]*\
                        np.log(prate1[k])+\
                        scattCoeff_ice[1,2]*np.log(prate1[k])**2)
            kscat=kext*salb
            g=scattCoeff_ice[3,0]+scattCoeff_ice[3,1]*(prate1[k])+\
                scattCoeff_ice[3,2]*np.log(prate1[k])

            kext_tot=kextAtm1d[175-k]+kext
            kscat_tot=kscat
            gtot=kscat*g
            kext1d[175-k]=kext_tot
            salb1d[175-k]=kscat_tot/kext_tot
            asym1d[175-k]=gtot/(kscat_tot+1e-9)
            nv=grauprate.shape[0]
            #print(nv)
            n1=b.bisectm(grauprate,nv,prate1[k])
            kext=kextg[n1,4]
            salb=salbg[n1,4]
            asym=asymg[n1,4]
            if kext>-15:
                nv=hailrate.shape[0]
                n1=b.bisectm(hailrate,nv,prate1[k])
                kext=kexth[n1,4]
                salb=salbh[n1,4]
                asym=asymh[n1,4]
            kext_tot=kextAtm1d[175-k]+kext
            kscat_tot=kscat
            gtot=kscat*g
            kext1d_h[175-k]=kext_tot
            salb1d_h[175-k]=kscat_tot/kext_tot
            asym1d_h[175-k]=gtot/(kscat_tot+1e-9)
    for k in range(bzd,bcf):
        if prate1[k]>0.01:
            kext=np.exp(scattCoeff_w[0,0]+scattCoeff_w[0,1]*\
                        np.log(prate1[k])+\
                        scattCoeff_w[0,2]*np.log(prate1[k])**2)
            salb=np.exp(scattCoeff_w[1,0]+scattCoeff_w[1,1]*\
                        np.log(prate1[k])+\
                        scattCoeff_w[1,2]*np.log(prate1[k])**2)
            kscat=kext*salb
            g=scattCoeff_w[3,0]+scattCoeff_w[3,1]*(prate1[k])+\
                scattCoeff_w[3,2]*np.log(prate1[k])

            kext_tot=kextAtm1d[175-k]+kext
            kscat_tot=kscat
            gtot=kscat*g
            kext1d[175-k]=kext_tot
            salb1d[175-k]=kscat_tot/kext_tot
            asym1d[175-k]=gtot/(kscat_tot+1e-9)
            kext1d_h[175-k]=kext_tot
            salb1d_h[175-k]=kscat_tot/kext_tot
            asym1d_h[175-k]=gtot/(kscat_tot+1e-9)
    return kext1d,salb1d,asym1d,kext1d_h,salb1d_h,asym1d_h

for f in sorted(fs[:60]):
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

stop
hlayer=np.arange(177)*0.125
lambert=1
emis=0.9
ebar=0.9
fisot=2.7
umu=np.cos(53/180.0*np.pi)
simTbL=[]
ifail=0
for i, pRate in enumerate(precipRateL):
    kext1d,salb1d,asym1d,\
        kext1d_h,salb1d_h,asym1d_h=calcTb(pRate,bzdL[i],bcfL[i],\
                                          kextAtm1d,T1d,\
                                          scattCoeff_w,scattCoeff_ice,\
                                          hailrate,kexth,salbh,asymh,\
                                          grauprate,kextg,salbg,asymg)
    n0=int((175-bcfL[i])/2)
    T1d_2=T1d[::2]
    h_2=hlayer[::2]
    kext1d_2=kext1d[::2]
    salb1d_2=salb1d[::2]
    asym1d_2=asym1d[::2]
    tb =sdsu.radtran(umu,T1d_2[n0],T1d_2[n0:],h_2[n0:],kext1d_2[n0:],\
                     salb1d_2[n0:],\
                     asym1d_2[n0:],fisot,emis,ebar,lambert)
    kext1d_2=kext1d_h[::2]
    salb1d_2=salb1d_h[::2]
    asym1d_2=asym1d_h[::2]
    #tb_h =sdsu.radtran(umu,T1d_2[n0],T1d_2[n0:],h_2[n0:],kext1d_2[n0:],\
    #                 salb1d_2[n0:],\
    #                 asym1d_2[n0:],fisot,emis,ebar,lambert)

    if tb!=tb:
        print("failure %i",ifail)
        ifail+=1
        stop
    else:
        simTbL.append([tb,pctL[i][1]])
    #stop


def bisect_test(xvec,r):
    n1=0
    n2=xvec.shape[0]-1
    print(n1,n2)
    if r<xvec[0]:
        return 0
    if r>xvec[n2]:
        return n2
    nmid=int((n1+n2)/2)
    #print(nmid)
    print("here")
    it=0
    while not (r>=xvec[nmid-1] and r<chist[nmid]) and it<7:
        it+=1
        #print(chist[nmid-1],r,chist[nmid],nmid,n1,n2)
        if r>chist[nmid-1]:
            n1=nmid
        else:
            n2=nmid
        nmid=int((n1+n2)/2)
    return nmid
