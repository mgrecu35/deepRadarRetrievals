from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
f=Dataset('cm1out_000032.nc')
dbz=f['dbz'][0,:,:,:]
dbz=np.ma.array(dbz,mask=dbz<10)

plt.pcolormesh(dbz[:,5,:],cmap='jet')
plt.xlim(250,450)
plt.ylim(0,60)
plt.colorbar()

from forwardModel import *
fname='cm1out_000032.nc'

def processFile(fname):
    qr,qs,qg,ncr,ncs,ncg,rho,z,t2c,f=read_wrf(fname,-1)


    swc=rho*(qs)*1.05e3
    gwc=rho*qg*1.5e3
    rwc=rho*qr*1.05e3
    ncr=ncr*rho*1
    ncg=ncg*rho*1
    ncs=(ncs)*rho*2
    zt=swc.copy()*0+1e-9
    prate=swc.copy()*0
    att=swc.copy()*0
    a=np.nonzero(rwc>0.001)
    nwr_a,z_r_a,att_r_a,prate_r=calcZkuR(rwc[a],Deq_r,bscat_r,ext_r,\
                                       vfall_r,mu,wl,nw_dm)
    print(rwc[a].mean())
    prate2=(10**(0.1*z_r_a)/300)**(1/1.4)
    #print(prate.mean(),prate.mean())
    print(np.corrcoef(prate_r,prate2))
    #plt.figure()
    #plt.scatter(rwc[a],prate)
    #plt.show()
    prate[a]=prate_r
    zt[a]+=10**(0.1*z_r_a)
    att[a]+=att_r_a
    a=np.nonzero(swc>0.001)
   
    #stop
    nws_a,z_s_a,att_s_a,prate_s=calcZkuS_2m(swc[a],ncs[a],t2c[a],\
                                            Deq,bscat,ext,vfall,mu,wl)
    prate[a]+=prate_s
    prate_s2=(10**(0.1*z_s_a)/75)**(1/2.0)
    print(np.corrcoef(prate_s,prate_s2))
    #plt.figure()
    #plt.scatter(prate_s,prate_s2)
    #plt.show()
    zt[a]+=10**(0.1*z_s_a)
    att[a]+=att_s_a
    a=np.nonzero(gwc>0.0001)
    nwg_a,z_g_a,att_g_a,prate_g=calcZkuS_2m(gwc[a],ncg[a],t2c[a],Deq,\
                                    bscat,ext,vfall,mu,wl)
    prate[a]+=prate_g
    zt[a]+=10**(0.1*z_g_a)
    att[a]+=att_g_a
    #plt.figure()
    zt=np.log10(zt[:,:,:])*10
    zt_att=zt.copy()
    ztm=np.ma.array(zt,mask=zt<0)
    piaKuL=np.zeros((200,10),float)
    for i in range(250,450):
        for ix in range(10):
            piaKu=0
            for j in range(79,-1,-1):
                piaKu+=att[j,ix,i]*0.25
                zt_att[j,ix,i]-=piaKu
                piaKu+=att[j,ix,i]*0.25
            piaKuL[i-250,ix]=piaKu

    zt_attm=np.ma.array(zt_att,mask=zt_att<0)
    return np.array(piaKuL), zt_attm, prate
nt=0
for i in range(52,53):
    fname='cm1out_0000%2i.nc'%i
    piaKuL,zt_attm,prate=processFile(fname)
    a=np.nonzero(piaKuL>10)
    nt+=len(a[0])
    print(nt)
plt.pcolormesh(zt_attm[:,5,:],cmap='jet',vmin=0)
plt.xlim(250,450)
plt.ylim(0,60)
plt.colorbar()
plt.figure()
import matplotlib
plt.pcolormesh(prate[:,5,:],cmap='jet',vmin=0.01,norm=matplotlib.colors.LogNorm())
plt.xlim(250,450)
plt.ylim(0,60)
plt.colorbar()
