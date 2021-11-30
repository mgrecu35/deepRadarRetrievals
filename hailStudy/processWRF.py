import numpy as np
import wrf as wrf
from netCDF4 import Dataset
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
from read_wrf import *

from scipy.ndimage import gaussian_filter1d

def processWRF(fname,it,itmax,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb,sfcRainRateL,hsL,piaL,tL,pRateL,ifreq):
    ncFile=Dataset(fname)
    print(fname,it)
    dbz2=wrf.getvar(ncFile,'dbz',it)
    h=wrf.getvar(ncFile,'z')
    ncFile.close()
    qr,qs,qg,ncr,ncs,ncg,rho,T,hf,f=read_wrf(fname,it)
    print(qs.max())
    rwc=qr*rho*1e3
    swc=qs*rho*1e3
    gwc=qg*rho*1e3
    ncr=ncr*rho
    ncs=ncs*rho
    ncg=ncg*rho
    a1=np.nonzero(dbz2[0,:,:].data>42)
    #print(a1[0][0],a1[1][0])
    #print(swc[:,100,219])
    hf=hf/9.81e3
    nz=swc.shape[0]
    for it in range(itmax):
        print(it)
        for i0,j0 in zip(a1[0],a1[1]):
            piatot=0
            zmL=[]
            f=1.+np.random.rand()
            f=np.exp(0.5*gaussian_filter1d(np.random.randn(61),sigma=1.5))
            fn=np.exp(0.5*gaussian_filter1d(np.random.randn(61),sigma=1.5))
            prate1=[]
            t1=[]
            for k in range(nz-1,-1,-1):
                t1.append(T[k,i0,j0]-273.15)
                if gwc[k,i0,j0]>0.01:
                    dm_g=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*\
                          (4+mu)**(-mu-1)*f[k]*gwc[k,i0,j0]/fn[k]/ncg[k,i0,j0])**(1/3.0)
                    nw_g=4**4/(np.pi*rhow*1e3)*f[k]*gwc[k,i0,j0]/\
                        (100.0*dm_g)**4
                    lwcg,zg_bh,attg_bh,grauprate_bh,\
                        kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(nw_g,\
                                                                        f_mu,dm_g*1e3,mu,wl[ifreq],\
                                                               refr_ind_s,rhow,rhos)
                    piatot+=attg_bh*(hf[k+1,i0,j0]-hf[k,i0,j0])*2
                else:
                    zg_bh=0
                    grauprate_bh=0
                if swc[k,i0,j0]>0.01:
                    dm_s=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*f[k]*swc[k,i0,j0]/fn[k]/ncg[k,i0,j0])**(1/3.0)
                    nw_s=4**4/(np.pi*rhow*1e3)*f[k]*swc[k,i0,j0]/(100.0*dm_s)**4
                    lwcs,zs_bh,atts_bh,snowrate_bh,\
                        kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(nw_s,\
                                                                        f_mu,dm_s*1e3,mu,wl[ifreq],\
                                                               refr_ind_s,rhow,rhos)
                    piatot+=atts_bh*(hf[k+1,i0,j0]-hf[k,i0,j0])*2
                else:
                    zs_bh=0
                    snowrate_bh=0
                if rwc[k,i0,j0]>0.01:
                    nw_r=0.01
                    dmr=10*dm_lwc(nw_r,rwc[k,i0,j0],1000)
                    dm_r=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*rwc[k,i0,j0]*f[k]/fn[k]/ncr[k,i0,j0])**(1/3.0)
                    nw_r=4**4/(np.pi*rhow*1e3)*f[k]*rwc[k,i0,j0]/(100.0*dm_r)**4
                    lwc,z,att_bh,rrate_bh,\
                    kext_bh,kscat_bh,g_h =bh.dsdintegral(nw_r,f_mu,dm_r*1e3,mu,wl[ifreq],\
                                                         refr_ind_w,rhow)
                    piatot+=att_bh*(hf[k+1,i0,j0]-hf[k,i0,j0])*2
                else:
                    z=0
                    rrate_bh=0
                zm=np.log10(10**(0.1*zs_bh)+10**(0.1*zg_bh)+10**(0.1*z))*10-piatot
                s='%6.2f %6.3f %6.2f %6.2f %6.3f %6.2f '%(gwc[k,i0,j0],rwc[k,i0,j0],zs_bh,z,piatot,zm)
                #print(s)
                zmL.append(zm)
                prate1.append(rrate_bh+snowrate_bh+grauprate_bh)
            #stop
            
            piaL.append(piatot)
            zmint=np.interp(1.75+np.arange(60)*0.25,h[:,i0,j0]/1e3,zmL[::-1])
            tint=np.interp(1.75+np.arange(60)*0.25,h[:,i0,j0]/1e3,t1[::-1])
            prate_int=np.interp(1.75+np.arange(60)*0.25,h[:,i0,j0]/1e3,prate1[::-1])
            pRateL.append(prate_int)
            tL.append(tint)
            zdb.append(zmint)
            sfcRainRateL.append(rrate_bh+snowrate_bh+grauprate_bh)
            hsL.append(hf[0,i0,j0])
        
