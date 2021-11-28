import numpy as np

import wrf as wrf
from netCDF4 import Dataset
#from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from read_wrf import *
fname='wrfout_d03_2018-07-29_22:40:00'
ncFile=Dataset(fname)
it=-1
dbz=wrf.getvar(ncFile,'dbz',it)
h=wrf.getvar(ncFile,'z')
ncFile.close()
qr,qs,qg,ncr,ncs,ncg,rho,T,hf,f=read_wrf(fname,it)

rwc=qr*rho*1e3
swc=qs*rho*1e3
gwc=qg*rho*1e3
ncr=ncr*rho
ncs=ncs*rho
ncg=ncg*rho


def dm_lwc(nw,lwc,rho):
    dm=(lwc*1e-3*4**4/(nw*np.pi*rho))**(0.25)
    return dm

from scipy.special import gamma as gam

def fmu(mu):
    return 6/4**4*(4+mu)**(mu+4)/gam(mu+4)


nw=0.08
lwc=np.arange(40)*0.1+0.05
mu=2.0
f_mu=fmu(mu)
dm=10*dm_lwc(nw,lwc,1000)

from bhmief import bhmie
import bhmief as bh
import pytmatrix.refractive
import pytmatrix.refractive as refr

wl=[pytmatrix.refractive.wl_Ku,pytmatrix.refractive.wl_Ka,\
    pytmatrix.refractive.wl_W]
refr_ind_w=pytmatrix.refractive.m_w_0C[wl[0]]


zL=[]
attL=[]
rrateL=[]
snowRateL=[]
refr_ind_w=pytmatrix.refractive.m_w_0C[wl[0]]
rhow=1000.0
rhos=400
refr_ind_s=refr.mi(wl[0],rhos/rhow)

dm2=10*dm_lwc(nw*2,lwc,1000)
for i in range(10,11):
    lwc,z,att_bh,rrate_bh,\
        kext_bh,kscat_bh,g_h =bh.dsdintegral(nw,f_mu,dm[i],mu,wl[0],\
                                             refr_ind_w,rhow)

    lwcs,zs_bh,atts_bh,snowrate_bh,\
        kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_snow(2*nw,\
                                                       f_mu,dm2[i],mu,wl[0],\
                                                       refr_ind_s,rhow,rhos)
    

i0=105
j0=221

a=np.nonzero(gwc>0.1)
dm_g=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*gwc[a]/ncg[a])**(1/3.0)

def nw_lambd(swc,nc,mu):
    rhow=1e6
    lambd=(nc*rhow*np.pi*gam(4+mu)/gam(1+mu)/6.0/swc)**(0.333)  # m-1
    n0=nc*lambd/gam(1+mu) # m-4
    n0*=1e-3 # mm-1 m-3
    #lambd*=1e-2 # cm-1
    return n0,lambd

n0,lambd=nw_lambd(gwc[a],ncg[a],mu)
dm_g2=(4+mu)/lambd
nw_g=4**4/(np.pi*rhow*1e3)*gwc[a].data/(100.0*dm_g.data)**4

#for gwc1,nw1,dm1 in zip(gwc[a],nw_g,dm_g):
#     lwcs,zs_bh,atts_bh,snowrate_bh,\
#        kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_snow(nw1,\
#                                                       f_mu,dm1*1e3,mu,wl[0],\
#                                                       refr_ind_s,rhow,rhos)
#     print(gwc1,lwcs,zs_bh)

piatot=0
hf=hf/9.81e3


f=2.0
rhos=500.0
refr_ind_s=refr.mi(wl[0],rhos/rhow)
a1=np.nonzero(dbz[0,:,:].data>42)
piaL=[]
for i0,j0 in zip(a1[0],a1[1]):
    piatot=0
    zmL=[]
    f=1.+np.random.rand()
    print(i0,j0)
    for k in range(59,-1,-1):
        if gwc[k,i0,j0]>0.01:
            dm_g=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*f*gwc[k,i0,j0]/ncg[k,i0,j0])**(1/3.0)
            nw_g=4**4/(np.pi*rhow*1e3)*f*gwc[k,i0,j0]/(100.0*dm_g)**4
            lwcs,zs_bh,atts_bh,snowrate_bh,\
                kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_snow(nw_g,\
                                                               f_mu,dm_g*1e3,mu,wl[0],\
                                                               refr_ind_s,rhow,rhos)
            piatot+=atts_bh*(hf[k+1,i0,j0]-hf[k,i0,j0])*2
        else:
            zs_bh=0
        if rwc[k,i0,j0]>0.01:
            nw_r=0.01
            dmr=10*dm_lwc(nw_r,rwc[k,i0,j0],1000)
            dm_r=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*f*rwc[k,i0,j0]/ncr[k,i0,j0])**(1/3.0)
            nw_r=4**4/(np.pi*rhow*1e3)*f*rwc[k,i0,j0]/(100.0*dm_r)**4
            lwc,z,att_bh,rrate_bh,\
                kext_bh,kscat_bh,g_h =bh.dsdintegral(nw_r,f_mu,dm_r*1e3,mu,wl[0],\
                                                     refr_ind_w,rhow)
            piatot+=att_bh*(hf[k+1,i0,j0]-hf[k,i0,j0])*2
        else:
            z=0
        zm=np.log10(10**(0.1*zs_bh)+10**(0.1*z))*10-piatot
        s='%6.2f %6.3f %6.2f %6.2f %6.3f %6.2f '%(gwc[k,i0,j0],rwc[k,i0,j0],zs_bh,z,piatot,zm)
        zmL.append(zm)
        #print(s)
        piaL.append(piatot)
    plt.scatter(zmL,h[::-1,i0,j0])
