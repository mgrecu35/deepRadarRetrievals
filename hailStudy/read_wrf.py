import numpy as np

import wrf as wrf
from netCDF4 import Dataset

def read_wrf(fname,it):
    f=Dataset(fname)
    qv=f['QVAPOR'][it,:,:,:]    # water vapor
    qr=f['QRAIN'][it,:,:,:]     # rain mixing ratio
    qs=f['QSNOW'][it,:,:,:]     # snow mixing ratio
    qc=f['QCLOUD'][it,:,:,:]    # cloud mixing ratio
    qg=f['QGRAUP'][it,:,:,:]  # graupel mixing ratio
    ncr=f['QNRAIN'][it,:,:,:]     # rain mixing ratio
    ncs=f['QNSNOW'][it,:,:,:]     # snow mixing ratio
    ncg=f['QNGRAUPEL'][it,:,:,:]   # graupel mixing ratio
    #z=f['z_coords'][:]/1000.             # height (km)
    th=f['T'][it,:,:,:]+300    # potential temperature (K)
    prs=f['P'][it,:,:,:]+f['PB'][it,:,:,:]
    # pressure (Pa)
    T=th*(prs/100000)**0.286  # Temperature
    h=f['PH'][it,:,:,:]+f['PHB'][it,:,:,:]
    t2c=T-273.15
    #stop
  
    R=287.058  #J*kg-1*K-1
    rho=prs/(R*T)
    return qr,qs,qg,ncr,ncs,ncg,rho,T,h,f
