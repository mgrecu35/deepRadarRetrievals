import glob
import datetime
from netCDF4 import Dataset
import numpy as np
mypath='/itedata/ITE755/'
import xarray as xr

s=datetime.datetime(2018,10,1)
dwrL=[[],[],[],[],[],[],[],[],[]]
for iday in range(31):
    c=s+datetime.timedelta(days=iday)
    dir3=mypath+'%4.4i/%2.2i/%2.2i/radar/2A.GPM.DPR.V9*'%(c.year,c.month,c.day)
    fs=glob.glob(dir3)
    fs=sorted(fs)
    print(iday)
    for f in fs:
        print(f)
        cAlg=Dataset(f)
        bz=cAlg['FS/VER/binZeroDeg'][:,:]
        pType=(cAlg['FS/CSF/typePrecip'][:,:]/1e7).astype(int)
        sType=(cAlg['FS/PRE/landSurfaceType'][:,:])
        zM=cAlg['FS/PRE/zFactorMeasured'][:,:,:,:]
        bbTop=cAlg['FS/CSF/binBBTop'][:,:]

        a=np.nonzero(pType==1)
        b=np.nonzero(sType[a]!=0)
        for i1,j1 in zip(a[0][b],a[1][b]):
            if abs(j1-24)<12 and bbTop[i1,j1]<bz[i1,j1] and bz[i1,j1]>100 and\
               bbTop[i1,j1]>0:
                bzb=bz[i1,j1]
                for k in range(9):
                    if zM[i1,j1,bzb-2-4*k,0]>10 and zM[i1,j1,bzb-2-4*k,1]>10:
                        dwrL[k].append([zM[i1,j1,bzb-2-4*k,0],\
                                        zM[i1,j1,bzb-2-4*k,1],bzb])

    
    dwrL_0=xr.DataArray(dwrL[0],dims=['n0','ny'])
    dwrL_1=xr.DataArray(dwrL[1],dims=['n1','ny'])
    dwrL_2=xr.DataArray(dwrL[2],dims=['n2','ny'])
    dwrL_3=xr.DataArray(dwrL[3],dims=['n3','ny'])
    dwrL_4=xr.DataArray(dwrL[4],dims=['n4','ny'])
    dwrL_5=xr.DataArray(dwrL[5],dims=['n5','ny'])
    dwrL_6=xr.DataArray(dwrL[6],dims=['n6','ny'])
    dwrL_7=xr.DataArray(dwrL[7],dims=['n7','ny'])
    dwrL_8=xr.DataArray(dwrL[8],dims=['n8','ny'])
    
    d=xr.Dataset({"dwrL_0":dwrL_0,"dwrL_1":dwrL_1,"dwrL_2":dwrL_2,\
                  "dwrL_3":dwrL_3,"dwrL_4":dwrL_4,"dwrL_5":dwrL_5,\
                  "dwrL_6":dwrL_6,"dwrL_7":dwrL_7,"dwrL_8":dwrL_8})
    d.to_netcdf("dwr%2.2i.nc"%iday)
