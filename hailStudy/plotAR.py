import numpy as np
from wrf import *
import wrf as wrf
from netCDF4 import Dataset
#from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# draw tissot's indicatrix to show distortion.

ncFile=Dataset('wrfout_d03_2018-07-29_22:40:00')#wrfout_d03_2019-05-20_13:00:00')

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
rainnc=ncFile.variables['RAINNC'][:,:,:]+ncFile.variables['RAINC'][:,:,:]+ncFile.variables['SNOWNC'][:,:,:]
lon=ncFile.variables['XLONG'][0,:,:]
lat=ncFile.variables['XLAT'][0,:,:]

pw=wrf.getvar(ncFile,'pw',-1)

qc=ncFile['QCLOUD'][-1,:,:,:]
tk=wrf.getvar(ncFile,'tk',-1)
import cartopy
import cartopy.crs as ccrs

import cartopy.feature as cf

nx0=120
ny0=150
i0=int(31043/252)
j0=int(31043-i0*252)
i0=113
j0=215
for it in range(0,2):
    fig=plt.figure()
    proj=ccrs.PlateCarree()
    ax=fig.add_axes([0.1,0.1,0.8,0.8],projection=proj)
    dbz=wrf.getvar(ncFile,'dbz',it)
    dbzm=np.ma.array(dbz,mask=dbz<0)

    c=plt.pcolormesh(lon,lat,dbzm[0,:,:],cmap='jet',transform=proj,vmin=0,vmax=40)
    ax.coastlines()
    ax.add_feature(cf.STATES)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    plt.colorbar(c,orientation='horizontal')
    #plt.xlim(125,140)
#plt.ylim(-54,-44)



h=wrf.getvar(ncFile,'height')
h=wrf.getvar(ncFile,'z')
plt.figure()
plt.pcolormesh(range(dbzm.shape[-1]),h[:,i0-2,j0],dbzm[:,i0,:],cmap='jet')
plt.colorbar()

it=0
#qh=ncFile['QHAIL'][1,:,:,:]
qr=ncFile['QRAIN'][it,:,:,:]
qg=ncFile['QGRAUP'][it,:,:,:]
qs=ncFile['QSNOW'][it,:,:,:]

plt.figure()
#plt.plot(qh[:,i0,j0],h[:,i0,j0])
plt.plot(qs[:,i0,j0],h[:,i0,j0])
plt.plot(qg[:,i0,j0],h[:,i0,j0])
plt.plot(qr[:,i0,j0],h[:,i0,j0])
