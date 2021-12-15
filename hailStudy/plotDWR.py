import matplotlib.pyplot as plt

from netCDF4 import Dataset
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

fh=Dataset('dwr30.nc')
import numpy as np
fig=plt.figure(figsize=(11,8))
for i in range(9):
    dwr=fh["dwrL_%i"%i][:]
    plt.subplot(3,3,i+1)
    plt.hist(dwr[:,0]-dwr[:,1],density=True,bins=-10+np.arange(25))
    plt.title('%3.3i m above the BB'%(250+i*4*125))
    plt.grid()
    if i>5:
        plt.xlabel("[dB]")
    if i%3==0:
        plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("DWR_KuKa_Distributions.png")
plt.show()
