import numpy as np
from bisectm import *
zS_coeff=np.array([1.34204631, 2.11016229])
def calcZ(dNw,dng,rain_rate,graup_rate,sdsu):
    if rain_rate/10**dNw>1e-2:
        ibin=bisectm(sdsu.tablep2.rainrate[:289],289,rain_rate/10**dNw)
        zKur=sdsu.tablep2.zkur[ibin]+10*dNw
        attKur=sdsu.tablep2.attkur[ibin]*10**dNw
    else:
        zKur=-0
        attKur=0
    if graup_rate/10**dng>1e-3:
        ibin=bisectm(sdsu.tablep2.snowrate[:253],253,graup_rate/10**dng)
        if graup_rate/10**dng<333:
            if ibin<251:
                zKug=sdsu.tablep2.zkus[ibin+1]+10*dng
            else:
                zKug=sdsu.tablep2.zkus[ibin]+10*dng
        else:
            zKug=10*(zS_coeff[0]*np.log10(graup_rate/10**dng)+\
                     zS_coeff[1]+dng)
        attKug=sdsu.tablep2.attkus[ibin]*10**dng
    else:
        zKug=-0
        attKug=0
    return zKur,attKur,zKug,attKug

from scipy.ndimage import gaussian_filter


    

def backwards_kuret(dr,sdsu,iz,zku_obs,piaKu):
    nz=zku_obs.shape[0]
   
    dn_coeff=np.array([-0.01489413,  0.91568719])

    zKu=np.zeros((nz),float)
    precipRate=np.zeros((nz),float)
    ztrue1D=np.zeros((nz),float)
    dNw=np.zeros((nz),float)
    dnp=np.random.randn(47)
    dnp=gaussian_filter(dnp,sigma=4)*0


    for k in range(nz-1,-1,-1):
        dng=-.5+(iz-k)*0.04
        if k==33:
            dng-=0.3
        if k==32:
            dng-=0.2
        if k==31:
            dng-=0.2
        if k==30:
            dng-=0.1
        if k==29:
            dng-=0.0
            
        
        attKu=0
        dng+=dnp
        if k>33:
            for it in range(2):
                z_true=zku_obs[k]+piaKu-attKu*dr
                dNw[k]=np.polyval(dn_coeff,z_true)+dnp[k]
                          
                ibin=int((z_true-10*dNw[k]+12)/0.25)
                ibin=max(0,ibin)
                ibin=min(ibin,288)
                #print(piaKu,zku_obs[k],z_true,dNw[k],ibin)
                precipRate[k]=sdsu.tablep2.rainrate[ibin]*10**dNw[k]
                attKu=sdsu.tablep2.attkur[ibin]*10**dNw[k]
            ztrue1D[k]=z_true
            piaKu-=2*attKu*dr
            piaKu=max(0,piaKu)
        else:
            if k>25:
                prate1=precipRate[k+1]
              
                f=(k-26)/8
                f=f
                dNw[:34]=dNw[34]
                for it in range(6):
                    prate11=prate1+1
                    zKur,attKur,zKug,attKug=calcZ(dNw[k],dng[k],f*prate1,(1-f)*prate1,sdsu)
                    zKu1=10*np.log10(10**(0.1*zKur)+10**(0.1*zKug))
                    #print(f*prate1,(1-f)*prate1,zKur,zKug,zku_obs[k],zKu1)
                    #print(dng,f)
                    zKur2,attKur2,zKug2,attKug2=calcZ(dNw[k],dng[k],f*prate11,(1-f)*prate11,sdsu)
                    zKu2=10*np.log10(10**(0.1*zKur2)+10**(0.1*zKug2))
                    dzKu=(zKu2-(attKur2+attKug2)*dr-(zKu1-(attKur+attKug)*dr))/(prate11-prate1)
                    attKu=attKur+attKug
                    prate1+=0.75*(zku_obs[k]-(zKu1+piaKu-attKu*dr))*(dzKu)/(dzKu**2+0.001)
                    prate1=max(prate1,0.01)
                prate1*=1.1
                zKur,attKur,zKug,attKug=calcZ(dNw[k],dng[k],f*prate1,(1-f)*prate1,sdsu)
                zKu1=10*np.log10(10**(0.1*zKur)+10**(0.1*zKug))
                #print(f*prate1,(1-f)*prate1,zku_obs[k],'=',zKu1)
                precipRate[k]=prate1
                ztrue1D[k]=zKu1
                piaKu-=2*attKu*dr
                piaKu=max(0,piaKu)
            else:
                for it in range(2):
                    z_true=zku_obs[k]+max(0,piaKu-attKu*dr)
                    ibin=int((z_true-10*dNw[k]+12)/0.25)
                    ibin=max(0,ibin)
                    ibin=min(ibin,252)
                    precipRate[k]=sdsu.tablep2.snowrate[ibin]*10**dNw[k]
                    attKu=sdsu.tablep2.attkus[ibin]*10**dNw[k]

                piaKu-=2*attKu*dr
                piaKu=max(0,piaKu)
                ztrue1D[k]=z_true
        #piaKu+=(attKug+attKur)*dr
        #zKu[k]=10*np.log10(10**(0.1*zKur)+10**(0.1*zKug))-piaKu
        #piaKu+=(attKug+attKur)*dr
        ztrue1D[ztrue1D<0]=0
    return precipRate,piaKu,ztrue1D
