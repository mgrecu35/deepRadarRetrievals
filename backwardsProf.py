import numpy as np
from bisectm import *
zS_coeff=np.array([1.34204631, 2.11016229])
zS_coeff=np.array([13.42841596, 21.09608922])
zS_coeff=np.array([ 9.9200479 , 26.23596085])
zS_coeff=np.array([ 9.29199571, 27.60262757])
zR_coeff=np.array([16.41039975, 21.23238237]) #np.polyfit(np.log10(sdsu.tablep2.rainrate[200:282]),sdsu.tablep2.zkur[200:282],1)
rainRateCoeff=np.array([ 0.06093458, -1.29372223])

def calcZ(dNw,dng,rain_rate,graup_rate,sdsu):
    if rain_rate/10**dNw>1e-2:
        ibin=bisectm(sdsu.tablep2.rainrate[:289],289,rain_rate/10**dNw)
        zKur=sdsu.tablep2.zkur[ibin]+10*dNw
        if ibin==288:
            zKur=(zR_coeff[0]*np.log10(rain_rate/10**dNw)+\
                  zR_coeff[1]+10*dNw)
        attKur=sdsu.tablep2.attkur[ibin]*10**dNw
    else:
        zKur=-0
        attKur=0
    if graup_rate/10**dng>1e-3:
        ibin=bisectm(sdsu.tablep2.snowrate[:253],253,graup_rate/10**dng)
        if ibin<251:
                zKug=sdsu.tablep2.zkus[ibin]+10*dng
        else:
            zKug=(zS_coeff[0]*np.log10(graup_rate/10**dng)+\
                  zS_coeff[1]+10*dng)
        attKug=sdsu.tablep2.attkus[ibin]*10**dng
    else:
        zKug=-0
        attKug=0
    return zKur,attKur,zKug,attKug

from scipy.ndimage import gaussian_filter

#snowRateCoeff=np.polyfit(sdsu.tablep2.zkus[:252],np.log10(sdsu.tablep2.snowrate[:252]),1)
#attSnowCoeff=np.polyfit(sdsu.tablep2.zkus[:252],np.log10(sdsu.tablep2.attkus[:252]),1)
snowRateCoeff=np.array([ 0.07410346, -1.5639222 ])
snowRateCoeff=np.array([ 0.10070804, -2.64039573])
attSnowCoeff=np.array([ 0.09691713, -5.08300025])
def backwards_kuret(dr,sdsu,iz,zku_obs,piaKu):
    nz=zku_obs.shape[0]
   
    dn_coeff=np.array([-0.01489413,  0.91568719])

    zKu=np.zeros((nz),float)
    precipRate=np.zeros((nz),float)
    ztrue1D=np.zeros((nz),float)
    attKu1D=np.zeros((nz),float)
    dNw=np.zeros((nz),float)
    dnp=np.random.randn(47)
    dnp=gaussian_filter(dnp,sigma=4)*1.5


    for k in range(nz-1,-1,-1):
        dng=-.75+(iz-k)*0.04
        #if k==33:
        #    dng-=0.3
        #if k==32:
        #    dng-=0.2
        #if k==31:
        #    dng-=0.2
        #if k==30:
        #    dng-=0.1
        #if k==29:
        #    dng-=0.0
            
        
        attKu=0
        dng+=dnp[k]
        if k>33:
            for it in range(2):
                z_true=zku_obs[k]+piaKu-attKu*dr
                dNw[k]=np.polyval(dn_coeff,z_true)+dnp[k]-0.15*(k-32)/16.
                          
                ibin=int((z_true-10*dNw[k]+12)/0.25)
                ibin=max(0,ibin)
                ibin=min(ibin,288)
                #print(piaKu,zku_obs[k],z_true,dNw[k],ibin)
                precipRate[k]=sdsu.tablep2.rainrate[ibin]*10**dNw[k]
                attKu=sdsu.tablep2.attkur[ibin]*10**dNw[k]
            ztrue1D[k]=z_true
            piaKu-=2*attKu*dr
            attKu1D[k]=attKu*dr
            piaKu=max(0,piaKu)
        else:
            if k>25:
                prate1=precipRate[k+1]
              
                f=(k-26)/8
                #if f<0.9:
                #    f=0
                f=f
                dNw[:34]=dNw[34]
                for it in range(7):
                    prate11=prate1+1
                    zKur,attKur,zKug,attKug=calcZ(dNw[k],dng,f*prate1,(1-f)*prate1,sdsu)
                    zKu1=10*np.log10(10**(0.1*zKur)+10**(0.1*zKug))
                    #print(f*prate1,(1-f)*prate1,zKur,zKug,zku_obs[k],zKu1)
                    #print(dng,f)
                    zKur2,attKur2,zKug2,attKug2=calcZ(dNw[k],dng,f*prate11,(1-f)*prate11,sdsu)
                    zKu2=10*np.log10(10**(0.1*zKur2)+10**(0.1*zKug2))
                    dzKu=(zKu2-(attKur2+attKug2)*dr-(zKu1-(attKur+attKug)*dr))/(prate11-prate1)
                    attKu=attKur+attKug      
                    prate1+=0.75*(zku_obs[k]-(zKu1-(piaKu-attKu*dr)))*(dzKu)/(dzKu**2+0.0001)
                    prate1=max(prate1,0.01)
                
                prate1*=1.0
                zKur,attKur,zKug,attKug=calcZ(dNw[k],dng,f*prate1,(1-f)*prate1,sdsu)
                zKu1=10*np.log10(10**(0.1*zKur)+10**(0.1*zKug))
                #print(f*prate1,(1-f)*prate1,zku_obs[k],'=',zKu1)
                precipRate[k]=prate1
                attKu=attKur+attKug
                #if abs(zku_obs[k]-(zKu1+piaKu-attKu*dr))>2:
                #    print(prate1,dng,dNw[k],f,zku_obs[k],(zKu1-(piaKu-attKu*dr)))
                attKu1D[k]=attKu*dr
                ztrue1D[k]=zKu1
                piaKu-=2*attKu*dr
                piaKu=max(0,piaKu)
            else:
                attKu=0
                for it in range(2):
                    z_true=zku_obs[k]+max(0,piaKu-attKu*dr)
                    ibin=bisectm(sdsu.tablep2.zkus[:253],253,z_true-10*dng)
                    ibin=max(0,ibin)
                    ibin=min(ibin,252)
                    precipRate[k]=sdsu.tablep2.snowrate[ibin]*10**dng
                    attKu=sdsu.tablep2.attkus[ibin]*10**dng
                    if ibin==252:
                        precipRate[k]=10**np.polyval(snowRateCoeff,z_true-10*dng)*10**dng
                attKu1D[k]=attKu*dr
                if zku_obs[k]>40:
                    zKur,attKur,zKug,attKug=calcZ(dNw[k],dng,0.0,precipRate[k],sdsu)
                    if abs(z_true-zKug)>2:
                        print(precipRate[k],dng,z_true,zKug,ibin)
                piaKu-=2*attKu*dr
                piaKu=max(0,piaKu)
                ztrue1D[k]=z_true
        #piaKu+=(attKug+attKur)*dr
        #zKu[k]=10*np.log10(10**(0.1*zKur)+10**(0.1*zKug))-piaKu
        #piaKu+=(attKug+attKur)*dr
    ztrue1D[ztrue1D<0]=0

    piaKuF=0
    zku_sim=np.zeros((nz),float)

    for k in range(nz):
        piaKuF+=attKu1D[k]
        zku_sim[k]=ztrue1D[k]-piaKuF
        piaKuF+=attKu1D[k]
    #print(piaKuF,attKu1D[k])
    return precipRate,piaKu,ztrue1D,zku_sim,piaKu



def forward_kuret(dr,sdsu,zku_obs):
    nz=zku_obs.shape[0]
   
    dn_coeff=np.array([-0.01489413,  0.91568719])

    zKu=np.zeros((nz),float)
    precipRate=np.zeros((nz),float)
    ztrue1D=np.zeros((nz),float)
    attKu1D=np.zeros((nz),float)
    dNw=np.zeros((nz),float)
    dnp=np.random.randn(47)
    dnp=gaussian_filter(dnp,sigma=4)*1.5

    piaKu=0
    f=np.interp(range(47),[0,28,34,36,47],[1.2,1.2,1.,1.0,0.975])
    for k in range(0,28):
        if zku_obs[k]>30:
            dng=-1+0.1*(40-zku_obs[k])
        else:
            dng=0
        attKu=0
        attKu=0
        for it in range(2):
            z_true=zku_obs[k]+max(0,piaKu+attKu*dr)
            ibin=bisectm(sdsu.tablep2.zkus[:253],253,z_true-10*dng)
            ibin=max(0,ibin)
            ibin=min(ibin,252)
            precipRate[k]=sdsu.tablep2.snowrate[ibin]*10**dng
            attKu=sdsu.tablep2.attkus[ibin]*10**dng
            if ibin==252:
                precipRate[k]=10**np.polyval(snowRateCoeff,z_true-10*dng)*10**dng
                attKu=10**np.polyval(attSnowCoeff,z_true-10*dng)*10**dng
        piaKu+=2*attKu*dr
    for k in range(28,47):
        r1=np.random.randn()
        if r1>=0:
            precipRate[k]=precipRate[k-1]*f[k]+0.2*precipRate[27]
        else:
            precipRate[k]=precipRate[k-1]-0.2*precipRate[27]
    return precipRate,piaKu
