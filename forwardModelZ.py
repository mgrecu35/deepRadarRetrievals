from bisectm import *
import numpy as np

def forward_model_ku(graup_rate,rain_rate,fint,dNw,dr,di,sdsu,iz):
    graup_rate[:]=graup_rate[:]*fint
    rain_rate[:]=rain_rate[:]*(1-fint)
    rain_rate[33:37]*=np.array([1.125,1.09,1.05,1.02])
    nz=graup_rate.shape[0]
    zS_coeff=np.array([1.34204631, 2.11016229])

    piaKu=0
    zKu=np.zeros((nz),float)
    graup_rate[33]*=1
    for k in range(nz):
        dng=-1+(iz-k)*0.04
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
        if rain_rate[k]/10**dNw[k]>1e-2:
            ibin=bisectm(sdsu.tablep2.rainrate[:289],289,rain_rate[k]/10**dNw[k])
            zKur=sdsu.tablep2.zkur[ibin]+10*dNw[k]
            attKur=sdsu.tablep2.attkur[ibin]*10**dNw[k]
        else:
            zKur=-0
            attKur=0
        if graup_rate[k]/10**dng>1e-3:
            ibin=bisectm(sdsu.tablep2.snowrate[:253],253,graup_rate[k]/10**dng)
            if graup_rate[k]/10**dng<333:
                zKug=sdsu.tablep2.zkus[ibin]+10*dng
            else:
                zKug=10*(zS_coeff[0]*np.log10(graup_rate[k]/10**dng)+\
                         zS_coeff[1]+dng)
            attKug=sdsu.tablep2.attkus[ibin]*10**dng
        else:
            zKug=-0
            attKug=0
        piaKu+=(attKug+attKur)*dr
        zKu[k]=10*np.log10(10**(0.1*zKur)+10**(0.1*zKug))-piaKu
        piaKu+=(attKug+attKur)*dr
    return zKu,piaKu+(attKug+attKur)*dr*2*di
