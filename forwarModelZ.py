from bisectm import *
import numpy as np

def forward_model_ku(graup_rate,rain_rate,fint,dNw,dr,di):
    graup_rate=graup_rate*fint
    rain_rate=rain_rate*(1-fint)
    nz=graup_rate.shape
    dng=-0.75
    piaKu=0
    zKu=np.zeros((nz),float)
    for k in range(nz):
        if rain_rate[k]/10**dNw[k]>1e-2:
            ibin=bisectm(sdsu.tablep2.rainrate[:289],289,rain_rate[k]/10**dN[k])
            zKur=sdsu.tablep2.zkar[ibin]+10*dN[k]*10**dN[k]
            attKur=sdsu.tablep2.attkar[ibin]
        else:
            zKur=-0
            attKur=0
        if graup_rate[k]/10**dng:
            ibin=bisectm(sdsu.tablep2.snowrate[:253],253,graup_rate[k]/10**dng)
            zKug=sdsu.tablep2.zkag[ibin]+10*dng
            attKug=sdsu.tablep2.attkag[ibin]*10**dng
        else:
            zKug=-0
            attKug=0
        piaKu+=(attKug+attKur)*dr
        zKu[k]=10*np.log10(10**(0.1*zKur)+10**(0.1*zKug))-piaKu
        piaKu+=(attKug+attKur)*dr
    return zKu,piaKu+(attKug+attKur)*dr*2*di
