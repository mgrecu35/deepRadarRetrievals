import numpy as np

dmgCoeff=np.array([ 0.01462585, -0.26336104])
dmrCoeff=np.array([ 0.01348375, -0.24574921])

rrateCoeff=np.array([ 0.62968272, -1.63180766])
grateCoeff=np.array([ 0.65260312, -1.78462983])
#grateCOeff=np.array([ 0.70787995, -1.86986949])

from scipy.special import gamma as gam

def fmu(mu):
    return 6/4**4*(4+mu)**(mu+4)/gam(mu+4)


mu=2.0
f_mu=fmu(mu)
