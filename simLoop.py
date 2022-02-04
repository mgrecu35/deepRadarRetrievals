import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

def simLoop(f,zKuL,zKu_msL,xL,beta,maxHL,zKu_cL,sdsu,zka_3d,
            zka_3d_ms, zka_3d_true,\
            pRateL,attKuL,rwcL,swcL,f1L,f2L,ijL):
    R=287
    h=(250/2.+np.arange(76)*250)/1e3
    h1=(0.+np.arange(77)*250)/1e3
    fh=Dataset(f)
    z=fh['zh'][:]
    qr=fh["qr"][0,:,:,:]
    qg=fh["qg"][0,:,:,:]
    qs=fh["qs"][0,:,:,:]
    a=np.nonzero(qr[0,:,:]>-0.00015)
    print(a[0].shape[0],qr[0,:,:].max())
    T=fh["th"][0,:,:,:]
    prs=fh["prs"][0,:,:,:]
    T=T*(prs/1e5)**(0.286)
    rho=prs/(R*T)
    rwc=qr*rho*1e3
    gwc=(qg+qs)*rho*1e3
    qv=fh['qv'][0,:,:,:]
    wv=qv*rho
    
    zka_3d=zka_3d*0.0-99
    zka_3d_ms=zka_3d_ms*0.0-99
    zka_3d_true=zka_3d_true*0.0-99
    #plt.figure(figsize=(8,11))
    nz,nx,ny=rwc.shape
    for i1,i2 in zip(a[0],a[1]):
        zKa_1D=[]
        att_1D=[]
        pwc_1D=[]
        prate_1D=[]
       
        piaKa=0
        dr=0.250
        dn1=np.zeros((76),float)-0.5
        rwc1=np.interp(h,z[:],rwc[:,i1,i2])*1.2
        swc1=np.interp(h,z[:],gwc[:,i1,i2])*1.2
        temp=np.interp(h,z[:],T[:,i1,i2])
        temp1=np.interp(h1,z[:],T[:,i1,i2])
        press=np.interp(h,z[:],prs[:,i1,i2])
        wv1=np.interp(h,z[:],wv[:,i1,i2])
        a=np.nonzero(swc1>0.01)
        if len(a[0])>5:
            ht1=h[a[0][-1]]
            ht2=ht1+np.random.random()*5
            ht2=min(h[-1],ht2)
            hb1=h[a[0][0]]
            dh=(ht2-ht1)/len(a[0]-1)*np.arange(len(a[0]))
            hint=h[a[0]]+dh
            ntop=min(int(ht2/dr)+1,75)
            swc11=np.interp(h[a[0][0]:ntop],h[a[0]]+dh,swc1[a[0]])
            for k in range(a[0][0],ntop):
                swc1[k]=max(swc1[k],swc11[k-a[0][0]])
        f1=np.exp(np.random.randn(76)*1)
        f2=np.exp(np.random.randn(76)*1)
        f1=gaussian_filter(f1, sigma=3)
        f2=gaussian_filter(f2, sigma=3)
        rwc1=f1*rwc1
        swc1=f2*swc1
        zka_m ,zka_t, attka, piaka, \
            kext,salb,asym,kext_,salb_,asym_,pRate\
            =sdsu.reflectivity_ku(rwc1,swc1,wv1,dn1,temp,press,dr)
        dr=0.25
        noms=0
        alt=400.
        freq=13.8
        nonorm=0
        theta=0.35
        
        if zka_m.max()>40 and i1>0 and i1<nx-1 and i2>0 and i2<ny-1:
            ijL.append([i1,i2])
            zms = sdsu.multiscatterf(kext[::-1],salb[::-1],asym[::-1],\
                                 zka_t[::-1],dr,noms,alt,\
                                 theta,freq,nonorm)
            
            pRateL.append(pRate)
            zKuL.append(zka_m)
            f1L.append(f1)
            f2L.append(f2)
            zKu_msL.append(zms[::-1])
            attKuL.append(attka)
            rwcL.append(rwc1)
            swcL.append(swc1)
            zKu_cL.append(zka_t)
            ind=np.argmax(zka_m[12:24])
            maxHL.append([zka_m[12+ind],ind,zka_t[12+ind]-zka_m[12+ind]])
            if zka_t[0]>40  and zka_t[10]>140:
                q=0.2*np.log(10)
                zKum=zka_m[::-1]
                zeta=q*beta*alpha_1*10**(0.1*zKum*beta)*dr
                zetamax=0.9998621204047031
                if zeta.cumsum()[-1]>zetamax:
                    eps=0.9999*zetamax/zeta.cumsum()[-1]
                    zeta=eps*zeta
                else:
                    eps=1.0
                corrc=eps*zeta.cumsum()
                zc=zKum-10/beta*np.log10(1-corrc)
                stop
            #x1=list(swc1[0:64])
            #x2=list(rwc1[0])
        zka_3d[:,i1,i2]=zka_m
        #zms = sdsu.multiscatterf(kext[::-1],salb[::-1],asym[::-1],\
        #                         zka_t[::-1],dr,noms,alt,\
        #                         theta,freq,nonorm)
        #kext1=np.zeros((75),float)
        #salb1=np.zeros((75),float)
        #asym1=np.zeros((75),float)
        #for k in range(75):
        #    kext1[k]=kext[2*k:2*k+2].mean()
        #    salb1[k]=salb[2*k:2*k+2].mean()
        #    asym1[k]=asym[2*k:2*k+2].mean()

        #kext1_=np.zeros((75),float)
        #salb1_=np.zeros((75),float)
        #asym1_=np.zeros((75),float)
        #for k in range(75):
        #    kext1_[k]=kext_[2*k:2*k+2].mean()
        #    salb1_[k]=salb_[2*k:2*k+2].mean()
        #    asym1_[k]=asym_[2*k:2*k+2].mean()
        #emis=0.85+0.1*np.random.rand()
        #ebar=emis
        #lambert=1
        #salb1[salb1>0.99]=0.99
        #tb = sdsu.radtran(umu,temp1[0],temp1,h1/1000.,kext1,salb1,asym1,\
        #                  fisot,emis,ebar,lambert)
