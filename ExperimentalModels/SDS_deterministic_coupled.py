# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:19:48 2014

@author: john abel

"""

import numpy  as np
import casadi as cs
import stochkit_resources as stk
import modelbuilder as mb
import random
import matplotlib.pyplot as plt


EqCount = 22
ParamCount = 35
modelversion='SDSdet'

cellcount=100

period = 23.7000

vol=400

randomy0 = False

freq = 4
amp = 0#0.10

#better for stoch
y0in=np.array([ 0.27639502,  1.49578759,  0.23951501,  0.10907372,  0.00704751,
        0.963337  ,  0.59516789,  0.71254298,  0.28286947,  0.02583619,
        0.23034218,  0.27639502,  1.49578759,  0.23951501,  0.10907372,  0.00704751,
        0.963337  ,  0.59516789,  0.71254298,  0.28286947,  0.02583619,
        0.23034218])

        
param = [0.41461483697868157, 0.3499288858096469, 0.12350287248265006,
         0.263970881        , 0.155918619       , 0.47849585383903664,
         1.4393904836297102 , 2.2791141180959396, 0.00795480246      , 
         1.94345471         , 12.981443519586378, 0.0371990395       , 
         4.115788738594497  , 0.8403209832056411, 4.22802939         , 
         0.03064924686078563, 0.0861511571349129, 0.0454878164       , 
         0.49260192589200136, 0.0038008678900245, 7.508484061391405  , 
         
         0.23537826092440986, 0.2910262802750418, 0.115067882        , 
         1.351040563852193  , 0.109767551       , 1.2241672768536533 , 
         0.7234184731696952 , 1.2719593210520304, 1.00492158         , 
         5.497783842491617  , 0.7886075382047947, 1.4601996449999999 , 
         0.5721009178243037 , 10
         ]

#C2 KO
#param[2]=0
#C1 KO
#param[1]=0
#VIP KO
#param[22]=0

def ODEmodel():
    #==================================================================
    #State variable definitions
    #==================================================================
    p_0    = cs.ssym("p_0")
    c1_0   = cs.ssym("c1_0")
    c2_0   = cs.ssym("c2_0")
    vip_0  = cs.ssym("vip_0")
    P_0    = cs.ssym("P_0")
    C1_0   = cs.ssym("C1_0")
    C2_0   = cs.ssym("C2_0")
    eVIP_0 = cs.ssym("eVIP_0")
    C1P_0  = cs.ssym("C1P_0")
    C2P_0  = cs.ssym("C2P_0")
    CREB_0 = cs.ssym("CREB_0")
    
    p_1    = cs.ssym("p_1")
    c1_1   = cs.ssym("c1_1")
    c2_1   = cs.ssym("c2_1")
    vip_1  = cs.ssym("vip_1")
    P_1    = cs.ssym("P_1")
    C1_1   = cs.ssym("C1_1")
    C2_1   = cs.ssym("C2_1")
    eVIP_1 = cs.ssym("eVIP_1")
    C1P_1  = cs.ssym("C1P_1")
    C2P_1  = cs.ssym("C2P_1")
    CREB_1 = cs.ssym("CREB_1")
    
    #for Casadi
    y = cs.vertcat([p_0, c1_0, c2_0, vip_0, P_0, C1_0, C2_0, eVIP_0, C1P_0, C2P_0, CREB_0,
                    p_1, c1_1, c2_1, vip_1, P_1, C1_1, C2_1, eVIP_1, C1P_1, C2P_1, CREB_1])
    
    # Time Variable
    t = cs.ssym("t")
    
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    
    #mRNA
    vtpp    = cs.ssym("vtpp")
    vtpr = cs.ssym("vtpr")
    vtc1r   = cs.ssym("vtc1r")
    vtc2r   = cs.ssym("vtc2r")
    knpr   = cs.ssym("knpr")
    kncr   = cs.ssym("kncr")
    vtvr   = cs.ssym("vtvr")
    knvr   = cs.ssym("knvr")
    
    vdp    = cs.ssym("vdp")         #degradation of per
    vdc1   = cs.ssym("vdc1")        #degradation of cry1
    vdc2   = cs.ssym("vdc2")        #degradation of cry2
    kdp    = cs.ssym("kdp")         #degradation of per
    kdc    = cs.ssym("kdc")         #degradation of cry1 and 2
    vdv    = cs.ssym("vdv")         #degradation of vip
    kdv    = cs.ssym("kdv")         #degradation of vip
    
    #Proteins
    vdP    = cs.ssym("vdP")         #degradation of Per
    vdC1   = cs.ssym("vdC1")        #degradation of Cry1
    vdC2   = cs.ssym("vdC2")        #degradation of Cry2
    vdC1n   = cs.ssym("vdC1n")        #degradation of Cry1-Per or Cry2-Per complex into nothing (nuclear)
    vdC2n   = cs.ssym("vdC2n")
    kdP    = cs.ssym("kdP")         #degradation of Per
    kdC    = cs.ssym("kdC")         #degradation of Cry1 & Cry2
    kdCn   = cs.ssym("kdCn")        #degradation of Cry1-Per or Cry2-Per complex into nothing (nuclear)
    vaCP   = cs.ssym("vaCP")        #formation of Cry1-Per or Cry2-Per complex
    vdCP   = cs.ssym('vdCP')        #degradation of Cry1-Per or Cry2-Per into components in ctyoplasm
    ktlnp  = cs.ssym('ktlnp')       #translation of per
    ktlnc  = cs.ssym('ktlnc')
    
    #Signalling
    vdVIP = cs.ssym("vdVIP")
    kdVIP = cs.ssym("kdVIP")
    vdCREB= cs.ssym("vdCREB")
    kdCREB= cs.ssym("kdCREB")
    ktlnv = cs.ssym("ktlnv")
    vgpka = cs.ssym("vgpka")
    kgpka = cs.ssym("kgpka")
    kcouple = cs.ssym("kcouple")    #necessary for stochastic model, does not need to be fit

    paramset = cs.vertcat([vtpr  , vtc1r , vtc2r  , knpr   , kncr   , 
                           vdp   , vdc1  , vdc2   , kdp    , kdc    ,
                           vdP   , kdP   , vdC1   , vdC2   , kdC    , 
                           vdC1n , vdC2n , kdCn   , vaCP   , vdCP   , ktlnp,
                        
                           vtpp  , vtvr   , 
                           knvr   , vdv    ,
                           kdv   , vdVIP , kdVIP  , 
                           vdCREB, kdCREB, ktlnv  , vgpka  , 
                           kgpka, ktlnc,kcouple])
                        
    
    #===================================================================
    # Model Equations
    #===================================================================
    
    ode = [[]]*EqCount
    
    # CELL 0    
    
    # MRNA Species
    ode[0] = mb.prtranscription2(CREB_0,(C1P_0)+amp*np.sin(freq*t),C2P_0,vtpr,vtpp,knpr)   - mb.michaelisMenten(p_0,vdp,kdp)
    ode[1] = mb.rtranscription((C1P_0)+amp*np.sin(freq*t),C2P_0,vtc1r,kncr,1)  - mb.michaelisMenten(c1_0,vdc1,kdc)
    ode[2] = mb.rtranscription((C1P_0)+amp*np.sin(freq*t),C2P_0,vtc2r,kncr,1)  - mb.michaelisMenten(c2_0,vdc2,kdc)
    ode[3] = mb.rtranscription((C1P_0)+amp*np.sin(freq*t),C2P_0,vtvr,knvr,1)   - mb.michaelisMenten(vip_0,vdv,kdv)
    
    # Free Proteins
    ode[4] = mb.translation(p_0,ktlnp)    - mb.michaelisMenten(P_0,vdP,kdP)        - mb.Complexing(vaCP,C1_0,P_0,vdCP,(C1P_0)+amp*np.sin(freq*t))  - mb.Complexing(vaCP,C2_0,P_0,vdCP,C2P_0)
    ode[5] = mb.translation(c1_0,ktlnc)       - mb.michaelisMenten(C1_0,vdC1,kdC)      - mb.Complexing(vaCP,C1_0,P_0,vdCP,(C1P_0)+amp*np.sin(freq*t))
    ode[6] = mb.translation(c2_0,ktlnc)       - mb.michaelisMenten(C2_0,vdC2,kdC)      - mb.Complexing(vaCP,C2_0,P_0,vdCP,C2P_0)
    ode[7] = mb.translation(vip_1,ktlnv)  + mb.lineardeg(kdVIP,eVIP_0)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[8] = mb.Complexing(vaCP,C1_0,P_0,vdCP,(C1P_0)+amp*np.sin(freq*t)) -mb.sharedDegradation((C1P_0)+amp*np.sin(freq*t),C2P_0,vdC1n,kdCn) 
    ode[9] = mb.Complexing(vaCP,C2_0,P_0,vdCP,C2P_0) -mb.sharedDegradation(C2P_0,((C1P_0)+amp*np.sin(freq*t))+amp*np.sin(freq*t),vdC2n,kdCn)

    #Sgnaling Pathway
    ode[10] = mb.ptranscription(eVIP_0,vgpka,kgpka,1) - mb.michaelisMenten(CREB_0,vdCREB,kdCREB)
    

    # CELL 1    
    
    # MRNA Species
    ode[11] = mb.prtranscription2(CREB_1,C1P_1,C2P_1,vtpr,vtpp,knpr)   - mb.michaelisMenten(p_1,vdp,kdp)
    ode[12] = mb.rtranscription(C1P_1,C2P_1,vtc1r,kncr,1)  - mb.michaelisMenten(c1_1,vdc1,kdc)
    ode[13] = mb.rtranscription(C1P_1,C2P_1,vtc2r,kncr,1)  - mb.michaelisMenten(c2_1,vdc2,kdc)
    ode[14] = mb.rtranscription(C1P_1,C2P_1,vtvr,knvr,1)   - mb.michaelisMenten(vip_1,vdv,kdv)
    
    # Free Proteins
    ode[15] = mb.translation(p_1,ktlnp)    - mb.michaelisMenten(P_1,vdP,kdP)        - mb.Complexing(vaCP,C1_1,P_1,vdCP,C1P_1)  - mb.Complexing(vaCP,C2_1,P_1,vdCP,C2P_1)
    ode[16] = mb.translation(c1_1,ktlnc)       - mb.michaelisMenten(C1_1,vdC1,kdC)      - mb.Complexing(vaCP,C1_1,P_1,vdCP,C1P_1)
    ode[17] = mb.translation(c2_1,ktlnc)       - mb.michaelisMenten(C2_1,vdC2,kdC)      - mb.Complexing(vaCP,C2_1,P_1,vdCP,C2P_1)
    ode[18] = mb.translation(vip_0,ktlnv)  + mb.lineardeg(kdVIP,eVIP_1)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[19] = mb.Complexing(vaCP,C1_1,P_1,vdCP,C1P_1) -mb.sharedDegradation(C1P_1,C2P_1,vdC1n,kdCn) 
    ode[20] = mb.Complexing(vaCP,C2_1,P_1,vdCP,C2P_1) -mb.sharedDegradation(C2P_1,C1P_1,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[21] = mb.ptranscription(eVIP_1,vgpka,kgpka,1) - mb.michaelisMenten(CREB_1,vdCREB,kdCREB)    
    
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","3state")
    
    return fn
    


if __name__ == "__main__":
    
    
    import circadiantoolbox as ctb
    import PlotOptions as plo
    from numpy import fft
    import pdb
    
    #param[1]=0
    #param[2]=0    
    
    # create deterministic circadian object
    coupled_odes = ctb.CircEval(ODEmodel(), param, y0in)
    #coupled_odes.burnTransient_sim()
    coupled_odes.intODEs_sim(200)
    
    #Fixing signal to show perturbation
    coupled_odes.sol[:,8] = coupled_odes.sol[:,8]+amp*np.sin(freq*coupled_odes.ts)
    
    plo.PlotOptions()
    """
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,8],label='C1P 1')
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,3], label='vip 1')
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,18], label = 'VIP 2')
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,21], label = 'CREB 2')
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,11], label = 'per 2')
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,0], label = 'per 1')
    plt.legend(loc=1)
    # now doing the frequency analysis
    """
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,2],label='p1')
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,1],label='cry1')
    plt.plot(coupled_odes.ts, coupled_odes.sol[:,7],label='VIP')
    
    pdb.set_trace()
    # fft setup data
    tf=200
    step=2.00020002E-02
    Fs = 1/step # sampling freq
    t = np.arange(0, tf, step=step)
    
    #freq domain
    connections=np.zeros((6,1))
    #PC0, vip0, VIP1, CREB1, p1
    n = len(t)
    df = Fs/n
    fft_freqs = np.arange(-Fs/2,Fs/2,df)
    
    # find pure signal
    fx2 = np.sin(freq*t)
    Fk2 = fft.fftshift(fft.fft(fft.fftshift(fx2)))
    freq_index = np.argmax(np.abs(Fk2))
    
    inds = [8, 3, 18, 21, 11,0]
    for out in range(6):
        
        fx = coupled_odes.sol[:,inds[out]]
        Fk = fft.fftshift(fft.fft(fft.fftshift(fx)))
        
        #plt.plot(fft_freqs,np.abs(Fk), label=str(out))
        connections[out] = np.abs(Fk[freq_index])**2
    
    
    plt.plot



