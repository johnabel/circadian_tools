# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:19:48 2014

@author: john abel

v13: Noncompetitive inhibition with C1P,C2P, and also necessary activation by CREB

"""

import numpy  as np
import casadi as cs
import pdb
from jha_CommonFiles import modelbuilder as mb

EqCount = 31
ParamCount = 47
modelversion='deg_sync_v16'

y0in = np.array([ 0.09909523,  0.70371313,  0.2269922 ,  0.10408456,  0.00490967,
        0.86826377,  0.89688085,  0.06720938,  0.42133251,  0.00728958,
        0.47681956])


#Peter's Model Initial Values   
param_found = np.array([2.21515143e-01,   1.54421782e-01,   7.13499145e-02,
             2.33302244e-01,   1.80969725e-01,   2.64695797e-01,
             6.18556532e-01,   9.62626209e-01,   7.72231383e-03,
             1.88557482e+00,   3.47836153e+00,   2.50631434e-02,
             2.04947462e+00,   6.06189856e-01,   2.88824913e+00,
             5.17521933e-02,   5.175e-02     ,   5.34713225e-02,
             7.87365479e-02,   1.44504672e-03,   3.89020380e+00,
             2.21515143e-01,   1.54421782e-01,   7.13499145e-02])
             
             
param=[  2.25159896e-01,   1.90031674e-01,   6.70692205e-02,
         2.63970881e-01,   1.55918619e-01,   2.59850992e-01,
         7.81672489e-01,   1.23769111e+00,   7.95480246e-03,
         1.94345471e+00,   7.04967650e+00,   3.71990395e-02,
         2.23511192e+00,   4.56343016e-01,   4.22802939e+00,
         1.66443181e-02,   4.67850734e-02,   4.54878164e-02,
         2.67511407e-01,   2.06409164e-03,   4.07754219e+00,
         1.27824043e-01,   2.96422108e-01,   9.31597225e-02,
         3.00583908e-01,   1.58044144e-01,   1.84497550e-01,
         1.21730846e+00,   5.49724882e-01,   1.15067882e-01,
         7.33693360e-01,   1.09767551e-01,   6.64793809e-01,
         1.30952725e-01,   3.18395912e-01,   5.85189667e-02,
         6.90747660e-01,   1.00492158e+00,   9.95205130e-01,
         1.17750519e-01,   4.28259617e-01,   6.82074313e-01,
         4.86733215e-01,   1.71831427e+00,   3.28515848e-01,
         3.10683969e-01,   1.38006374e+01]           
"""
    paramset = cs.vertcat([vtp  , vtc1 , vtc2 , 
                           knp  , knc  , vtv  ,  
                           knv  , vdp  , vdc1 , 
                           
                           vdc2 , kdp  , kdc  , 
                           vdv  , kdv  , vdP  , 
                           kdP  , vdC1 , vdC2 , 
                           kdC  , vdCn , MC2n , 
                           
                           kdCn , vaCP , vdCP ,
                           ktlnp, vdVIP, kdVIP,
                           vgpcr, kgpcr, vdPKA,
                           kdPKA])
"""



def model():
    """ trying non-reversible nuclear entry and fitting knockouts. """
       #==================================================================
    #State variable definitions
    #==================================================================
    p    = cs.ssym("p")
    c1   = cs.ssym("c1")
    c2   = cs.ssym("c2")
    vip  = cs.ssym("vip")
    P    = cs.ssym("P")
    C1   = cs.ssym("C1")
    C2   = cs.ssym("C2")
    eVIP = cs.ssym("eVIP")
    C1P  = cs.ssym("C1P")
    C2P  = cs.ssym("C2P")
    CREB = cs.ssym("CREB")
    
    p_2    = cs.ssym("p_2")
    c1_2   = cs.ssym("c1_2")
    c2_2   = cs.ssym("c2_2")
    vip_2  = cs.ssym("vip_2")
    P_2    = cs.ssym("P_2")
    C1_2   = cs.ssym("C1_2")
    C2_2   = cs.ssym("C2_2")
    C1P_2  = cs.ssym("C1P_2")
    C2P_2  = cs.ssym("C2P_2")
    CREB_2 = cs.ssym("CREB_2")
    
    p_3    = cs.ssym("p_3")
    c1_3   = cs.ssym("c1_3")
    c2_3   = cs.ssym("c2_3")
    vip_3  = cs.ssym("vip_3")
    P_3    = cs.ssym("P_3")
    C1_3   = cs.ssym("C1_3")
    C2_3   = cs.ssym("C2_3")
    C1P_3  = cs.ssym("C1P_3")
    C2P_3  = cs.ssym("C2P_3")
    CREB_3 = cs.ssym("CREB_3")
    
    #for Casadi
    y = cs.vertcat([p, c1, c2, vip, P, C1, C2, eVIP, C1P, C2P, CREB,
                    p_2, c1_2, c2_2, vip_2, P_2, C1_2, C2_2, C1P_2, C2P_2, CREB_2,
                    p_3, c1_3, c2_3, vip_3, P_3, C1_3, C2_3, C1P_3, C2P_3, CREB_3])
    
    # Time Variable
    t = cs.ssym("t")
    
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    
    #mRNA
    vtpp    = cs.ssym("vtpp")         #transcription of per
    vtpr = cs.ssym("vtpr")
    vtc1p   = cs.ssym("vtc1p")        #transcription of cry1
    vtc1r   = cs.ssym("vtc1r")
    vtc2p   = cs.ssym("vtc2p")        #transcription of cry2
    vtc2r   = cs.ssym("vtc2r")
    knpp   = cs.ssym("knpp")         #transcription of per
    knpr   = cs.ssym("knpr")
    kncp    = cs.ssym("kncp")         #transcription of cry 1&2
    kncr   = cs.ssym("kncr")
    vtvp    = cs.ssym("vtvp")         #transcription of vip
    vtvr   = cs.ssym("vtvr")
    knvp    = cs.ssym("knvp")         #transcription of vip
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
    vdC2n   = cs.ssym("vdC2n")        #Cry2 degradation multiplier
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
    vgpcr = cs.ssym("vgpcr")
    kgpcr = cs.ssym("kgpcr")
    vdCREB= cs.ssym("vdCREB")
    kdCREB= cs.ssym("kdCREB")
    ktlnv = cs.ssym("ktlnv")
    vdpka = cs.ssym("vdpka")
    vgpka = cs.ssym("vgpka")
    kdpka = cs.ssym("kdpka")
    kgpka = cs.ssym("kgpka")
    kdc1=cs.ssym("kdc1")
    kdc2=cs.ssym("kdc2")
    kcouple = cs.ssym("kcouple")
    
    #for Casadi
    paramset = cs.vertcat([vtpr  , vtc1r , vtc2r  , knpr   , kncr   , 
                           vdp   , vdc1  , vdc2   , kdp    , kdc    ,
                           vdP   , kdP   , vdC1   , vdC2   , kdC    , 
                           vdC1n  , vdC2n  , kdCn   , vaCP   , vdCP   , ktlnp,
                        
                           vtpp  , vtc1p , vtc2p  , vtvp   , vtvr   , 
                           knpp  , kncp  , knvp   , knvr   , vdv    ,
                           kdv   , vdVIP , kdVIP  , vgpcr  , kgpcr  , 
                           vdCREB, kdCREB, ktlnv  , vdpka  , vgpka  , 
                           kdpka , kgpka, kdc1,kdc2,ktlnc,kcouple])
    #for optional Stochastic Simulation
                        
    
    #================================================================================
    # Model Equations
    #================================================================================
    
    ode = [[]]*EqCount

    # MRNA Species
    ode[0] = mb.prtranscription2(CREB,C1P,C2P,vtpr,vtpp,knpr)   - mb.michaelisMenten(p,vdp,kdp)
    ode[1] = mb.rtranscription(C1P,C2P,vtc1r,kncr,1)  - mb.michaelisMenten(c1,vdc1,kdc)
    ode[2] = mb.rtranscription(C1P,C2P,vtc2r,kncr,1)  - mb.michaelisMenten(c2,vdc2,kdc)
    ode[3] = mb.rtranscription(C1P,C2P,vtvr,knvr,1)   - mb.michaelisMenten(vip,vdv,kdv)
    
    # Free Proteins
    ode[4] = mb.translation(p,ktlnp)    - mb.michaelisMenten(P,vdP,kdP)        - mb.Complexing(vaCP,C1,P,vdCP,C1P)  - mb.Complexing(vaCP,C2,P,vdCP,C2P)
    ode[5] = mb.translation(c1,ktlnc)       - mb.michaelisMenten(C1,vdC1,kdC)      - mb.Complexing(vaCP,C1,P,vdCP,C1P)
    ode[6] = mb.translation(c2,ktlnc)       - mb.michaelisMenten(C2,vdC2,kdC)      - mb.Complexing(vaCP,C2,P,vdCP,C2P)
    ode[7] = mb.translation(vip,ktlnv) + mb.translation(vip_2,ktlnv) + mb.translation(vip_3,ktlnv)  -kdVIP*eVIP

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[8] = mb.Complexing(vaCP,C1,P,vdCP,C1P) -mb.sharedDegradation(C1P,C2P,vdC1n,kdCn) 
    ode[9] = mb.Complexing(vaCP,C2,P,vdCP,C2P) -mb.sharedDegradation(C2P,C1P,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[10] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB,vdCREB,kdCREB)
    
    ##
    ## cell 2
    ##
    ode[11] = mb.prtranscription2(CREB_2,C1P_2,C2P_2,vtpr,vtpp,knpr)   - mb.michaelisMenten(p_2,vdp,kdp)
    ode[12] = mb.rtranscription(C1P_2,C2P_2,vtc1r,kncr,1)  - mb.michaelisMenten(c1_2,vdc1,kdc)
    ode[13] = mb.rtranscription(C1P_2,C2P_2,vtc2r,kncr,1)  - mb.michaelisMenten(c2_2,vdc2,kdc)
    ode[14] = mb.rtranscription(C1P_2,C2P_2,vtvr,knvr,1)   - mb.michaelisMenten(vip_2,vdv,kdv)
    
    # Free Proteins
    ode[15] = mb.translation(p_2,ktlnp)    - mb.michaelisMenten(P_2,vdP,kdP)        - mb.Complexing(vaCP,C1_2,P_2,vdCP,C1P_2)  - mb.Complexing(vaCP,C2_2,P_2,vdCP,C2P_2)
    ode[16] = mb.translation(c1_2,ktlnc)       - mb.michaelisMenten(C1_2,vdC1,kdC)      - mb.Complexing(vaCP,C1_2,P_2,vdCP,C1P_2)
    ode[17] = mb.translation(c2_2,ktlnc)       - mb.michaelisMenten(C2_2,vdC2,kdC)      - mb.Complexing(vaCP,C2_2,P_2,vdCP,C2P_2)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[18] = mb.Complexing(vaCP,C1_2,P_2,vdCP,C1P_2) -mb.sharedDegradation(C1P_2,C2P_2,vdC1n,kdCn) 
    ode[19] = mb.Complexing(vaCP,C2_2,P_2,vdCP,C2P_2) -mb.sharedDegradation(C2P_2,C1P_2,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[20] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB_2,vdCREB,kdCREB)
    
    ##
    ## cell 3
    ##
    ode[21] = mb.prtranscription2(CREB_3,C1P_3,C2P_3,vtpr,vtpp,knpr)   - mb.michaelisMenten(p_3,vdp,kdp)
    ode[22] = mb.rtranscription(C1P_3,C2P_3,vtc1r,kncr,1)  - mb.michaelisMenten(c1_3,vdc1,kdc)
    ode[23] = mb.rtranscription(C1P_3,C2P_3,vtc2r,kncr,1)  - mb.michaelisMenten(c2_3,vdc2,kdc)
    ode[24] = mb.rtranscription(C1P_3,C2P_3,vtvr,knvr,1)   - mb.michaelisMenten(vip_3,vdv,kdv)
    
    # Free Proteins
    ode[25] = mb.translation(p_3,ktlnp)    - mb.michaelisMenten(P_3,vdP,kdP)        - mb.Complexing(vaCP,C1_3,P_3,vdCP,C1P_3)  - mb.Complexing(vaCP,C2_3,P_3,vdCP,C2P_3)
    ode[26] = mb.translation(c1_3,ktlnc)       - mb.michaelisMenten(C1_3,vdC1,kdC)      - mb.Complexing(vaCP,C1_3,P_3,vdCP,C1P_3)
    ode[27] = mb.translation(c2_3,ktlnc)       - mb.michaelisMenten(C2_3,vdC2,kdC)      - mb.Complexing(vaCP,C2_3,P_3,vdCP,C2P_3)
    
    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[28] = mb.Complexing(vaCP,C1_3,P_3,vdCP,C1P_3) -mb.sharedDegradation(C1P_3,C2P_3,vdC1n,kdCn) 
    ode[29] = mb.Complexing(vaCP,C2_3,P_3,vdCP,C2P_3) -mb.sharedDegradation(C2P_3,C1P_3,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[30] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB_3,vdCREB,kdCREB)
    
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","deg_sync")
    
    return fn
