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

EqCount = 11
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
             
             
param=[  2.52842366e-01,   1.59939424e-01,   3.90985236e-02,
         2.50626476e-01,   2.53538488e-01,   2.22689371e-01,
         7.19685658e-01,   5.69684279e-01,   9.79077757e-03,
         2.74827878e+00,   9.56883828e+00,   2.32977513e-02,
         2.85348494e+00,   6.49044199e-01,   3.15294401e+00,
         3.43566880e-02,   6.25789991e-02,   5.61307057e-02,
         4.65220038e-01,   9.80493785e-04,   2.29318424e+00,
         2.10973657e-01,   1.09639644e-01,   4.52018123e-02,
         2.94732811e+00,   3.31500641e-01,   2.19167273e-01,
         4.48314137e-01,   1.63811026e-01,   1.43833697e-01,
         2.28382024e+00,   1.23931082e-01,   1.20572832e+00,
         2.31809092e-01,   1.73027409e+00,   1.37723285e+01,
         5.43839238e-01,   3.20904613e-01,   2.37878512e+00,
         5.80498047e-01,   4.65733163e-01,   6.35753415e-01,
         5.25003214e+00,   5.12751019e-01,   9.19325486e-02,
         4.13887793e-01,   1.14550605e+00]           
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
    
    #for Casadi
    y = cs.vertcat([p, c1, c2, vip, P, C1, C2, eVIP, C1P, C2P, CREB])
    
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
    ode[7] = mb.translation(vip,ktlnv)  + mb.lineardeg(kdVIP,eVIP)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[8] = mb.Complexing(vaCP,C1,P,vdCP,C1P) -mb.sharedDegradation(C1P,C2P,vdC1n,kdCn) 
    ode[9] = mb.Complexing(vaCP,C2,P,vdCP,C2P) -mb.sharedDegradation(C2P,C1P,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[10] = mb.ptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB,vdCREB,kdCREB)

    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","deg_sync")
    
    return fn
