# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:19:48 2014

@author: john abel

v12 activation of per only by creb

"""

import numpy  as np
import casadi as cs
import pdb
from jha_CommonFiles import modelbuilder as mb

EqCount = 11
ParamCount = 47
modelversion='deg_sync_v12'

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
             
             
param=[  1.73564788e-01,   1.68271735e-01,   7.67949292e-02,
         2.25978583e-01,   2.51643997e-01,   3.58732149e-01,
         8.83156115e-01,   1.22166615e+00,   1.16706253e-02,
         2.41333615e+00,   4.41271352e+00,   2.04822625e-02,
         2.03988219e+00,   4.17489699e-01,   3.31961649e+00,
         3.83195563e-02,   1.53912105e-01,   2.83569232e-02,
         8.57270152e-02,   5.75703957e-04,   3.74820091e+00,
         1.20942284e-01,   2.01782050e-01,   1.01756316e-01,
         2.27815467e-01,   1.93374172e-01,   2.36627057e+00,
         1.08718849e+00,   3.13474638e-01,   5.17830544e-02,
         9.03830968e+00,   4.01147274e-01,   2.57114776e-01,
         3.55654955e+00,   3.84177683e-02,   1.54348631e-01,
         2.34991678e+00,   1.19338021e+01,   1.02419071e+02,
         3.57207890e-01,   4.00881292e-01,   3.12654084e-01,
         4.87995870e+00,   1.24113862e-01,   7.06827451e-01,
         4.46249930e-01,   112.75488733e+00]   
         
param_2 = [  1.76568860e-01,   1.88581618e-01,   7.30324027e-02,
         1.86154767e-01,   4.72469606e-02,   2.06071685e-01,
         8.08613027e-01,   6.57902620e-01,   4.55274905e-03,
         2.72826682e+00,   3.34831043e+00,   2.94563068e-02,
         1.75494943e+00,   7.84795347e-01,   2.12198440e+00,
         4.06179122e-02,   2.99165179e-01,   8.21228306e-02,
         1.12095987e-01,   2.39298633e-03,   3.13099116e+00,
         1.38297114e-01,   1.03900965e-01,   1.10820091e-05,
         5.31894383e-01,   2.32689001e-01,   1.66309604e+04,
         5.85422665e-01,   1.29522633e-01,   9.82489892e-02,
         7.26935408e-01,   6.67077877e-02,   4.50603115e-02,
         2.55806297e+00,   8.74213455e+02,   2.45427927e+00,
         7.26967256e-01,   7.83783004e-01,   1.86209038e+01,
         9.31368989e-02,   4.49569511e-01,   4.88112915e-01,
         3.60704010e-01,   6.51219141e-02,   2.64001560e-01,
         5.21660042e-01,   1.23360256e-01]
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
    ode[0] = mb.ptranscription(CREB,vtpp,  knpp,  1) + mb.rtranscription(C1P,C2P,vtpr,knpr,1)   - mb.michaelisMenten(p,vdp,kdp)
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
