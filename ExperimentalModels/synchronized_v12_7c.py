# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:19:48 2014

@author: john abel

5.0: repression and promotion terms combined
5.1: repression and promotion terms separate

"""

import numpy  as np
import casadi as cs
import pdb

EqCount = 71
ParamCount = 46
modelversion='syn10'

y0in = np.array([ 0.09909523,  0.70371313,  0.2269922 ,  0.10408456,  0.00490967,
        0.86826377,  0.89688085,  0.06720938,  0.42133251,  0.00728958,
        0.47681956])
period = 26.38

param = [  1.73564788e-01,   1.68271735e-01,   7.67949292e-02,
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
         
param_2=[  1.76568860e-01,   1.88581618e-01,   7.30324027e-02,
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
       
#Peter's Model Initial Values   
param_found = np.array([2.21515143e-01,   1.54421782e-01,   7.13499145e-02,
             2.33302244e-01,   1.80969725e-01,   2.64695797e-01,
             6.18556532e-01,   9.62626209e-01,   7.72231383e-03,
             1.88557482e+00,   3.47836153e+00,   2.50631434e-02,
             2.04947462e+00,   6.06189856e-01,   2.88824913e+00,
             5.17521933e-02,   3.08882883e+00,   5.34713225e-02,
             7.87365479e-02,   1.44504672e-03,   3.89020380e+00,
             2.21515143e-01,   1.54421782e-01,   7.13499145e-02,
             2.21515143e-01,   1.54421782e-01])
             
"""
period = 23.69999994


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
    
    p_4    = cs.ssym("p_4")
    c1_4   = cs.ssym("c1_4")
    c2_4   = cs.ssym("c2_4")
    vip_4  = cs.ssym("vip_4")
    P_4    = cs.ssym("P_4")
    C1_4   = cs.ssym("C1_4")
    C2_4   = cs.ssym("C2_4")
    C1P_4  = cs.ssym("C1P_4")
    C2P_4  = cs.ssym("C2P_4")
    CREB_4 = cs.ssym("CREB_4")
    
    p_5    = cs.ssym("p_5")
    c1_5   = cs.ssym("c1_5")
    c2_5   = cs.ssym("c2_5")
    vip_5  = cs.ssym("vip_5")
    P_5    = cs.ssym("P_5")
    C1_5   = cs.ssym("C1_5")
    C2_5   = cs.ssym("C2_5")
    C1P_5  = cs.ssym("C1P_5")
    C2P_5  = cs.ssym("C2P_5")
    CREB_5 = cs.ssym("CREB_5")
    
    p_6    = cs.ssym("p_6")
    c1_6   = cs.ssym("c1_6")
    c2_6   = cs.ssym("c2_6")
    vip_6  = cs.ssym("vip_6")
    P_6    = cs.ssym("P_6")
    C1_6   = cs.ssym("C1_6")
    C2_6   = cs.ssym("C2_6")
    C1P_6  = cs.ssym("C1P_6")
    C2P_6  = cs.ssym("C2P_6")
    CREB_6 = cs.ssym("CREB_6")
    
    p_7    = cs.ssym("p_7")
    c1_7   = cs.ssym("c1_7")
    c2_7   = cs.ssym("c2_7")
    vip_7  = cs.ssym("vip_7")
    P_7    = cs.ssym("P_7")
    C1_7   = cs.ssym("C1_7")
    C2_7   = cs.ssym("C2_7")
    C1P_7  = cs.ssym("C1P_7")
    C2P_7  = cs.ssym("C2P_7")
    CREB_7 = cs.ssym("CREB_7")

    
    #for Casadi
    y = cs.vertcat([p, c1, c2, vip, P, C1, C2, eVIP, C1P, C2P, CREB,
                    p_2, c1_2, c2_2, vip_2, P_2, C1_2, C2_2, C1P_2, C2P_2, CREB_2,
                    p_3, c1_3, c2_3, vip_3, P_3, C1_3, C2_3, C1P_3, C2P_3, CREB_3,
                    p_4, c1_4, c2_4, vip_4, P_4, C1_4, C2_4, C1P_4, C2P_4, CREB_4,
                    p_5, c1_5, c2_5, vip_5, P_5, C1_5, C2_5, C1P_5, C2P_5, CREB_5,
                    p_6, c1_6, c2_6, vip_6, P_6, C1_6, C2_6, C1P_6, C2P_6, CREB_6,
                    p_7, c1_7, c2_7, vip_7, P_7, C1_7, C2_7, C1P_7, C2P_7, CREB_7])
    
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
    kcouple = cs.ssym('kcouple')

    paramset = cs.vertcat([vtpr  , vtc1r , vtc2r  , knpr   , kncr   , 
                           vdp   , vdc1  , vdc2   , kdp    , kdc    ,
                           vdP   , kdP   , vdC1   , vdC2   , kdC    , 
                           vdC1n  , vdC2n  , kdCn   , vaCP   , vdCP   , ktlnp,
                        
                           vtpp  , vtc1p , vtc2p  , vtvp   , vtvr   , 
                           knpp  , kncp  , knvp   , knvr   , vdv    ,
                           kdv   , vdVIP , kdVIP  , vgpcr  , kgpcr  , 
                           vdCREB, kdCREB, ktlnv  , vdpka  , vgpka  , 
                           kdpka , kgpka, kdc1,kdc2,ktlnc,kcouple])
                        
    
    
    # Model Equations
    ode = [[]]*EqCount
    
    def transcription(rep1,rep2,vmax,km,n):
        return vmax/(km + (rep1 + rep2)**n)
    
    def rtranscription(rep1,rep2,vmax,km,n):
        return vmax/(km + (rep1 + rep2)**(n))
        
    def ptranscription(pro,vmax,km,n):
        return (vmax*pro**n)/(km+pro**n)
        
    def viptranscription(pro,vmax,km,n):
        return (vmax*pro/3)/(km+pro/3)
    
    def translation(mrna,kt):
        return kt*mrna
    
    def michaelisMenten(s,vmax,km):
        return vmax*s/(km+s)
       
    def sharedDegradation(currentspecies,otherspecies,vmax,km):
        return vmax*(currentspecies)/(km + (currentspecies + otherspecies))
    
    def HillTypeActivation(a, vmax, km, n):
        return (a*vmax)/(km + a**n)
        
    def HillTypeRepression(rep1,rep2, vmax, km, n):
        return (vmax)/(km + (rep1+rep2)**n)
    
    def Complexing(ka,species1,species2,kd,complex):
        #Leave positive for complexes, negative for reacting species
        return ka*species1*species2 - kd*complex
    
    
    
    # MRNA Species
    ode[0] = ptranscription(CREB,vtpp,  knpp,  1) + rtranscription(C1P,C2P,vtpr,knpr,1)   - michaelisMenten(p,vdp,kdp)
    ode[1] = rtranscription(C1P,C2P,vtc1r,kncr,1)  - michaelisMenten(c1,vdc1,kdc)
    ode[2] = rtranscription(C1P,C2P,vtc2r,kncr,1)  - michaelisMenten(c2,vdc2,kdc)
    ode[3] = rtranscription(C1P,C2P,vtvr,knvr,1)   - michaelisMenten(vip,vdv,kdv)
    
    # Free Proteins
    ode[4] = translation(p,ktlnp)    - michaelisMenten(P,vdP,kdP)        - Complexing(vaCP,C1,P,vdCP,C1P)  - Complexing(vaCP,C2,P,vdCP,C2P)
    ode[5] = translation(c1,ktlnc)       - michaelisMenten(C1,vdC1,kdC)      - Complexing(vaCP,C1,P,vdCP,C1P)
    ode[6] = translation(c2,ktlnc)       - michaelisMenten(C2,vdC2,kdC)      - Complexing(vaCP,C2,P,vdCP,C2P)
    ode[7] = translation(vip,ktlnv) + translation(vip_2,ktlnv) +translation(vip_3,ktlnv) - kdVIP*eVIP

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[8] = Complexing(vaCP,C1,P,vdCP,C1P) -sharedDegradation(C1P,C2P,vdC1n,kdCn) 
    ode[9] = Complexing(vaCP,C2,P,vdCP,C2P) -sharedDegradation(C2P,C1P,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[10] = viptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB,vdCREB,kdCREB)
    
    ##
    ## cell 2
    ##
    ode[11] = ptranscription(CREB_2,vtpp,  knpp,  1) + rtranscription(C1P_2,C2P_2,vtpr,knpr,1)   - michaelisMenten(p_2,vdp,kdp)
    ode[12] = rtranscription(C1P_2,C2P_2,vtc1r,kncr,1)  - michaelisMenten(c1_2,vdc1,kdc)
    ode[13] = rtranscription(C1P_2,C2P_2,vtc2r,kncr,1)  - michaelisMenten(c2_2,vdc2,kdc)
    ode[14] = rtranscription(C1P_2,C2P_2,vtvr,knvr,1)   - michaelisMenten(vip_2,vdv,kdv)
    
    # Free Proteins
    ode[15] = translation(p_2,ktlnp)    - michaelisMenten(P_2,vdP,kdP)        - Complexing(vaCP,C1_2,P_2,vdCP,C1P_2)  - Complexing(vaCP,C2_2,P_2,vdCP,C2P_2)
    ode[16] = translation(c1_2,ktlnc)       - michaelisMenten(C1_2,vdC1,kdC)      - Complexing(vaCP,C1_2,P_2,vdCP,C1P_2)
    ode[17] = translation(c2_2,ktlnc)       - michaelisMenten(C2_2,vdC2,kdC)      - Complexing(vaCP,C2_2,P_2,vdCP,C2P_2)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[18] = Complexing(vaCP,C1_2,P_2,vdCP,C1P_2) -sharedDegradation(C1P_2,C2P_2,vdC1n,kdCn) 
    ode[19] = Complexing(vaCP,C2_2,P_2,vdCP,C2P_2) -sharedDegradation(C2P_2,C1P_2,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[20] = viptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB_2,vdCREB,kdCREB)
    ##
    ## cell 3
    ##
    ode[21] = ptranscription(CREB_3,vtpp,  knpp,  1) + rtranscription(C1P_3,C2P_3,vtpr,knpr,1)   - michaelisMenten(p_3,vdp,kdp)
    ode[22] = rtranscription(C1P_3,C2P_3,vtc1r,kncr,1)  - michaelisMenten(c1_3,vdc1,kdc)
    ode[23] = rtranscription(C1P_3,C2P_3,vtc2r,kncr,1)  - michaelisMenten(c2_3,vdc2,kdc)
    ode[24] = rtranscription(C1P_3,C2P_3,vtvr,knvr,1)   - michaelisMenten(vip_3,vdv,kdv)
    
    # Free Proteins
    ode[25] = translation(p_3,ktlnp)    - michaelisMenten(P_3,vdP,kdP)        - Complexing(vaCP,C1_3,P_3,vdCP,C1P_3)  - Complexing(vaCP,C2_3,P_3,vdCP,C2P_3)
    ode[26] = translation(c1_3,ktlnc)       - michaelisMenten(C1_3,vdC1,kdC)      - Complexing(vaCP,C1_3,P_3,vdCP,C1P_3)
    ode[27] = translation(c2_3,ktlnc)       - michaelisMenten(C2_3,vdC2,kdC)      - Complexing(vaCP,C2_3,P_3,vdCP,C2P_3)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[28] = Complexing(vaCP,C1_3,P_3,vdCP,C1P_3) -sharedDegradation(C1P_3,C2P_3,vdC1n,kdCn) 
    ode[29] = Complexing(vaCP,C2_3,P_3,vdCP,C2P_3) -sharedDegradation(C2P_3,C1P_3,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[30] = viptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB_3,vdCREB,kdCREB)
    
    ##
    ## cell 4
    ##
    ode[31] = ptranscription(CREB_4,vtpp,  knpp,  1) + rtranscription(C1P_4,C2P_4,vtpr,knpr,1)   - michaelisMenten(p_4,vdp,kdp)
    ode[32] = rtranscription(C1P_4,C2P_4,vtc1r,kncr,1)  - michaelisMenten(c1_4,vdc1,kdc)
    ode[33] = rtranscription(C1P_4,C2P_4,vtc2r,kncr,1)  - michaelisMenten(c2_4,vdc2,kdc)
    ode[34] = rtranscription(C1P_4,C2P_4,vtvr,knvr,1)   - michaelisMenten(vip_4,vdv,kdv)
    
    # Free Proteins
    ode[35] = translation(p_4,ktlnp)    - michaelisMenten(P_4,vdP,kdP)        - Complexing(vaCP,C1_4,P_4,vdCP,C1P_4)  - Complexing(vaCP,C2_4,P_4,vdCP,C2P_4)
    ode[36] = translation(c1_4,ktlnc)       - michaelisMenten(C1_4,vdC1,kdC)      - Complexing(vaCP,C1_4,P_4,vdCP,C1P_4)
    ode[37] = translation(c2_4,ktlnc)       - michaelisMenten(C2_4,vdC2,kdC)      - Complexing(vaCP,C2_4,P_4,vdCP,C2P_4)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[38] = Complexing(vaCP,C1_4,P_4,vdCP,C1P_4) -sharedDegradation(C1P_4,C2P_4,vdC1n,kdCn) 
    ode[39] = Complexing(vaCP,C2_4,P_4,vdCP,C2P_4) -sharedDegradation(C2P_4,C1P_4,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[40] = viptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB_4,vdCREB,kdCREB)
    
    ##
    ## cell 3
    ##
    ode[41] = ptranscription(CREB_5,vtpp,  knpp,  1) + rtranscription(C1P_5,C2P_5,vtpr,knpr,1)   - michaelisMenten(p_5,vdp,kdp)
    ode[42] = rtranscription(C1P_5,C2P_5,vtc1r,kncr,1)  - michaelisMenten(c1_5,vdc1,kdc)
    ode[43] = rtranscription(C1P_5,C2P_5,vtc2r,kncr,1)  - michaelisMenten(c2_5,vdc2,kdc)
    ode[44] = rtranscription(C1P_5,C2P_5,vtvr,knvr,1)   - michaelisMenten(vip_5,vdv,kdv)
    
    # Free Proteins
    ode[45] = translation(p_5,ktlnp)    - michaelisMenten(P_5,vdP,kdP)        - Complexing(vaCP,C1_5,P_5,vdCP,C1P_5)  - Complexing(vaCP,C2_5,P_5,vdCP,C2P_5)
    ode[46] = translation(c1_5,ktlnc)       - michaelisMenten(C1_5,vdC1,kdC)      - Complexing(vaCP,C1_5,P_5,vdCP,C1P_5)
    ode[47] = translation(c2_5,ktlnc)       - michaelisMenten(C2_5,vdC2,kdC)      - Complexing(vaCP,C2_5,P_5,vdCP,C2P_5)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[48] = Complexing(vaCP,C1_5,P_5,vdCP,C1P_5) -sharedDegradation(C1P_5,C2P_5,vdC1n,kdCn) 
    ode[49] = Complexing(vaCP,C2_5,P_5,vdCP,C2P_5) -sharedDegradation(C2P_5,C1P_5,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[50] = viptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB_5,vdCREB,kdCREB)
    
    ##
    ## cell 3
    ##
    ode[51] = ptranscription(CREB_6,vtpp,  knpp,  1) + rtranscription(C1P_6,C2P_6,vtpr,knpr,1)   - michaelisMenten(p_6,vdp,kdp)
    ode[52] = rtranscription(C1P_6,C2P_6,vtc1r,kncr,1)  - michaelisMenten(c1_6,vdc1,kdc)
    ode[53] = rtranscription(C1P_6,C2P_6,vtc2r,kncr,1)  - michaelisMenten(c2_6,vdc2,kdc)
    ode[54] = rtranscription(C1P_6,C2P_6,vtvr,knvr,1)   - michaelisMenten(vip_6,vdv,kdv)
    
    # Free Proteins
    ode[55] = translation(p_6,ktlnp)    - michaelisMenten(P_6,vdP,kdP)        - Complexing(vaCP,C1_6,P_6,vdCP,C1P_6)  - Complexing(vaCP,C2_6,P_6,vdCP,C2P_6)
    ode[56] = translation(c1_6,ktlnc)       - michaelisMenten(C1_6,vdC1,kdC)      - Complexing(vaCP,C1_6,P_6,vdCP,C1P_6)
    ode[57] = translation(c2_6,ktlnc)       - michaelisMenten(C2_6,vdC2,kdC)      - Complexing(vaCP,C2_6,P_6,vdCP,C2P_6)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[58] = Complexing(vaCP,C1_6,P_6,vdCP,C1P_6) -sharedDegradation(C1P_6,C2P_6,vdC1n,kdCn) 
    ode[59] = Complexing(vaCP,C2_6,P_6,vdCP,C2P_6) -sharedDegradation(C2P_6,C1P_6,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[60] = viptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB_6,vdCREB,kdCREB)
    
    ##
    ## cell 3
    ##
    ode[61] = ptranscription(CREB_7,vtpp,  knpp,  1) + rtranscription(C1P_7,C2P_7,vtpr,knpr,1)   - michaelisMenten(p_7,vdp,kdp)
    ode[62] = rtranscription(C1P_7,C2P_7,vtc1r,kncr,1)  - michaelisMenten(c1_7,vdc1,kdc)
    ode[63] = rtranscription(C1P_7,C2P_7,vtc2r,kncr,1)  - michaelisMenten(c2_7,vdc2,kdc)
    ode[64] = rtranscription(C1P_7,C2P_7,vtvr,knvr,1)   - michaelisMenten(vip_7,vdv,kdv)
    
    # Free Proteins
    ode[65] = translation(p_7,ktlnp)    - michaelisMenten(P_7,vdP,kdP)        - Complexing(vaCP,C1_7,P_7,vdCP,C1P_7)  - Complexing(vaCP,C2_7,P_7,vdCP,C2P_7)
    ode[66] = translation(c1_7,ktlnc)       - michaelisMenten(C1_7,vdC1,kdC)      - Complexing(vaCP,C1_7,P_7,vdCP,C1P_7)
    ode[67] = translation(c2_7,ktlnc)       - michaelisMenten(C2_7,vdC2,kdC)      - Complexing(vaCP,C2_7,P_7,vdCP,C2P_7)

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[68] = Complexing(vaCP,C1_7,P_7,vdCP,C1P_7) -sharedDegradation(C1P_7,C2P_7,vdC1n,kdCn) 
    ode[69] = Complexing(vaCP,C2_7,P_7,vdCP,C2P_7) -sharedDegradation(C2P_7,C1P_7,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[70] = viptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB_7,vdCREB,kdCREB)
    
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","deg_sync")
    
    return fn
