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

EqCount = 31
ParamCount = 47
modelversion='syn10'

y0in = np.array([ 0.09909523,  0.70371313,  0.2269922 ,  0.10408456,  0.00490967,
        0.86826377,  0.89688085,  0.06720938,  0.42133251,  0.00728958,
        0.47681956])
period = 26.38

param = [  2.52842366e-01,   1.59939424e-01,   3.90985236e-02,
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
    
    #for Casadi
    y = cs.vertcat([p, c1, c2, vip, P, C1, C2, eVIP, C1P, C2P, CREB,
                    p_2, c1_2, c2_2, vip_2, P_2, C1_2, C2_2, C1P_2, C2P_2, CREB_2
                    ,p_3, c1_3, c2_3, vip_3, P_3, C1_3, C2_3, C1P_3, C2P_3, CREB_3])
    
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
    
    def prtranscription(pro,vmax,km,rep1,rep2):
        return (vmax*pro)/((km+pro)*(km+rep1+rep2))
        
    def viptranscription(pro,vmax,km,n):
        return (vmax*pro/2)/(km+pro/2)
    
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
    ode[0] = prtranscription(CREB,vtpp, knpp,C1P,C2P)   - michaelisMenten(p,vdp,kdp)
    ode[1] = rtranscription(C1P,C2P,vtc1r,kncr,1)  - michaelisMenten(c1,vdc1,kdc)
    ode[2] = rtranscription(C1P,C2P,vtc2r,kncr,1)  - michaelisMenten(c2,vdc2,kdc)
    ode[3] = rtranscription(C1P,C2P,vtvr,knvr,1)   - michaelisMenten(vip,vdv,kdv)
    
    # Free Proteins
    ode[4] = translation(p,ktlnp)    - michaelisMenten(P,vdP,kdP)        - Complexing(vaCP,C1,P,vdCP,C1P)  - Complexing(vaCP,C2,P,vdCP,C2P)
    ode[5] = translation(c1,ktlnc)       - michaelisMenten(C1,vdC1,kdC)      - Complexing(vaCP,C1,P,vdCP,C1P)
    ode[6] = translation(c2,ktlnc)       - michaelisMenten(C2,vdC2,kdC)      - Complexing(vaCP,C2,P,vdCP,C2P)
    ode[7] = translation(vip,ktlnv) + translation(vip_2,ktlnv) + translation(vip_3,ktlnv) - kdVIP*eVIP

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[8] = Complexing(vaCP,C1,P,vdCP,C1P) -sharedDegradation(C1P,C2P,vdC1n,kdCn) 
    ode[9] = Complexing(vaCP,C2,P,vdCP,C2P) -sharedDegradation(C2P,C1P,vdC2n,kdCn)

    #Sgnaling Pathway
    ode[10] = ptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB,vdCREB,kdCREB)
    
    ##
    ## cell 2
    ##
    ode[11] = prtranscription(CREB_2,vtpp, knpp,C1P_2,C2P_2)   - michaelisMenten(p_2,vdp,kdp)
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
    ode[20] = ptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB_2,vdCREB,kdCREB)
    
    ##
    ## cell 3
    ##
    ode[21] = prtranscription(CREB_3,vtpp, knpp,C1P_3,C2P_3)   - michaelisMenten(p_3,vdp,kdp)
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
    ode[30] = ptranscription(eVIP,vgpka,kgpka,1) - michaelisMenten(CREB_3,vdCREB,kdCREB)
    
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","deg_sync")
    
    return fn
