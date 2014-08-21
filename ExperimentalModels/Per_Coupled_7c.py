# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:19:48 2014

@author: john abel

important bit is that C2 always KO

"""

import numpy  as np
import casadi as cs
import pdb
import jha_CommonFiles.stochkit_resources as stk
import jha_CommonFiles.modelbuilder as mb
import pylab as pl
import jha_CommonFiles.circadiantoolbox_raw as ctb
import jha_CommonFiles.Bioluminescence as bl
import random

EqCount = 36
ParamCount = 47
modelversion='SDScoupled16'

xlen = 15
ylen = 15

period = 23.7000

vol=50


randomy0 = False

#y0in=np.array([ 0.11059226,  1.13722226,  0.19297633,  
#				0.07438731,  0.00253855, 0.71412118])


        
y0in=np.ones(EqCount)


#Fit found for parameters          


param_found = []
param = [ 0.15704759,  0.21579994,  0.29338489,  0.18794926,  0.46814722,
        0.3462655 ,  0.18149654,  0.07259537,  0.03808789,  5.04749824,
        0.37843396,  0.15193618,  1.91396048,  0.40534079,  0.11299814,
        0.11369839,  0.13747806,  0.40330342,  0.22068537,  0.10426225,
        0.30429049,  0.25967946,  0.0786023 ,  2.18644298,  0.05054923,
        0.37496549,  0.11065244,  0.64290944,  0.70368903,  0.29934383,
        1.8759093 ,  0.32127314,  0.36273626,  0.21511456,  2.30089036,
        0.70699016,  0.14956068,  0.13514157,  1.00308042,  4.88617671,
        0.23975108,  0.30612171,  2.17658643,  0.93143448,  1.07122088,
        0.15528874,  0.        ]

      
#VIP KO
#param[24]=0
#param[25]=0


def ODEmodel():
    """ trying non-reversible nuclear entry and fitting knockouts. """
    #==================================================================
    #State variable definitions
    #==================================================================
    p    = cs.ssym("p")
    vip  = cs.ssym("vip")
    P    = cs.ssym("P")
    eVIP = cs.ssym("eVIP")
    Ptf  = cs.ssym("Ptf")
    CREB = cs.ssym("CREB")
    
    p_2    = cs.ssym("p_2")
    vip_2  = cs.ssym("vip_2")
    P_2    = cs.ssym("P_2")
    Ptf_2  = cs.ssym("Ptf_2")
    CREB_2 = cs.ssym("CREB_2")
    
    p_3    = cs.ssym("p_3")
    vip_3  = cs.ssym("vip_3")
    P_3    = cs.ssym("P_3")
    Ptf_3  = cs.ssym("Ptf_3")
    CREB_3 = cs.ssym("CREB_3")
    
    p_4    = cs.ssym("p_4")
    vip_4  = cs.ssym("vip_4")
    P_4    = cs.ssym("P_4")
    Ptf_4  = cs.ssym("Ptf_4")
    CREB_4 = cs.ssym("CREB_4")
    
    p_5    = cs.ssym("p_5")
    vip_5  = cs.ssym("vip_5")
    P_5    = cs.ssym("P_5")
    Ptf_5  = cs.ssym("Ptf_5")
    CREB_5 = cs.ssym("CREB_5")
    
    p_6    = cs.ssym("p_6")
    vip_6  = cs.ssym("vip_6")
    P_6    = cs.ssym("P_6")
    Ptf_6  = cs.ssym("Ptf_6")
    CREB_6 = cs.ssym("CREB_6")
    
    p_7    = cs.ssym("p_7")
    vip_7  = cs.ssym("vip_7")
    P_7    = cs.ssym("P_7")
    Ptf_7  = cs.ssym("Ptf_7")
    CREB_7 = cs.ssym("CREB_7")
    
    #for Casadi
    y = cs.vertcat([p, vip, P, eVIP, Ptf, CREB,
                    p_2,vip_2,P_2,Ptf_2,CREB_2,
                    p_3,vip_3,P_3,Ptf_3,CREB_3,
                    p_4,vip_4,P_4,Ptf_4,CREB_4,
                    p_5,vip_5,P_5,Ptf_5,CREB_5,
                    p_6,vip_6,P_6,Ptf_6,CREB_6,
                    p_7,vip_7,P_7,Ptf_7,CREB_7])
    
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
    vdC1n   = cs.ssym("vdC1n")      #degradation of Cry1-Per or 
									#Cry2-Per complex into nothing (nuclear)
    vdC2n   = cs.ssym("vdC2n")      #Cry2 degradation multiplier
    kdP    = cs.ssym("kdP")         #degradation of Per
    kdC    = cs.ssym("kdC")         #degradation of Cry1 & Cry2
    kdCn   = cs.ssym("kdCn")        #degradation of Cry1-Per or Cry2-Per co
									#mplex into nothing (nuclear)
    vaCP   = cs.ssym("vaCP")        #formation of Cry1-Per or Cry2-Per 
											#complex
    vdCP   = cs.ssym('vdCP') 		#degradation of Cry1-Per or Cry2-Per 
										#into components in ctyoplasm
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
                           vdC1n , vdC2n , kdCn   , vaCP   , vdCP   , ktlnp,
                        
                           vtpp  , vtc1p , vtc2p  , vtvp   , vtvr   , 
                           knpp  , kncp  , knvp   , knvr   , vdv    ,
                           kdv   , vdVIP , kdVIP  , vgpcr  , kgpcr  , 
                           vdCREB, kdCREB, ktlnv  , vdpka  , vgpka  , 
                           kdpka , kgpka, kdc1,kdc2,ktlnc,kcouple])
    #for optional Stochastic Simulation
                        
    
    #=====================================================
    # Model Equations
    #=====================================================
    
    ode = [[]]*EqCount

    # MRNA Species
    ode[0] = (mb.prtranscription3(CREB,Ptf,vtpr,vtpp,knpr)   
				- mb.michaelisMenten(p,vdp,kdp))
    ode[1] = mb.rtranscription(P,P,vtvr,knvr,1)   - mb.michaelisMenten(vip,vdv,kdv)


    # Free Proteins
    ode[2] = (mb.translation(p,ktlnp) -mb.PerTF(P,Ptf,vaCP,vdCP)   - mb.michaelisMenten(P,vdP,kdP))
    ode[3] = mb.translation(vip,ktlnv) + mb.translation(vip_2,ktlnv) + mb.translation(vip_3,ktlnv) - kdVIP*eVIP

    # PER/CRY Cytoplasm Complexes in the Nucleus           
    ode[4] = mb.PerTF(P,Ptf,vaCP,vdCP) -mb.michaelisMenten(Ptf,kdc,kdCn)

    #Sgnaling Pathway
    ode[5] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB,vdCREB,kdCREB)
    
    #cell 2
    ode[6] = (mb.prtranscription3(CREB_2,Ptf_2,vtpr,vtpp,knpr)   
				- mb.michaelisMenten(p_2,vdp,kdp))
    ode[7] = mb.rtranscription2(Ptf_2,vtvr,knvr,1)   - mb.michaelisMenten(vip_2,vdv,kdv)
    ode[8] = (mb.translation(p_2,ktlnp) -mb.PerTF(P_2,Ptf,vaCP,vdCP)   - mb.michaelisMenten(P_2,vdP,kdP))         
    ode[9] = mb.PerTF(P_2,Ptf_2,vaCP,vdCP) -mb.michaelisMenten(Ptf_2,kdc,kdCn)
    ode[10] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB_2,vdCREB,kdCREB)
    
    #cell 3
    ode[11] = (mb.prtranscription3(CREB_3,Ptf_3,vtpr,vtpp,knpr)   
				- mb.michaelisMenten(p_3,vdp,kdp))
    ode[12] = mb.rtranscription2(Ptf_3,vtvr,knvr,1)   - mb.michaelisMenten(vip_3,vdv,kdv)
    ode[13] = (mb.translation(p_3,ktlnp) -mb.PerTF(P_3,Ptf,vaCP,vdCP)   - mb.michaelisMenten(P_3,vdP,kdP))         
    ode[14] = mb.PerTF(P_3,Ptf_3,vaCP,vdCP) -mb.michaelisMenten(Ptf_3,kdc,kdCn)
    ode[15] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB_3,vdCREB,kdCREB)
    
    
    #cell 4
    ode[16] = (mb.prtranscription3(CREB_4,Ptf_4,vtpr,vtpp,knpr)   
				- mb.michaelisMenten(p_4,vdp,kdp))
    ode[17] = mb.rtranscription2(Ptf_4,vtvr,knvr,1)   - mb.michaelisMenten(vip_4,vdv,kdv)
    ode[18] = (mb.translation(p_4,ktlnp) -mb.PerTF(P_4,Ptf,vaCP,vdCP)   - mb.michaelisMenten(P_4,vdP,kdP))         
    ode[19] = mb.PerTF(P_4,Ptf_4,vaCP,vdCP) -mb.michaelisMenten(Ptf_4,kdc,kdCn)
    ode[20] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB_4,vdCREB,kdCREB)
    
    #cell 5
    ode[21] = (mb.prtranscription3(CREB_5,Ptf_5,vtpr,vtpp,knpr)   
				- mb.michaelisMenten(p_5,vdp,kdp))
    ode[22] = mb.rtranscription2(Ptf_5,vtvr,knvr,1)   - mb.michaelisMenten(vip_5,vdv,kdv)
    ode[23] = (mb.translation(p_5,ktlnp) -mb.PerTF(P_5,Ptf,vaCP,vdCP)   - mb.michaelisMenten(P_5,vdP,kdP))         
    ode[24] = mb.PerTF(P_5,Ptf_5,vaCP,vdCP) -mb.michaelisMenten(Ptf_5,kdc,kdCn)
    ode[25] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB_5,vdCREB,kdCREB)
    
    #cell 6
    ode[26] = (mb.prtranscription3(CREB_6,Ptf_6,vtpr,vtpp,knpr)   
				- mb.michaelisMenten(p_6,vdp,kdp))
    ode[27] = mb.rtranscription2(Ptf_6,vtvr,knvr,1)   - mb.michaelisMenten(vip_6,vdv,kdv)
    ode[28] = (mb.translation(p_6,ktlnp) -mb.PerTF(P_6,Ptf,vaCP,vdCP)   - mb.michaelisMenten(P_6,vdP,kdP))         
    ode[29] = mb.PerTF(P_6,Ptf_6,vaCP,vdCP) -mb.michaelisMenten(Ptf_6,kdc,kdCn)
    ode[30] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB_6,vdCREB,kdCREB)
    
    #cell 7
    ode[31] = (mb.prtranscription3(CREB_7,Ptf_7,vtpr,vtpp,knpr)   
				- mb.michaelisMenten(p_7,vdp,kdp))
    ode[32] = mb.rtranscription2(Ptf_7,vtvr,knvr,1)   - mb.michaelisMenten(vip_7,vdv,kdv)
    ode[33] = (mb.translation(p_7,ktlnp) -mb.PerTF(P_7,Ptf,vaCP,vdCP)   - mb.michaelisMenten(P_7,vdP,kdP))         
    ode[34] = mb.PerTF(P_7,Ptf_7,vaCP,vdCP) -mb.michaelisMenten(Ptf_7,kdc,kdCn)
    ode[35] = mb.viptranscription(eVIP,vgpka,kgpka,1) - mb.michaelisMenten(CREB_7,vdCREB,kdCREB)
    
    
    
    
    
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","PerOnly")
    
    return fn
    #==================================================================
    # Stochastic Model Portion
    #==================================================================
    
def SSAmodelC(fn,y0in,param):
    
    #Converts concentration to population
    y0in_ssa = (vol*y0in).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    if randomy0==False:
        y0in_pop = []
    

    #coupling section===========================
    for indx in range(xlen):
        for indy in range(ylen):
            index = '_'+str(indx)+'_'+str(indy)
            #loops to include all species, normally this is the only line needed without index
            species_array = species_array + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]
            state_names = state_names + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]  
            if randomy0==False:                
                y0in_pop = np.append(y0in_pop, y0in_ssa)       
                
    if randomy0 == True:
        #random initial locations
        y0in_pop = 1*np.ones(EqCount*xlen*ylen)
        for i in range(len(y0in_pop)):
            y0in_pop[i] = vol*1*random.random()

    #===========================================
            
    param_array   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    param_names   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    #Names model
    SSAmodel = stk.StochKitModel(name=modelversion)
    
    #creates SSAmodel class object
    SSA_builder = mb.SSA_builder(species_array,param_array,y0in_pop,param,SSAmodel,vol)
    

    #coupling section
    for indx in range(xlen):
        for indy in range(ylen):
            
            index = '_'+str(indx)+'_'+str(indy)
            
            
            
            # REACTIONS
    
            
            #per mRNA
    
            SSA_builder.SSA_PR16f('per mRNA activation'+index,'p'+index,'CREB'+index
                                ,'C1P'+index,'C2P'+index,'vtpr','vtpp','knpr')
            SSA_builder.SSA_MM('per mRNA degradation'+index,'vdp',km=['kdp'],Rct=['p'+index])
    
    
            #cry1 mRNA
            SSA_builder.SSA_MM('c1 mRNA repression'+index,'vtc1r',km=['kncr'],Prod=['c1'+index],Rep=['C1P'+index,'C2P'+index])
            SSA_builder.SSA_MM('c1 mRNA degradation'+index,'vdc1',km=['kdc'],Rct=['c1'+index])
            
            #cry2 mRNA
            SSA_builder.SSA_MM('c2 mRNA repression'+index,'vtc2r',km=['kncr'],Prod=['c2'+index],Rep=['C1P'+index,'C2P'+index])
            SSA_builder.SSA_MM('c2 mRNA degradation'+index,'vdc2',km=['kdc'],Rct=['c2'+index])
            
            #vip mRNA
            SSA_builder.SSA_MM('vip mRNA repression'+index,'vtvr',km=['knvr'],Prod=['vip'+index],Rep=['C1P'+index,'C2P'+index])
            SSA_builder.SSA_MM('vip mRNA degradation'+index,'vdv',km=['kdv'],Rct=['vip'+index])
            
            #CRY1, CRY2, PER, VIP creation and degradation
            SSA_builder.SSA_MA_tln('PER translation' +index,'P' +index  ,'ktlnp','p'+index)
            SSA_builder.SSA_MA_tln('CRY1 translation'+index,'C1'+index  ,'ktlnc','c1'+index)
            SSA_builder.SSA_MA_tln('CRY2 translation'+index,'C2'+index  ,'ktlnc','c2'+index)
            SSA_builder.SSA_MA_tln('VIP translation' +index,'eVIP'+index,'ktlnv','vip'+index)
            
            SSA_builder.SSA_MM('PER degradation'+index,'vdP',km=['kdP'],Rct=['P'+index])
            SSA_builder.SSA_MM('C1 degradation'+index,'vdC1',km=['kdC'],Rct=['C1'+index])
            SSA_builder.SSA_MM('C2 degradation'+index,'vdC2',km=['kdC'],Rct=['C2'+index])
            SSA_builder.SSA_MA_deg('eVIP degradation'+index,'eVIP'+index,'kdVIP')
            
            #CRY1 CRY2 complexing
            SSA_builder.SSA_MA_complex('CRY1-P complex'+index,'C1'+index,'P'+index,'C1P'+index,'vaCP','vdCP')
            SSA_builder.SSA_MA_complex('CRY2-P complex'+index,'C2'+index,'P'+index,'C2P'+index,'vaCP','vdCP')
            SSA_builder.SSA_MM('C1P degradation'+index,'vdC1n',km=['kdCn'],Rct=['C1P'+index,'C2P'+index])
            SSA_builder.SSA_MM('C2P degradation'+index,'vdC2n',km=['kdCn'],Rct=['C2P'+index,'C1P'+index])
            
            #VIP/CREB Pathway
            SSA_builder.SSA_MM('CREB formation'+index,'vgpka',km=['kgpka'],Prod=['CREB'+index],Act=['eVIP'+index])
            SSA_builder.SSA_MM('CREB degradation'+index,'vdCREB',km=['kdCREB'],Rct=['CREB'+index])

            SSA_builder.SSA_MA_meanfield('eVIP',indx,xlen,indy,ylen,'kcouple')
            
    return SSAmodel,state_names,param_names
    
def SSAmodelU(fn,y0in,param):
    
    #Converts concentration to population
    y0in_ssa = (vol*y0in).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    if randomy0==False:
        y0in_pop = []
    

    #coupling section===========================
    for indx in range(xlen):
        for indy in range(ylen):
            index = '_'+str(indx)+'_'+str(indy)
            #loops to include all species, normally this is the only line needed without index
            species_array = species_array + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]
            state_names = state_names + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                            for i in xrange(EqCount)]  
            if randomy0==False:                
                y0in_pop = np.append(y0in_pop, y0in_ssa)       
                
    if randomy0 == True:
        #random initial locations
        y0in_pop = 1*np.ones(EqCount*xlen*ylen)
        for i in range(len(y0in_pop)):
            y0in_pop[i] = vol*1*random.random()

    #===========================================
            
    param_array   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    param_names   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                    for i in xrange(ParamCount)]
    #Names model
    SSAmodel = stk.StochKitModel(name=modelversion)
    
    #creates SSAmodel class object
    SSA_builder = mb.SSA_builder(species_array,param_array,y0in_pop,param,SSAmodel,vol)
    

    #coupling section
    for indx in range(xlen):
        for indy in range(ylen):
            
            index = '_'+str(indx)+'_'+str(indy)
            
            
            
            # REACTIONS
    
            
            #per mRNA
    
            SSA_builder.SSA_PR16f('per mRNA activation'+index,'p'+index,'CREB'+index
                                ,'C1P'+index,'C2P'+index,'vtpr','vtpp','knpr')
            SSA_builder.SSA_MM('per mRNA degradation'+index,'vdp',km=['kdp'],Rct=['p'+index])
    
    
            #cry1 mRNA
            SSA_builder.SSA_MM('c1 mRNA repression'+index,'vtc1r',km=['kncr'],Prod=['c1'+index],Rep=['C1P'+index,'C2P'+index])
            SSA_builder.SSA_MM('c1 mRNA degradation'+index,'vdc1',km=['kdc'],Rct=['c1'+index])
            
            #cry2 mRNA
            SSA_builder.SSA_MM('c2 mRNA repression'+index,'vtc2r',km=['kncr'],Prod=['c2'+index],Rep=['C1P'+index,'C2P'+index])
            SSA_builder.SSA_MM('c2 mRNA degradation'+index,'vdc2',km=['kdc'],Rct=['c2'+index])
            
            #vip mRNA
            SSA_builder.SSA_MM('vip mRNA repression'+index,'vtvr',km=['knvr'],Prod=['vip'+index],Rep=['C1P'+index,'C2P'+index])
            SSA_builder.SSA_MM('vip mRNA degradation'+index,'vdv',km=['kdv'],Rct=['vip'+index])
            
            #CRY1, CRY2, PER, VIP creation and degradation
            SSA_builder.SSA_MA_tln('PER translation' +index,'P' +index  ,'ktlnp','p'+index)
            SSA_builder.SSA_MA_tln('CRY1 translation'+index,'C1'+index  ,'ktlnc','c1'+index)
            SSA_builder.SSA_MA_tln('CRY2 translation'+index,'C2'+index  ,'ktlnc','c2'+index)
            SSA_builder.SSA_MA_tln('VIP translation' +index,'eVIP'+index,'ktlnv','vip'+index)
            
            SSA_builder.SSA_MM('PER degradation'+index,'vdP',km=['kdP'],Rct=['P'+index])
            SSA_builder.SSA_MM('C1 degradation'+index,'vdC1',km=['kdC'],Rct=['C1'+index])
            SSA_builder.SSA_MM('C2 degradation'+index,'vdC2',km=['kdC'],Rct=['C2'+index])
            SSA_builder.SSA_MA_deg('eVIP degradation'+index,'eVIP'+index,'kdVIP')
            
            #CRY1 CRY2 complexing
            SSA_builder.SSA_MA_complex('CRY1-P complex'+index,'C1'+index,'P'+index,'C1P'+index,'vaCP','vdCP')
            SSA_builder.SSA_MA_complex('CRY2-P complex'+index,'C2'+index,'P'+index,'C2P'+index,'vaCP','vdCP')
            SSA_builder.SSA_MM('C1P degradation'+index,'vdC1n',km=['kdCn'],Rct=['C1P'+index,'C2P'+index])
            SSA_builder.SSA_MM('C2P degradation'+index,'vdC2n',km=['kdCn'],Rct=['C2P'+index,'C1P'+index])
            
            #VIP/CREB Pathway
            SSA_builder.SSA_MM('CREB formation'+index,'vgpka',km=['kgpka'],Prod=['CREB'+index],Act=['eVIP'+index])
            SSA_builder.SSA_MM('CREB degradation'+index,'vdCREB',km=['kdCREB'],Rct=['CREB'+index])
            

    return SSAmodel

def main():
    
    #Creates SSA version of model.
    ODEmodel,state_names,param_names=ODEmodel()
    
    ODEsol = ctb.CircEval(ODEmodel, param, y0in)
    y0LC = ODEsol.burnTransient_sim(tf = 1000, numsteps = 1000)
    sol = ODEsol.intODEs_sim(y0LC,500,numsteps=1000)
    tsol = ODEsol.ts
    periodsol = ODEsol.findPeriod(tsol,sol,'p')
    period = periodsol[0]
    ODEplts = ctb.ODEplots(tsol,vol*sol,period,ODEmodel)      
    ODEplts.fullPlot('p')

    """
    print period
    SSAmodelC = SSAmodelC(ODEmodel,y0in)
    SSAmodelU = SSAmodelU(ODEmodel,y0in)
    uncoupled = stk.stochkit(SSAmodelU,job_id='uncp',t=1000,number_of_trajectories=1,increment=0.1)
    coupled = stk.stochkit(SSAmodelC,job_id='coup',t=1000,number_of_trajectories=1,increment=0.1)
    CoupledEval = stk.StochPopEval(coupled,state_names,param_names,vol,EqCount)
    UncoupledEval = stk.StochPopEval(uncoupled,state_names,param_names,vol,EqCount)
    CoupledEval.PlotPopPartial('p',period,Desc='SCN Network, 225 Cells', tstart=1*period,color='blue',traces=False)
    UncoupledEval.PlotPopPartial('p',period,Desc='SCN Network, 225 Cells', tstart=1*period,color='black',traces=False)
    """
    
    pl.show()
    pdb.set_trace()

if __name__ == "__main__":
    main()  















