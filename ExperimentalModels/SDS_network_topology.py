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


EqCount = 11
ParamCount = 35
modelversion='SDSnet'

cellcount=20

period = 23.7000

vol=440

randomy0 = False

#better for stoch
y0in=np.array([ 0.27639502,  1.49578759,  0.23951501,  0.10907372,  0.00704751,
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
    
    fn.setOption("name","3state")
    
    return fn
    
    #==================================================================
    # Stochastic Model Portion
    #==================================================================
    
def SSAnetwork(fn,y0in,param,adjacency):
    """
    This is the network-level SSA model, with coupling. Call with:
        SSAcoupled,state_names,param_names = SSAmodelC(ODEmodel(),y0in,param)
    
    To uncouple the model, set adjacency matrix to zeros    
    """
    
    #Converts concentration to population
    y0in_ssa = (vol*y0in).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    if randomy0==False:
        y0in_pop = []
    

    #coupling section===========================
    for indx in range(cellcount):
        index = '_'+str(indx)+'_0'
        #loops to include all species, normally this is the only line needed without index
        species_array = species_array + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                        for i in xrange(EqCount)]
        state_names = state_names + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                        for i in xrange(EqCount)]  
        if randomy0==False:                
            y0in_pop = np.append(y0in_pop, y0in_ssa)       
                
    if randomy0 == True:
        #random initial locations
        y0in_pop = 1*np.ones(EqCount*cellcount)
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
    for indx in range(cellcount):
            
        index = '_'+str(indx)+'_0'
        
        
        
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
           
    for currentcell in range(cellcount):
        for affectedcell in range(cellcount):
            if adjacency[currentcell,affectedcell]!= 0:
                #The Coupling Part
                rxn=stk.Reaction(name='vip from '+str(currentcell)+' to '+str(affectedcell),
                                products={'eVIP_'+str(affectedcell)+'_0':1},
                                propensity_function='vip_'+str(currentcell)+"_0*"+'ktlnv'+'*' +str(adjacency[currentcell,affectedcell]),
                                annotation='')    
                SSAmodel.addReaction(rxn)
        
    return SSAmodel,state_names,param_names
    










