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
modelversion='sds230'


period = 23.7000

vol=400

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
    p    = cs.SX.sym("p")
    c1   = cs.SX.sym("c1")
    c2   = cs.SX.sym("c2")
    vip  = cs.SX.sym("vip")
    P    = cs.SX.sym("P")
    C1   = cs.SX.sym("C1")
    C2   = cs.SX.sym("C2")
    eVIP = cs.SX.sym("eVIP")
    C1P  = cs.SX.sym("C1P")
    C2P  = cs.SX.sym("C2P")
    CREB = cs.SX.sym("CREB")
    
    #for Casadi
    y = cs.vertcat([p, c1, c2, vip, P, C1, C2, eVIP, C1P, C2P, CREB])
    
    # Time Variable
    t = cs.SX.sym("t")
    
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    
    #mRNA
    vtpp    = cs.SX.sym("vtpp")
    vtpr = cs.SX.sym("vtpr")
    vtc1r   = cs.SX.sym("vtc1r")
    vtc2r   = cs.SX.sym("vtc2r")
    knpr   = cs.SX.sym("knpr")
    kncr   = cs.SX.sym("kncr")
    vtvr   = cs.SX.sym("vtvr")
    knvr   = cs.SX.sym("knvr")
    
    vdp    = cs.SX.sym("vdp")         #degradation of per
    vdc1   = cs.SX.sym("vdc1")        #degradation of cry1
    vdc2   = cs.SX.sym("vdc2")        #degradation of cry2
    kdp    = cs.SX.sym("kdp")         #degradation of per
    kdc    = cs.SX.sym("kdc")         #degradation of cry1 and 2
    vdv    = cs.SX.sym("vdv")         #degradation of vip
    kdv    = cs.SX.sym("kdv")         #degradation of vip
    
    #Proteins
    vdP    = cs.SX.sym("vdP")         #degradation of Per
    vdC1   = cs.SX.sym("vdC1")        #degradation of Cry1
    vdC2   = cs.SX.sym("vdC2")        #degradation of Cry2
    vdC1n   = cs.SX.sym("vdC1n")        #degradation of Cry1-Per or Cry2-Per complex into nothing (nuclear)
    vdC2n   = cs.SX.sym("vdC2n")
    kdP    = cs.SX.sym("kdP")         #degradation of Per
    kdC    = cs.SX.sym("kdC")         #degradation of Cry1 & Cry2
    kdCn   = cs.SX.sym("kdCn")        #degradation of Cry1-Per or Cry2-Per complex into nothing (nuclear)
    vaCP   = cs.SX.sym("vaCP")        #formation of Cry1-Per or Cry2-Per complex
    vdCP   = cs.SX.sym('vdCP')        #degradation of Cry1-Per or Cry2-Per into components in ctyoplasm
    ktlnp  = cs.SX.sym('ktlnp')       #translation of per
    ktlnc  = cs.SX.sym('ktlnc')
    
    #Signalling
    vdVIP = cs.SX.sym("vdVIP")
    kdVIP = cs.SX.sym("kdVIP")
    vdCREB= cs.SX.sym("vdCREB")
    kdCREB= cs.SX.sym("kdCREB")
    ktlnv = cs.SX.sym("ktlnv")
    vgpka = cs.SX.sym("vgpka")
    kgpka = cs.SX.sym("kgpka")
    kcouple = cs.SX.sym("kcouple")    #necessary for stochastic model, does not need to be fit

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
def ssa_desync(fn,y0in,param,cellcount,adjacency=0):
    """
    This is the network-level SSA model, with coupling. Call with:
        SSAcoupled,state_names,param_names = SSAmodelC(ODEmodel(),y0in,param)
    
    To uncouple the model, set adjacency matrix to zeros    
    """
    if adjacency==0:
        adjacency = np.zeros([cellcount,cellcount])
    #Converts concentration to population
    y0in_ssa = (vol*y0in).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    y0in_pop = []
    

    #coupling section===========================
    for indx in range(cellcount):
        index = '_'+str(indx)+'_0'
        #loops to include all species, normally this is the only line needed without index
        species_array = species_array + [fn.inputExpr(cs.DAE_X)[i].getName()+index
                        for i in xrange(EqCount)]
        state_names = state_names + [fn.inputExpr(cs.DAE_X)[i].getName()+index
                        for i in xrange(EqCount)]  
           
        y0in_pop = np.append(y0in_pop, y0in_ssa)       
                


    #===========================================
            
    param_array   = [fn.inputExpr(cs.DAE_P)[i].getName()
                    for i in xrange(ParamCount)]
    param_names   = [fn.inputExpr(cs.DAE_P)[i].getName()
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
    
def ssa_resync(fn,y0in_desync,param,cellcount,adjacency):
    """
    This is the network-level SSA model, with coupling. Call with:
        SSAcoupled,state_names,param_names = SSAmodelC(ODEmodel(),y0in,param)
    
    The raw numbers in the adjacency matrix MATTER A LOT here--each row should 
    be normalized. Anything unconnected is connected at half-weight to the mean field.
    """
    
    #Converts concentration to population
    y0in_ssa = (vol*y0in).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    y0in_pop = y0in_desync
    

    #coupling section===========================
    for indx in range(cellcount):
        index = '_'+str(indx)+'_0'
        #loops to include all species, normally this is the only line needed without index
        species_array = species_array + [fn.inputExpr(cs.DAE_X)[i].getName()+index
                        for i in xrange(EqCount)]
        state_names = state_names + [fn.inputExpr(cs.DAE_X)[i].getName()+index
                        for i in xrange(EqCount)]  
        if randomy0==False:                
            y0in_pop = np.append(y0in_pop, y0in_ssa)       
                
    if randomy0 == True:
        #random initial locations
        y0in_pop = 1*np.ones(EqCount*cellcount)
        for i in range(len(y0in_pop)):
            y0in_pop[i] = vol*1*random.random()

    #===========================================
            
    param_array   = [fn.inputExpr(cs.DAE_P)[i].getName()
                    for i in xrange(ParamCount)]
    param_names   = [fn.inputExpr(cs.DAE_P)[i].getName()
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
        
    # set up the adjacency matrix correctly so each cell gets the same amount in
    np.fill_diagonal(adjacency,1)
    row_sums = adjacency.sum(axis=1)
    new_matrix = adjacency / row_sums[:, np.newaxis]
    adjacency=np.nan_to_num(new_matrix)
    

    for row in range(cellcount):
        if row_sums[row]==1:
            adjacency[row,row]=0.75
            viplist = '('
            for i in range(cellcount):
                viplist+='vip_'+str(i)+'_0+'
            viplist=viplist[:-1]+')'
            rxn=stk.Reaction(name='vip from all to '+str(row),
                                products={'eVIP_'+str(row)+'_0':1},
                                propensity_function=viplist+"*"+'ktlnv'+'*0.25*' +str(1./cellcount),
                                annotation='')    
            SSAmodel.addReaction(rxn)
            

    for currentcell in range(cellcount):
        for affectedcell in range(cellcount):
            if adjacency[currentcell,affectedcell]!= 0:
                #The Coupling Part
                rxn=stk.Reaction(name='vip from '+str(currentcell)+' to '+str(affectedcell),
                                products={'eVIP_'+str(affectedcell)+'_0':1},
                                propensity_function='vip_'+str(currentcell)+"_0*"+'ktlnv'+'*' +str(adjacency[affectedcell,currentcell]),
                                annotation='')    
                SSAmodel.addReaction(rxn)


        
    return SSAmodel,state_names,param_names
    










