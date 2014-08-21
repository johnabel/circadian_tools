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

EqCount = 11
ParamCount = 47
modelversion='SDScoupled16'

xlen = 15
ylen = 15

period = 23.7000

vol=200


randomy0 = False

y0in=np.array([ 0.11059226,  1.13722226,  0.19297633,  0.07438731,  0.00253855,
        0.71412118,  0.48822501,  0.47379907,  0.36659393,  0.04421288,
        0.23519741])


        
#y0in=random.random()*np.ones(EqCount)


#Fit found for parameters          
"""
param = [0.225159896, 0.190031674, 0.0670692205, 0.263970881, 0.155918619, 
         0.259850992, 0.781672489, 1.23769111, 0.00795480246, 1.94345471, 
         7.0496765, 0.0371990395, 2.23511192, 0.456343016, 4.22802939, 
         0.0166443181, 0.0467850734, 0.0454878164, 0.267511407, 0.00206409164, 
         4.07754219, 0.127824043, 0.296422108, 0.0931597225, 0.300583908, 
         0.158044144, 0.18449755, 1.21730846, 0.549724882, 0.115067882, 
         0.73369336, 0.109767551, 0.664793809, 0.392858175, 0.318395912, 
         0.0585189667, 0.69074766, 1.00492158, 2.98561539, 0.117750519, 
         0.428259617, 0.682074313, 1.4601996449999999, 1.71831427, 0.328515848, 
         0.310683969, 10]
"""
param = [0.41461483697868157, 0.3499288858096469, 0.12350287248265006,
         0.263970881, 0.155918619, 0.47849585383903664,
         1.4393904836297102, 2.2791141180959396, 0.00795480246, 
         1.94345471, 12.981443519586378, 0.0371990395, 
         4.115788738594497, 0.8403209832056411, 4.22802939, 
         0.030649246860785632, 0.08615115713491293, 0.0454878164, 
         0.49260192589200136, 0.0038008678900245166, 7.508484061391405, 
         0.23537826092440986, 0.5458387846532722, 0.17154654911244369, 
         0.553502423068427, 0.29102628027504185, 0.18449755, 
         1.21730846, 0.549724882, 0.115067882, 
         1.351040563852193, 0.109767551, 1.2241672768536533, 
         0.7234184731696952, 0.5863018747732884, 0.0585189667, 
         1.2719593210520304, 1.00492158, 5.497783842491617, 
         0.21682863203729738, 0.7886075382047947, 0.682074313, 
         1.4601996449999999, 1.71831427, 0.328515848, 
         0.5721009178243037, 10]

#C2 KO
param[2]=0
param[23]=0
#C1 KO
#param[1]=0
#param[22]=0
#VIP KO
#param[24]=0
#param[25]=0


def ODEmodel():
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
    
    fn.setOption("name","SDS16")
    
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
    ODEplts.fullPlot('c1')
    ODEplts.fullPlot('c2')
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















