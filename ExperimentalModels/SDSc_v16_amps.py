# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:19:48 2014

@author: john abel

"""

import numpy  as np
import casadi as cs
import pdb
import stochkit_resources as stk
import modelbuilder as mb
import pylab as pl
import circadiantoolbox_raw as ctb
import Bioluminescence as bl
import random

EqCount = 11
ParamCount = 47
modelversion='SDScoupled16'

xlen = 15
ylen = 15

period = 5
vol=220

randomy0 = True

y0in=np.array([ 0.11059226,  1.13722226,  0.19297633,  0.07438731,  0.00253855,
        0.71412118,  0.48822501,  0.47379907,  0.36659393,  0.04421288,
        0.23519741])


        
#y0in=random.random()*np.ones(EqCount)

af=5
afp=1
afc = 1


#Fit found for parameters            
param=[  2.25159896e-01*afp,   1.90031674e-01*afc,   6.70692205e-02*afc,
         2.63970881e-01,   1.55918619e-01,   2.59850992e-01*afp,
         7.81672489e-01*afc,   1.23769111e+00*afc,   7.95480246e-03*afp,
         1.94345471e+00,   7.04967650e+00,   3.71990395e-02,
         2.23511192e+00,   4.56343016e-01,   4.22802939e+00,
         1.66443181e-02,   4.67850734e-02,   4.54878164e-02,
         2.67511407e-01,   2.06409164e-03,   4.07754219e+00/afp,
         1.27824043e-01*afp,   2.96422108e-01,   9.31597225e-02,
         3.00583908e-01,   1.58044144e-01,   1.84497550e-01,
         1.21730846e+00,   5.49724882e-01,   1.15067882e-01,
         7.33693360e-01,   1.09767551e-01,   6.64793809e-01,
      af*1.30952725e-01,   3.18395912e-01,   5.85189667e-02,
         6.90747660e-01,   1.00492158e+00,af*9.95205130e-01,
         1.17750519e-01,   4.28259617e-01,   6.82074313e-01,
      af*4.86733215e-01,   1.71831427e+00,   3.28515848e-01,
         3.10683969e-01/afc,   10]
#C2 KO
#param[2]=0
#param[23]=0
#C1 KO
#param[1]=0
#param[22]=0
#VIP KO
#param[24]=0
#param[25]=0

def FullModel(Conversion=False):
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


    """
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
    """
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","SDS16")
    
    stoch=0
    #==================================================================
    # Stochastic Model Portion
    #==================================================================
    
    if Conversion==True:
        print 'Now converting model to StochKit XML format...'

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

        #Add mixing functions
        stoch = SSAmodel
    return fn,stoch,state_names,param_names

def main():
    
    #Creates SSA version of model.
    ODEmodel,SSAmodel,state_names,param_names=FullModel(True)
    
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

    pdb.set_trace()
    print period

    period=50.789
    trajectories = stk.stochkit(SSAmodel,job_id='coupledhi',t=1000,number_of_trajectories=1,increment=0.1)
    trajectories = stk.stochkit(SSAmodel,job_id='coupledhi',t=1000,number_of_trajectories=1,increment=0.1)
    StochPopEval = stk.StochPopEval(trajectories,state_names,param_names,vol,EqCount)
    #StochPopEval.RepIndivPlot('p','Individual Cry1 -/- SCN Neurons',period,tstart = 2*period,tend=7*period)
    StochPopEval.PlotPopPartial('p',period,Desc='SCN Network, 225 Cells', tstart=0,color='black',traces=True)
    StochPopEval.PlotPopPartial('c1',period,Desc='SCN Network, 225 Cells', tstart=0,color='blue',traces=True)
    StochPopEval.PlotPopPartial('c2',period,Desc='SCN Network, 225 Cells', tstart=0,color='red',traces=True)
    
    StochPopEval.PlotPopPartial('C1P',period,Desc='SCN Network, 225 Cells', tstart=0,color='blue',traces=False,fignum=2)
    StochPopEval.PlotPopPartial('C2P',period,Desc='SCN Network, 225 Cells', tstart=0,color='red',traces=False,fignum=2)
    
    pl.show()
    pdb.set_trace()

if __name__ == "__main__":
    main()  















