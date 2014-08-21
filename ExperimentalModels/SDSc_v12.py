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
modelversion='SDScoupled12'

y0in = np.array([ 0.09909523,  0.70371313,  0.2269922 ,  0.10408456,  0.00490967,
        0.86826377,  0.89688085,  0.06720938,  0.42133251,  0.00728958,
        0.47681956])

period = 24
vol=100
        
#parameter initialization
xlen = 5
ylen = 5

#Fit found for parameters            
param=[  1.46008989e-01,   1.60721010e-01,   7.38795774e-02,
         3.10521600e-01,   1.89227965e-01,   2.97741161e-01,
         7.38035484e-01,   8.68469114e-01,   9.11115937e-03,
         2.44835657e+00,   4.97955007e+00,   2.91848311e-02,
         1.95955825e+00,   4.58981583e-01,   3.26773051e+00,
         3.13845632e-02,   5.89802601e-02,   3.71071655e-02,
         1.06608705e-01,   1.92066870e-03,   3.04212390e+00,
         1.74563138e-01,   2.06094237e-01,   7.16095309e-02,
         9.32982489e-01,   1.65104689e-01,   6.59762203e-01,
         1.21148416e-01,   2.68779027e-01,   2.78716314e-01,
         4.18992713e-01,   1.99486030e-01,   5.27947599e-02,
         2.01964977e-01,   1.69274661e-01,   2.13025004e-01,
         5.00622529e-01,   7.28291408e-01,   2.23015399e-01,
         2.44455646e-01,   8.72577837e-01,   7.74039863e-02,
         2.69694015e-01,   1.58775582e-01,   1.05096774e+00,
         2.85343544e-01,   10]
         
def FullModel(Conversion=False):
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
    
    fn.setOption("name","SDS")
    
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
                                
                y0in_pop = np.append(y0in_pop, y0in_ssa)       
        
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
        
                SSA_builder.SSA_MM('per mRNA activation'+index,'vtpp',km=['knpp'],Prod=['p'+index],Act=['CREB'+index])
                SSA_builder.SSA_MM('per mRNA repression'+index,'vtpr',km=['knpr'],Prod=['p'+index],Rep=['C1P'+index,'C2P'+index])
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
        
        # END REACTIONS
        #stringSSAmodel = SSAmodel.serialize()
        #print stringSSAmodel
        
        stoch = SSAmodel

    return fn,stoch,state_names,param_names

def main():
    
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


    print period
    pdb.set_trace()

    period=50.789
    trajectories = stk.stochkit(SSAmodel,job_id='coupledhi',t=500,number_of_trajectories=1,increment=0.1)
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















