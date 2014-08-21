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
vol=2000

#Fit found for parameters            
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
         4.46249930e-01,   1]

#1.65788965e-01,   1.72360616e-01


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
        
        #parameter initialization
        xlen = 4
        ylen = 4
        
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
                SSA_builder.SSA_MM('CREB formation'+index,'vgpka',km=['kgpka'],Prod=['CREB'+index],Act=['eVIP_0_0'])
                SSA_builder.SSA_MM('CREB degradation'+index,'vdCREB',km=['kdCREB'],Rct=['CREB'+index])
                
                #SSA_builder.SSA_MA_meanfield('eVIP',indx,xlen,indy,ylen,'kcouple')

        #Add mixing functions
        
        # END REACTIONS
        #stringSSAmodel = SSAmodel.serialize()
        #print stringSSAmodel
        
        stoch = SSAmodel

    return fn,stoch,state_names,param_names

def main():
    
    #Creates SSA version of model.
    ODEmodel,SSAmodel,state_names,param_names=FullModel(True)
    trajectories = stk.stochkit(SSAmodel,job_id='coupled2',t=500,number_of_trajectories=1,increment=0.1)
    StochPopEval = stk.StochPopEval(trajectories,state_names,param_names,vol,EqCount)
    StochPopEval.PlotPop('p',traces=False)
    bl_object=StochPopEval.bl_obj('p')
    """
    pl.plot(trajectories[0][:,8])
    pl.plot(trajectories[0][:,19])
    pl.plot(trajectories[0][:,30])
    pl.plot(trajectories[0][:,41])
    pl.plot(trajectories[0][:,52])
    pl.plot(trajectories[0][:,63])
    pl.plot(trajectories[0][:,74])
    pl.plot(trajectories[0][:,85])
    pl.plot(trajectories[0][:,96])
    """

    #StochPopEval.waveletplot(bl_object,fignum=2)
    pl.show()
    pdb.set_trace()

    pdb.set_trace()

if __name__ == "__main__":
    main()  















