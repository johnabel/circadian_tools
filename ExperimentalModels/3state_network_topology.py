# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:19:48 2014

@author: john abel

"""
from __future__ import division
import numpy  as np
import casadi as cs
import stochkit_resources as stk
import circadiantoolbox as ctb
import modelbuilder as mb
import random
import pdb
import matplotlib.pyplot as plt


EqCount = 3
ParamCount = 13
modelversion='3state'

cellcount=3

period = 23.7000

vol=50

randomy0 = False

y0in=np.array([2.1, 0.34, 0.42])
        
param = [0.83,0/50,     100/50.0, 0.5,
         4.0, 21.05/50, 50/50.0,   25/50.0, 
         0.417, 58.35/50, 6.5/50, 0.417, 
         0.5]

def ODEmodel():
    #==================================================================
    #State variable definitions
    #==================================================================
    M    = cs.ssym("M")
    Pc   = cs.ssym("Pc")
    Pn   = cs.ssym("Pn")
    
    #for Casadi
    y = cs.vertcat([M, Pc, Pn])
    
    # Time Variable
    t = cs.ssym("t")
    
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    
    vs0 = cs.ssym('vs0')
    light = cs.ssym('light')
    alocal = cs.ssym('alocal')
    couplingStrength = cs.ssym('couplingStrength')
    n = cs.ssym('n')
    vm = cs.ssym('vm')
    k1 = cs.ssym('k1')
    km = cs.ssym('km')
    ks = cs.ssym('ks')
    vd = cs.ssym('vd')
    kd = cs.ssym('kd')
    k1_ = cs.ssym('k1_')
    k2_ = cs.ssym('k2_')    

    paramset = cs.vertcat([vs0, light, alocal, couplingStrength,
                           n,   vm,    k1,     km, 
                           ks,  vd,    kd,     k1_, 
                           k2_])
                        
    
    #===================================================================
    # Model Equations
    #===================================================================
    
    ode = [[]]*EqCount

    def pm_prod(Pn, K, v0, n, light, a, M, Mi):
        vs = v0+light+a*(M-Mi)
        return (vs*K**n)/(K**n + Pn**n)
    
    def pm_deg(M, Km, vm):
        return vm*M/(Km+M)
    
    def Pc_prod(ks, M):
        return ks*M
    
    def Pc_deg(Pc, K, v):
        return v*Pc/(K+Pc)
    
    def Pc_comp(k1,k2,Pc,Pn):
        return -k1*Pc + k2*Pn
    
    def Pn_comp(k1,k2,Pc,Pn):
        return k1*Pc - k1*Pn

    #Rxns
    ode[0] = (pm_prod(Pn, k1, vs0, n, light, alocal, M, M) - pm_deg(M,km,vm))
    ode[1] = Pc_prod(ks,M) - Pc_deg(Pc,kd,vd) + Pc_comp(k1_,k2_,Pc,Pn)
    ode[2] = Pn_comp(k1_,k2_,Pc,Pn)

    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","SDS16")
    
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
        SSA_builder.SSA_MM('M degradation'+index,'vm',
                           km=['km'],Rct=['M'+index])
        SSA_builder.SSA_MA_tln('Pc translation'+index, 'Pc'+index,
                               'ks','M'+index)
        SSA_builder.SSA_MM('Pc degradation'+index,'vd',
                           km=['kd'],Rct=['Pc'+index])
        
        #The complexing four:
        SSA_builder.SSA_MA_cytonuc('cyto/nuc'+index,'Pc'+index,
                                   'Pn'+index,'k1_','k2_')
       
    #coupled mRNA production       
    for tocell in range(cellcount):
        #loops for all cells accumulating their input
        avg = '0'
        mcount = 0
        for fromcell in range(cellcount):
            if adjacency[fromcell,tocell]!= 0:
                #The Coupling Part
                mcount = mcount+1
                avg = avg+'+M_'+str(fromcell)+'_0'

        
        weight = 1.0/mcount
        rxn=stk.Reaction(name='per production for '+str(tocell),
                        products={'M_'+str(tocell)+'_0':1},
                        propensity_function=('std::max(0.0,(vs0+alocal('+avg+')*'
                                    +str(weight)+'-M'+str(tocell)+'_0))'+ 
                                    '*pow(k1,n)/(pow(k1,n)+pow(M'+str(tocell)+'_0,n)))'
                                    ),
                        annotation='')    
        SSAmodel.addReaction(rxn)
    return SSAmodel,state_names,param_names
    

if __name__=='__main__':
    
    print param
    
    ODEsolC = ctb.CircEval(ODEmodel(), param, y0in)
    sol = ODEsolC.intODEs_sim(tf=100)
    tsol = ODEsolC.ts
    plt.plot(sol)
    plt.show()
    
    tf=100
    inc = 0.05
    adjacency = np.array([[1,1,0],[0,1,1],[0,0,1]])#, 
                              #[0,0.5,0.5,0], 
                              #[0,0,0.5,0],
                              #[0,0,0,0]])
    
    SSAnet,state_names,param_names = SSAnetwork(ODEmodel(),
                                                    y0in,param,adjacency)
    
    pdb.set_trace()
    traj = stk.stochkit(SSAnet,job_id='3state',t=tf,
                               number_of_trajectories=1,increment=inc,
                               seed=11)
    plt.plot(traj[:,1:])
                   
    pass








