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
modelversion='threest'

cellcount=191

period = 23.7000
couplingstr = 0.005 #default is 1

vol=40

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
        return k1*Pc - k2*Pn

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
def ssa_desync(fn,y0in,param,adjacency=np.zeros([cellcount,cellcount])):
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
    y0in_pop = []
    
    np.fill_diagonal(adjacency,1)

    #coupling section===========================
    for indx in range(cellcount):
        index = '_'+str(indx)+'_0'
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
    for indx in range(cellcount):
            
        index = '_'+str(indx)+'_0'
        
        #Coupled terms - - -------------------------------------------------
        #loops for all cells accumulating their input
        avg = '0'
        mcount = 0
        for fromcell in range(cellcount):
            if adjacency[fromcell,indx]!= 0:
                #The Coupling Part
                mcount = mcount+1
                avg = avg+'+M_'+str(fromcell)+'_0'

        
        weight = 1.0/mcount
        #FIXES PARAMETERS FROM DETERMINISTIC TO STOCHASTIC VALUES
        if (str(SSA_builder.pvaldict['vs0']) ==
                        SSA_builder.SSAmodel.listOfParameters['vs0'].expression):
                SSA_builder.SSAmodel.setParameter('vs0',
                                SSA_builder.SSAmodel.listOfParameters['vs0'].expression+'*('+str(SSA_builder.vol)+')')
        if (str(SSA_builder.pvaldict['k1']) ==
                        SSA_builder.SSAmodel.listOfParameters['k1'].expression):
                SSA_builder.SSAmodel.setParameter('k1',
                                SSA_builder.SSAmodel.listOfParameters['k1'].expression+'*('+str(SSA_builder.vol)+')')
        
        if (str(SSA_builder.pvaldict['alocal']) ==
                        SSA_builder.SSAmodel.listOfParameters['alocal'].expression):
                SSA_builder.SSAmodel.setParameter('alocal',
                                SSA_builder.SSAmodel.listOfParameters['alocal'].expression+'*('+str(SSA_builder.vol)+')')
        #####
        #Adds reaction
        rxn=stk.Reaction(name='Cell'+str(indx)+'_Reaction0',
                         reactants={},
                        products={'M_'+str(indx)+'_0':1},
                        propensity_function=('std::max(0.0,(vs0+alocal*(('+avg+')*'
                                    +str(weight)+'-M_'+str(indx)+'_0)))'+ 
                                    '*pow(k1,n)/(pow(k1,n)+pow(Pn_'+str(indx)+'_0,n))'),annotation='')#
   
        SSAmodel.addReaction(rxn)
        #-------------------------------------------------------------------
        
        # REACTIONS
        SSA_builder.SSA_MM('Cell'+str(indx)+'_Reaction1','vm',
                           km=['km'],Rct=['M'+index])
        
        SSA_builder.SSA_MA_tln('Cell'+str(indx)+'_Reaction2', 'Pc'+index,
                               'ks','M'+index)
        
        SSA_builder.SSA_MM('Cell'+str(indx)+'_Reaction3','vd',
                           km=['kd'],Rct=['Pc'+index])
        
        #The complexing four:
        SSA_builder.SSA_MA_cytonuc('Cell'+str(indx)+'_Reaction4','Pc'+index,
                                   'Pn'+index,'k1_','k2_')
        

    return SSAmodel,state_names,param_names
    
def ssa_resync(fn,y0in_desync,param,adjacency):
    """
    This is the network-level SSA model, with coupling. Call with:
        SSAcoupled,state_names,param_names = SSAmodelC(ODEmodel(),y0in,param)
    
    To uncouple the model, set adjacency matrix to zeros    
    """
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = []
    state_names=[]
    

    #coupling section===========================
    for indx in range(cellcount):
        index = '_'+str(indx)+'_0'
        #loops to include all species, normally this is the only line needed without index
        species_array = species_array + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                        for i in xrange(EqCount)]
        state_names = state_names + [fn.inputSX(cs.DAE_X)[i].getDescription()+index
                        for i in xrange(EqCount)]       
    
    y0in_pop = y0in_desync

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
        
        #Coupled terms - - -------------------------------------------------
        #loops for all cells accumulating their input
        avg = 'M'+index
        mcount = 1

        for fromcell in range(cellcount):
            if adjacency[fromcell,indx]!= 0:
                #The Coupling Part
                mcount = mcount+couplingstr
                avg = avg+'+M_'+str(fromcell)+'_0*'+str(couplingstr)


        weight = 1.0/mcount
        #FIXES PARAMETERS FROM DETERMINISTIC TO STOCHASTIC VALUES
        if (str(SSA_builder.pvaldict['vs0']) ==
                        SSA_builder.SSAmodel.listOfParameters['vs0'].expression):
                SSA_builder.SSAmodel.setParameter('vs0',
                                SSA_builder.SSAmodel.listOfParameters['vs0'].expression+'*('+str(SSA_builder.vol)+')')
        if (str(SSA_builder.pvaldict['k1']) ==
                        SSA_builder.SSAmodel.listOfParameters['k1'].expression):
                SSA_builder.SSAmodel.setParameter('k1',
                                SSA_builder.SSAmodel.listOfParameters['k1'].expression+'*('+str(SSA_builder.vol)+')')
        
        if (str(SSA_builder.pvaldict['alocal']) ==
                        SSA_builder.SSAmodel.listOfParameters['alocal'].expression):
                SSA_builder.SSAmodel.setParameter('alocal',
                                SSA_builder.SSAmodel.listOfParameters['alocal'].expression+'*('+str(SSA_builder.vol)+')')
        #####
        #Adds reaction
        rxn=stk.Reaction(name='Cell'+str(indx)+'_Reaction0',
                         reactants={},
                        products={'M_'+str(indx)+'_0':1},
                        propensity_function=('std::max(0.0,(vs0+alocal*(('+avg+')*'
                                    +str(weight)+'-M_'+str(indx)+'_0)))'+ 
                                    '*pow(k1,n)/(pow(k1,n)+pow(Pn_'+str(indx)+'_0,n))'),annotation='')#
   
        SSAmodel.addReaction(rxn)
        #-------------------------------------------------------------------
        
        # REACTIONS
        SSA_builder.SSA_MM('Cell'+str(indx)+'_Reaction1','vm',
                           km=['km'],Rct=['M'+index])
        
        SSA_builder.SSA_MA_tln('Cell'+str(indx)+'_Reaction2', 'Pc'+index,
                               'ks','M'+index)
        
        SSA_builder.SSA_MM('Cell'+str(indx)+'_Reaction3','vd',
                           km=['kd'],Rct=['Pc'+index])
        
        #The complexing four:
        SSA_builder.SSA_MA_cytonuc('Cell'+str(indx)+'_Reaction4','Pc'+index,
                                   'Pn'+index,'k1_','k2_')
        

    return SSAmodel,state_names,param_names
    

if __name__=='__main__':
    
    print param
    
    ODEsolC = ctb.CircEval(ODEmodel(), param, y0in)
    sol = ODEsolC.intODEs_sim(y0in,100)
    tsol = ODEsolC.ts
    plt.plot(tsol,sol)
    plt.show()
    
    tf=10
    inc = 0.05
    cellcount=3
    adjacency = np.array([[1,1,0],[0,1,1],[0,0,1]])#, 
                              #[0,0.5,0.5,0], 
                              #[0,0,0.5,0],
                              #[0,0,0,0]])
    
    SSAnet,state_names,param_names = ssa_desync(ODEmodel(),
                                                    y0in,param,adjacency)
    
    pdb.set_trace()
    traj = stk.stochkit(SSAnet,job_id='threestate',t=tf,
                               number_of_trajectories=1,increment=inc,
                               seed=11)
    plt.plot(traj[:,1:])
                   
    pass








