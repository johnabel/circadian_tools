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
import pdb
import matplotlib.pyplot as plt
import PlotOptions as plo


EqCount = 3
ParamCount = 13
modelversion='three-state'

period = 23.7000

vol=40

y0in=np.array([2.1, 0.34, 0.42])
        
param = [0.83,0/50,     100/50.0, 0.5,
         4.0, 21.05/50, 50/50.0,   25/50.0, 
         0.417, 58.35/50, 6.5/50, 0.417, 
         0.5]


def ODEmodel():
    #==================================================================
    #State variable definitions
    #==================================================================
    M    = cs.SX.sym("M")
    Pc   = cs.SX.sym("Pc")
    Pn   = cs.SX.sym("Pn")
    
    #for Casadi
    y = cs.vertcat([M, Pc, Pn])
    
    # Time Variable
    t = cs.SX.sym("t")
    
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    
    vs0 = cs.SX.sym('vs0')
    light = cs.SX.sym('light')
    alocal = cs.SX.sym('alocal')
    couplingStrength = cs.SX.sym('couplingStrength')
    n = cs.SX.sym('n')
    vm = cs.SX.sym('vm')
    k1 = cs.SX.sym('k1')
    km = cs.SX.sym('km')
    ks = cs.SX.sym('ks')
    vd = cs.SX.sym('vd')
    kd = cs.SX.sym('kd')
    k1_ = cs.SX.sym('k1_')
    k2_ = cs.SX.sym('k2_')    

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
    
def SSAmodel(fn,y0in,param):
    """
    This is the network-level SSA model, with coupling. Call with:
    SSAcoupled,state_names,param_names = SSAmodelC(ODEmodel(),y0in,param)
    By default, there is no coupling in this model.    
    """
    
    #Converts concentration to population
    y0in_pop = (vol*y0in).astype(int)
    
    #collects state and parameter array to be converted to species and parameter objects,
    #makes copies of the names so that they are on record
    species_array = [fn.inputExpr(cs.DAE_X)[i].getName()
                    for i in xrange(EqCount)]
    param_array   = [fn.inputExpr(cs.DAE_P)[i].getName()
                    for i in xrange(ParamCount)]
    
    state_names = [fn.inputExpr(cs.DAE_X)[i].getName()
                        for i in xrange(EqCount)]  
    param_names   = [fn.inputExpr(cs.DAE_P)[i].getName()
                        for i in xrange(ParamCount)]

    #creates SSAmodel class object    
    SSAmodel = stk.StochKitModel(name=modelversion)
    SSA_builder = mb.SSA_builder(species_array,param_array,y0in_pop,param,SSAmodel,vol)
    

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
    
    
    # REACTIONS
    rxn0=stk.Reaction(name='Reaction0',
                         reactants={},
                        products={'M':1},
                        propensity_function=(
                        'vs0*pow(k1,n)/(pow(k1,n)+pow(Pn,n))'),annotation='')
   
    SSAmodel.addReaction(rxn0)    
    
    SSA_builder.SSA_MM('Reaction1','vm',
                       km=['km'],Rct=['M'])
    
    SSA_builder.SSA_MA_tln('Reaction2', 'Pc',
                           'ks','M')
    
    SSA_builder.SSA_MM('Reaction3','vd',
                       km=['kd'],Rct=['Pc'])

    SSA_builder.SSA_MA_cytonuc('Reaction4','Pc',
                               'Pn','k1_','k2_')
        

    return SSAmodel,state_names,param_names
    

if __name__=='__main__':
    
    # runs and compares one stochastic trajectory with deterministic solution
import matplotlib.gridspec as gridspec

odes = ctb.Oscillator(ODEmodel(), param, y0in)
odes.calc_y0()
odes.limit_cycle()
odes.first_order_sensitivity()
odes.find_prc()

tf=500.
inc = 0.1

SSAnet,state_names,param_names = SSAmodel(ODEmodel(),
                                                y0in,param)
                                                
traj = stk.stochkit(SSAnet,job_id='threestate',t=tf,
                           number_of_trajectories=300,increment=inc,
                           seed=11)



plo.PlotOptions()
plt.figure(figsize=(3.5*2,2.62))
gs = gridspec.GridSpec(1,2)

ax0=plt.subplot(gs[0,0])
ax0.plot(tsol,sol,label=['M','C','N'])
ax0.set_xlabel('Time, hr')
ax0.set_ylabel('SV Concentration')

ax1=plt.subplot(gs[0,1])
seval = stk.StochEval(traj,state_names,param_names,vol)
seval.PlotAvg('M')
ax1.set_xlabel('Time, hr')
ax1.set_ylabel('SV Count')

plt.tight_layout(**plo.layout_pad)
plt.show()
    pass








