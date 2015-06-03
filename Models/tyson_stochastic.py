# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:01:35 2014

@author: John H. Abel

# Model of a hysteresis-driven negative-feedback oscillator, taken from
# B. Novak and J. J. Tyson, "Design principles of biochemical
# oscillators," Nat. Rev. Mol. Cell Biol., vol. 9, no. 12, pp. 981-91,
# Dec. 2008.
# Figure 3, equation 8.
"""
import numpy  as np
import casadi as cs
import pdb
import stochkit_resources as stk
import modelbuilder as mb
import pylab as pl
import circadiantoolbox as ctb

modelversion = 'Tyson_Two_State'

# Sensitivities
abstol = 1e-11
reltol = 1e-9
maxnumsteps = 40000

# Constants and Equation Setup
EqCount = 2
ParamCount = 7

param = np.array([2., 20., 1., 0.005, 0.05, 0.1, 1.])
y0in = np.array([ 0.65609071,  0.85088331])

vol = 150

def FullModel(Stoch=False):
    
    # Variable Assignments
    X = cs.ssym("X")
    Y = cs.ssym("Y")

    sys = cs.vertcat([X,Y]) # vector version of y
    
    # Parameter Assignments
    P  = cs.ssym("P")
    kt = cs.ssym("kt")
    kd = cs.ssym("kd")
    a0 = cs.ssym("a0")
    a1 = cs.ssym("a1")
    a2 = cs.ssym("a2")
    kdx= cs.ssym("kdx")
    
    paramset = cs.vertcat([P, kt, kd, a0, a1, a2, kdx])
    
    # Time
    t = cs.ssym("t")
    
    ode = [[]]*EqCount
    ode[0] = 1 / (1 + Y**P) - kdx*X
    ode[1] = kt*X - kd*Y - Y/(a0 + a1*Y + a2*Y**2)
    
    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=sys, p=paramset), cs.daeOut(ode=ode))
                        
    fn.setOption("name","2state")
    

    #==================================================================
    # Stochastic Model Portion
    #==================================================================
    
    if Stoch==True:
        print 'Now converting model to StochKit XML format...'
        
        #Converts concentration to population
        y0in_pop = (vol*y0in).astype(int)
        
        #collects state and parameter array to be converted to species and parameter objects
        species_array = [fn.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(EqCount)]
        param_array   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(ParamCount)]
        
        #Names model
        SSAmodel = stk.StochKitModel(name=modelversion)
        
        #creates SSAmodel class object
        SSA_builder = mb.SSA_builder(species_array,param_array,y0in_pop,param,SSAmodel,vol)
        
        # REACTIONS
        SSA_builder.SSA_tyson_x('x prod nonlinear term','X','Y','P')
        SSA_builder.SSA_MA_deg('x degradation','X','kdx')
        
        SSA_builder.SSA_MA_tln('y creation', 'Y', 'kt', 'X')
        SSA_builder.SSA_MA_deg('y degradation, linear','Y','kd')
        SSA_builder.SSA_tyson_y('y nonlinear term','Y','a0','a1','a2')
        
        # END REACTIONS

    state_names = [fn.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(EqCount)]  
    param_names   = [fn.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(ParamCount)]

    return fn,SSAmodel,state_names,param_names

def main():
    #Creates SSA version of model.
    ODEmodel,SSAmodel,state_names,param_names=FullModel(True)
    
    ODEsol = ctb.CircEval(ODEmodel, param, y0in)
    y0LC = ODEsol.burnTransient_sim(tf = 000, numsteps = 1000)
    sol = ODEsol.intODEs_sim(y0LC,10,numsteps=1000)
    tsol = ODEsol.ts 
    
    pdb.set_trace()
    trajectories = stk.stochkit(SSAmodel,job_id='tyson',t=10,number_of_trajectories=100,increment=0.1)
    
    StochEval = stk.StochEval(trajectories,state_names,param_names,vol)
    StochEval.PlotAvg('X',color='red',traces=True)
    StochEval.PlotAvg('Y',color='black',traces=True)
    

    
    
    
    pl.plot(tsol,vol*sol)

    pl.show()
    pdb.set_trace()

if __name__ == "__main__":
    main()  
    
    
    
    
    
    
    
    
    
    
    