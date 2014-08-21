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
import casadi as cs
import numpy as np

# Sensitivities
abstol = 1e-11
reltol = 1e-9
maxnumsteps = 40000

# Constants and Equation Setup
EqCount = 2
ParamCount = 9

param = [0.05, 1., 4., 0.05, 1., 0.05, 1., 0.1, 2.]
y0in = np.ones(EqCount+1)

def model():
    #==================================================================
    #setup of symbolics
    #==================================================================
    x = cs.ssym("x")
    y = cs.ssym("y")
    
    sys = cs.vertcat([x,y])
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    
    k1  = cs.ssym("k1")
    Kd  = cs.ssym("Kd")
    P   = cs.ssym("P")
    kdx = cs.ssym("kdx")
    ksy = cs.ssym("ksy")
    kdy = cs.ssym("kdy")
    k2  = cs.ssym("k2")
    Km  = cs.ssym("Km")
    KI  = cs.ssym("KI")
    
    paramset = cs.vertcat([k1,Kd,P,kdx,ksy,kdy,k2,Km,KI])
    
    # Time
    t = cs.ssym("t")

    
    #===================================================================
    # set up the ode system
    #===================================================================
    
    ode = [[]]*EqCount #initializes vector
    ode[0] = k1*(Kd**P)/((Kd**P) + (y**P)) - kdx*x
    ode[1] = ksy*x - kdy*y - k2*y/(Km + y + KI*y**2)
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=sys, p=paramset),
                        cs.daeOut(ode=ode))
                        
    fn.setOption("name","2state")
    
    return fn
    
    
    
    
    
    
    
    
    
    
    