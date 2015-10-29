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

# Constants and Equation Setup
EqCount = 2
ParamCount = 9

param = [0.05, 1., 4., 0.05, 1., 0.05, 1., 0.1, 2.]
y0in = [0.6560881 ,   0.85088577]
period = 60.81

def model():
    #==================================================================
    #setup of symbolics
    #==================================================================
    x = cs.SX.sym("x")
    y = cs.SX.sym("y")
    
    sys = cs.vertcat([x,y])
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    
    k1  = cs.SX.sym("k1")
    Kd  = cs.SX.sym("Kd")
    P   = cs.SX.sym("P")
    kdx = cs.SX.sym("kdx")
    ksy = cs.SX.sym("ksy")
    kdy = cs.SX.sym("kdy")
    k2  = cs.SX.sym("k2")
    Km  = cs.SX.sym("Km")
    KI  = cs.SX.sym("KI")
    
    paramset = cs.vertcat([k1,Kd,P,kdx,ksy,kdy,k2,Km,KI])
    
    # Time
    t = cs.SX.sym("t")
    
    
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
    
    
    
    
    
    
    
    
    
    
    
