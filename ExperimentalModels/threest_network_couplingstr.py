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


    

if __name__=='__main__':
    
    print param
    
    ODEsolC = ctb.Oscillator(ODEmodel(), param, y0in)
    sol = ODEsolC.int_odes(100)
    tsol = ODEsolC.ts
    plt.plot(tsol,sol)
    









