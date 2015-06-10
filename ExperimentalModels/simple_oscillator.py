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
import minepy as mp
import itertools
from scoop import futures
from scipy import stats


EqCount = 8
ParamCount = 4
modelversion='simple'

cellcount=4

period = 2
couplingstr = 00.1 #default is 1

vol=50

randomy0 = False

y0in=3*(np.random.random([cellcount*2])-0.5)
        
param = [0.5,1,1,1]

def ODEmodel():
    #==================================================================
    #State variable definitions
    #==================================================================
    S1a   = cs.ssym("S1a")
    S2a   = cs.ssym("S2a")
    S1b   = cs.ssym("S1b")
    S2b   = cs.ssym("S2b")
    S1c   = cs.ssym("S1c")
    S2c   = cs.ssym("S2c")
    S1d   = cs.ssym("S1d")
    S2d   = cs.ssym("S2d")
    
    #for Casadi
    y = cs.vertcat([S1a,S2a,S1b,S2b,S1c,S2c,S1d,S2d])
    
    # Time Variable
    t = cs.ssym("t")
    
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    
    p1 = cs.ssym('p1')
    p2 = cs.ssym('p2')
    p3 = cs.ssym('p3')  
    p4 = cs.ssym('p4')
    
    paramset = cs.vertcat([p1,p2,p3,p4])
                        
    
    #===================================================================
    # Model Equations
    #===================================================================
    

    ode = [[]]*EqCount
    
    #Rxns
    ode[0] = (1/p1)*(S1a - (1/3)*S1a**3 - S2a) + couplingstr*(S1b-S1a)
    ode[1] = S1a + p1*S2a
    ode[2] = (1/p1)*(S1b - (1/3)*S1b**3 - S2b) + couplingstr*(S1a-S1b)  + couplingstr*(S1c-S1b)
    ode[3] = S1b + p1*S2b
    ode[4] = (1/p1)*(S1c - (1/3)*S1c**3 - S2c) + couplingstr*(S1b-S1c) + couplingstr*(S1d-S1c)
    ode[5] = S1c + p1*S2c
    ode[6] = (1/p1)*(S1d - (1/3)*S1d**3 - S2d) + couplingstr*(S1c-S1d)
    ode[7] = S1d + p1*S2d
    
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","simple")
    
    return fn
    

if __name__=='__main__':
    
    print param
    
    # deterministic solution    
    ODEsolC = ctb.CircEval(ODEmodel(), param, y0in)
    sol = ODEsolC.intODEs_sim(y0in,50)
    tsol = ODEsolC.ts
    plt.plot(tsol,sol[:,[0,2]])
    plt.plot(tsol,np.abs(sol[:,2]-sol[:,0]),color='k')
    
    #mic
    sph = 1
    sph_obs = int(np.round(1/(tsol[1]-tsol[0])))
    ts_data = sol[:,[0,2,4,6]]
    ts_pre_fix = np.copy(ts_data)
    ts_data = np.zeros([np.floor(sph*len(ts_data)/sph_obs),cellcount])
    for i in range(len(ts_data)):
        ts_data[i,:] = np.sum(ts_pre_fix[
                int(i*sph_obs/sph):int((1+i)*sph_obs/sph),:],axis=0)    

    mic = np.zeros([cellcount**2,4])
    aa = [range(cellcount),range(cellcount)]
    inds = list(itertools.product(*aa))
    for i in xrange(len(inds)):
        [c1,c2] = inds[i]
        ts1 = ts_data[:,c1]
        ts2 = ts_data[:,c2]
        mic[i,:] = [c1, c2, mp.minestats(ts1,ts2)['mic'], 
                            stats.linregress(ts1,ts2)[2]]
    
    hmp = np.zeros([cellcount,cellcount])
    hmp2 = np.copy(hmp)
    for i in range(len(mic)):
        ind1 = mic[i][0]
        ind2 = mic[i][1]
        hmp[ind1,ind2] = mic[i][2]
        hmp2[ind1,ind2] = mic[i][3]
    np.fill_diagonal(hmp,0)
    
    plt.figure(2)
    plt.pcolormesh(hmp2)
    plt.colorbar()
    
    plt.figure(3)
    from matplotlib import gridspec
    
    gs = gridspec.GridSpec(2,2)
    ax0 = plt.subplot(gs[0,0])
    plt.plot(sol[:,0],sol[:,2])
    ax1 = plt.subplot(gs[1,0])
    plt.plot(sol[:,0],sol[:,4])
    ax2 = plt.subplot(gs[0,1])
    plt.plot(sol[:,0],sol[:,6])
    ax3 = plt.subplot(gs[1,1])
    plt.plot(sol[:,2],sol[:,4])
    
    plt.show()







