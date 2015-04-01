"""
Created on Tue Jan 13 13:01:35 2014

@author: John H. Abel

Model of an electrochemical oscillator based on Wickramasinghe, 
Mrugacz, and Kiss (doi:10.1039/c1cp21429b) equations 7-9.

This is also the first model build using native gillespy, so let's see how
it goes. - jha
"""

# common imports
from __future__ import division

# python packages
import numpy as np
import casadi as cs
import gillespy as gl
import matplotlib.pyplot as plt

modelversion = 'gonze_model'

# constants and equations setup, trying a new method
neq = 8
npa  = 18

param = [  0.7,    1,    4, 0.35,    1,  0.7, 0.35,   
             1,  0.7, 0.35,    1, 0.35,    1,    1,
           0.4,    1,  0.5,    0
           ]
y0in = np.array([ 0.05069219,  0.10174506,  2.28099242,  0.01522458,  0.05069219,
        0.10174506,  2.28099242,  0.01522458])
period = 30.27

def gonze_model_ode():
    
    # For two oscillators
    X1 = cs.ssym('X1')
    Y1 = cs.ssym('Y1')
    Z1 = cs.ssym('Z1')
    V1 = cs.ssym('V1')
    
    X2 = cs.ssym('X2')
    Y2 = cs.ssym('Y2')
    Z2 = cs.ssym('Z2')
    V2 = cs.ssym('V2')
    
    state_set = cs.vertcat([X1, Y1, Z1, V1,
                            X2, Y2, Z2, V2])

    # Parameter Assignments
    v1  = cs.ssym('v1')
    K1  = cs.ssym('K1')
    n   = cs.ssym('n')
    v2  = cs.ssym('v2')
    K2  = cs.ssym('K2')
    k3  = cs.ssym('k3')
    v4  = cs.ssym('v4')
    K4  = cs.ssym('K4')
    k5  = cs.ssym('k5')
    v6  = cs.ssym('v6')
    K6  = cs.ssym('K6')
    k7  = cs.ssym('k7')
    v8  = cs.ssym('v8')
    K8  = cs.ssym('K8')
    vc  = cs.ssym('vc')
    Kc  = cs.ssym('Kc')
    K   = cs.ssym('K')
    L   = cs.ssym('L')
    
    param_set = cs.vertcat([v1,K1,n,v2,K2,k3,v4,K4,k5,v6,K6,k7,v8,K8,vc,Kc,K,L])

    # Time
    t = cs.ssym('t')

    # oscillators
    ode = [[]]*neq
    
    ode[0] = v1*K1**n/(K1**n + Z1**n) \
             - v2*X1/(K2+X1) \
             +vc*K*((V1+V2)/2)/(Kc +K*(V1+V2)/2) + L
    ode[1] = k3*X1 - v4*Y1/(K4+Y1)
    ode[2] = k5*Y1 - v6*Z1/(K6+Z1)
    ode[3] = k7*X1 - v8*V1/(K8+V1)
    
    ode[4] = v1*K1**n/(K1**n + Z2**n) \
             - v2*X2/(K2+X2) \
             +vc*K*((V1+V2)/2)/(Kc +K*(V1+V2)/2) + L
    ode[5] = k3*X2 - v4*Y2/(K4+Y2)
    ode[6] = k5*Y2 - v6*Z2/(K6+Z2)
    ode[7] = k7*X2 - v8*V2/(K8+V2)


    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t,x=state_set,p=param_set), 
            cs.daeOut(ode=ode))

    fn.setOption("name","gonze_model")

    return fn






if __name__ == "__main__":

    """
    Test suite for the model. We will run with both casadi deterministic 
    and gillespy stochastic simulations.
    """

    import circadiantoolbox as ctb

    # create deterministic circadian object
    gonze_odes = ctb.CircEval(gonze_model_ode(), param, y0in)
    gonze_odes.burnTransient_sim()
    gonze_odes.intODEs_sim(200)
    
    # plot deterministic solutions
    plt.plot(gonze_odes.ts,gonze_odes.sol[:,(0,4)]); plt.show()

