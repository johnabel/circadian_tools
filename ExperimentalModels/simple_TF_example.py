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
import gillespy as gsp
import matplotlib.pyplot as plt

modelversion = 'simple_model'

# constants and equations setup, trying a new method
neq = 2
npa  = 5

param = [  1.0, 1.0, 1.0,   0, 0
           ]
y0in = np.array([  3.25  ,  13.8125])
period = 30.27

def simple_model_ode():
    
    # For two oscillators
    x1 = cs.ssym('x1')
    x2 = cs.ssym('x2')
    
    state_set = cs.vertcat([x1, x2])

    # Parameter Assignments
    K0  = cs.ssym('K0')
    K1  = cs.ssym('K1')
    K2  = cs.ssym('K2')
    amp = cs.ssym('amp')
    freq= cs.ssym('freq')
    
    param_set = cs.vertcat([K0, K1,K2,amp,freq])

    # Time
    t = cs.ssym('t')

    # oscillators
    ode = [[]]*neq
    
    ode[0] = K0 - K2*(x1+amp*np.sin(2*np.pi*freq*t))
    ode[1] = K1*(x1+amp*np.sin(2*np.pi*freq*t)) - K2*x2

    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t,x=state_set,p=param_set), 
            cs.daeOut(ode=ode))

    fn.setOption("name","simple_model")

    return fn


class simple_gillespy_model(gsp.Model):
    """
    This is the simple model done in the gillespy format. Takes:
        
        ode_model = simple_model_ode(): the casadi function that defines the 
                    model
        parameter_values = param: the usual parameters, adjustable
        y0_values = y0: the initial conditions, in terms of CONCENTRATION
        volumve = 100: the stochastic volume parameter
    """
    
    def __init__(self, ode_model=simple_model_ode(),
                 parameter_values=param, y0_values=y0in,
                 volume=500):
        
        #setup, attaching the ode model
        self.ode_model = ode_model
        self.neq = neq
        self.npa = npa
        
        
        # so we can pull from casadi
        ylabels = [ode_model.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(self.neq)]
        plabels = [ode_model.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(self.npa)]
        
        #Initialize the model
        gsp.Model.__init__(self,name='simple')
        
        # Parameters
        for i in xrange(self.npa):
            self.add_parameter(gsp.Parameter(name=plabels[i],
                                             expression=parameter_values[i]))                 
        # Species
        for i in xrange(self.neq):
            self.add_species(gsp.Species(name=ylabels[i],
                                initial_value = int(volume*y0_values[i])))
        # Reactions




if __name__ == "__main__":

    """
    Test suite for the model. We will run with both casadi deterministic 
    and gillespy stochastic simulations.
    """

    import circadiantoolbox as ctb
    
    param[-2]=0.5
    param[-1] = 10
    # create deterministic circadian object
    simple_odes = ctb.CircEval(simple_model_ode(), param, y0in)
    simple_odes.burnTransient_sim()
    simple_odes.intODEs_sim(10)
    
    simple_odes.sol[:,0] = (simple_odes.sol[:,0] +param[-2]*np.sin(2*np.pi*param[-1]*simple_odes.ts))
    # plot deterministic solutions
    plt.plot(simple_odes.ts,simple_odes.sol); plt.show()

