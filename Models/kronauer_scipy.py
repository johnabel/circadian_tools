# -*- coding: utf-8 -*-
"""
Created on Jan 31 12:01:35 2016

@author: John H. Abel

# model for Kronauer Forger Jewett 1999 Quantifying Human Circadian Pacemaker Response to Brief, Extended, and Repeated Light Stimuli over the Phototopic Range
http://journals.sagepub.com/doi/abs/10.1177/074873049901400609
"""
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

# Constants and Equation Setup
param_P = [0.13, 0.55, 1./3, 24.2]
param_L = [0.16, 0.013, 19.875, 0.6, 9500] #alpha0, beta, G, p, I0
y0in = [ -0.17,  -1.22, 0.5]

def tyson_model(ys, t0, param_P, param_L, I):
    #can change parameters by feeding in a different set to the function
    #this function takes in parameters and puts out the dy1/dt and dy2/dt 
    #this function then fed to the 

    #assign parameter values to names
    alpha0, beta, G, p, I0 = param_L
    mu, k, q, taux = param_P
    
    # light input fcn
    def alpha(t):
        return alpha0*(I(t)/I0)**p
    
        # output drive onto the pacemaker for the prompt response model
    def Bhat(t):
        return G*alpha(t)*(1-n)
    
    # circadian modulation of photic sensitivity
    def B(t):
        return (1-0.4*x)*(1-0.4*xc)*Bhat(t)
    
    # current state
    x, xc, n = ys
    
    ode = np.zeros([3])
    ode[0] = (np.pi/12)*(xc +mu*(x/3. + (4/3.)*x**3 - (256/105.)*x**7)+B(t))
    ode[1] = (np.pi/12)*(q*B(t)*xc - ((24/(0.99729*taux))**2 + k*B(t))*x)
    ode[2] = 60*(alpha(t)*(1-n)-beta*n)
    
    return ode #gives values of derivatives
    
    
    
if __name__ == "__main__":

    t = np.arange(0,300,0.1) #time steps from 0 to 10 with 100 intervals
    
    def I(t):
        return 0
    
    #integration
    y = integrate.odeint(tyson_model, y0in, t, args=(param_P, param_L, I))
    # output = integrate(dy/dt, initial value, time points, extra args)
    
    # plotting
    plt.plot(t,y)
    plt.show()
