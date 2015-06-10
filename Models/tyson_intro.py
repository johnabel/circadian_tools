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
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

# Constants and Equation Setup
EqCount = 2
ParamCount = 9


def tyson_model(ys,t0,param):
    #can change parameters by feeding in a different set to the function
    #this function takes in parameters and puts out the dy1/dt and dy2/dt 
    #this function then fed to the 

    #assign parameter values to names
    [k1,Kd,P,kdx,ksy,kdy,k2,Km,KI] = param
    
    #assign state values to names
    [y1, y2] = ys
    
    
    #set up the set of ODEs
    dysdt = np.zeros([2])
    dysdt[0] = k1*(Kd**P)/((Kd**P) + (y2**P)) - kdx*y1
    dysdt[1] = ksy*y1 - kdy*y2 - k2*y2/(Km + y2 + KI*y2**2)
    
    return dysdt #gives values of derivatives
    
    
    
if __name__ == "__main__":
    # if this file is the one we are running, run the following procedure
    
    param = [0.05, 1., 4., 0.05, 1., 0.05, 1., 0.1, 2.]
    y0in = [0.6560881 ,   0.85088577]
    period = 60.81
    
    
    t = np.arange(0,300,0.1) #time steps from 0 to 10 with 100 intervals
    
    #integration
    y = integrate.odeint(tyson_model, y0in, t, args=(param,))
    # output = integrate(dy/dt, initial value, time points, extra args)
    
    # plotting
    plt.plot(t,y)
    plt.show()