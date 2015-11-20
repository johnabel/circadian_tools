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
import gillespy

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


class tyson_ssa(gillespy.Model):
    """
    Here, as a test case, we run a simple two-state oscillator (Novak & Tyson 
    2008) as an example of a stochastic reaction system.
    """
    def __init__(self, 
                 param=[0.05, 1.0, 4.0, 0.05, 1.0, 0.05, 1.0, 0.1, 2.0], 
                 volume = 200,
                 timespan = np.linspace(0,500,501)):
        """
        """
        gillespy.Model.__init__(self, name="tyson-2-state", volume=volume)
        self.timespan(timespan)
        # =============================================
        # Define model species, initial values, parameters, and volume
        # =============================================    
        
        # Parameter values  for this biochemical system are given in 
        # concentration units. However, stochastic systems must use population
        # values. For example, a concentration unit of 0.5mol/(L*s)
        # is multiplied by a volume unit, to get a population/s rate
        # constant. Thus, for our non-mass action reactions, we include the 
        # parameter "vol" in order to convert population units to concentration
        # units. Volume here = 300.

        k1 = gillespy.Parameter(name='k1', expression = param[0])
        Kd = gillespy.Parameter(name='Kd', expression = param[1])
        P = gillespy.Parameter(name='P', expression = param[2])
        kdx = gillespy.Parameter(name='kdx', expression = param[3])
        ksy = gillespy.Parameter(name='ksy', expression = param[4])
        kdy = gillespy.Parameter(name='kdy', expression = param[5])
        k2 = gillespy.Parameter(name='k2', expression = param[6])
        Km = gillespy.Parameter(name='Km', expression = param[7])
        KI = gillespy.Parameter(name='KI', expression = param[8])
        volm = gillespy.Parameter(name='volume', expression = volume)
        
        
        
        self.add_parameter([k1, Kd, P, kdx, ksy, kdy, k2, Km, KI, volm])
        
        # Species
        # Initial values of each species (concentration converted to pop.)
        X = gillespy.Species(name='X', initial_value=int(0.65609071*volume))
        Y = gillespy.Species(name='Y', initial_value=int(0.85088331*volume))
        self.add_species([X, Y])
        
        # =============================================  
        # Define the reactions within the model
        # =============================================  
        
        # creation of X:
        rxn1 = gillespy.Reaction(name = 'X production',
                        reactants = {},
                        products = {X:1},
                        propensity_function = 'volume*(k1*pow(Kd,P))/(pow(Kd,P)+pow(Y,P)/pow(volume,P))')
        
        # degradadation of X:
        rxn2 = gillespy.Reaction(name = 'X degradation',
                    reactants = {X:1},
                    products = {},
                    rate = kdx)
        
        # creation of Y:
        rxn3 = gillespy.Reaction(name = 'Y production',
                    reactants = {X:1},
                    products = {X:1, Y:1},
                    rate = ksy)
        
        # degradation of Y:
        rxn4 = gillespy.Reaction(name = 'Y degradation',
                    reactants = {Y:1},
                    products = {},
                    rate = kdy)
            
        # nonlinear Y term:
        rxn5 = gillespy.Reaction(name = 'Y nonlin',
                    reactants = {Y:1},
                    products = {},
                    propensity_function = 'volume*(Y/volume)/(Km + (Y/volume)+KI*(Y*Y)/(volume*volume))')
        
        self.add_reaction([rxn1,rxn2,rxn3,rxn4,rxn5])
    
    
    
    
    
    
