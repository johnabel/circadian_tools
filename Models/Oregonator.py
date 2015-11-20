# A model of a simple mass-action chemical limit cycle oscillator,
# originally proposed in R. J. Field and R. M. Noyes, J. Chem. Phys.,
# 60, 1877 (1973)
#
# Stochastic version using Gillespie's algorithm is presented in his
# original paper, D. T. Gillespie, J. Phys. Chem. 81, 2340-2361 (1977).
#

import numpy as np
import casadi as cs
import gillespy

# Aux. parameters, according to Gillespie (p. 2358)
y1s = 500.
y2s = 1000.
y3s = 2000.
p1s = 2000.
p2s = 50000.

# [c1x1, c2, c3x2, c4, c5x3], according to Gillespie's paper
param = [
    p1s/y2s,
    p2s/(y1s*y2s),
    (p1s + p2s)/y1s,
    2*p1s/y1s**2,
    (p1s + p2s)/y3s]


y0in = [2866.2056016028037, 585.492592883311, 6874.38507145332]
period= 0.7033292578982892

EqCount = 3
ParamCount = 5

def model():
    """ Create an ODE casadi SXFunction """

    # State Variables
    Y1 = cs.SX.sym("Y1")
    Y2 = cs.SX.sym("Y2")
    Y3 = cs.SX.sym("Y3")

    y = cs.vertcat([Y1, Y2, Y3])

    # Parameters
    c1x1 = cs.SX.sym("c1x1")
    c2   = cs.SX.sym("c2")
    c3x2 = cs.SX.sym("c3x2")
    c4   = cs.SX.sym("c4")
    c5x3 = cs.SX.sym("c5x3")

    symparamset = cs.vertcat([c1x1, c2, c3x2, c4, c5x3])

    # Time
    t = cs.SX.sym("t")


    # ODES
    ode = [[]]*EqCount
    ode[0] = c1x1*Y2 - c2*Y1*Y2 + c3x2*Y1 - c4*Y1**2
    ode[1] = -c1x1*Y2 - c2*Y1*Y2 + c5x3*Y3
    ode[2] = c3x2*Y1 - c5x3*Y3
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name", "Oregonator")
    
    return fn





class oregonator_ssa(gillespy.Model):
    """
    Here, as a test case, we run a simple two-state oscillator (Novak & Tyson 
    2008) as an example of a stochastic reaction system.
    """
    def __init__(self, 
                 param=param, 
                 volume = 0.1,
                 timespan = np.linspace(0,10,10./period*48)):
        """
        """
        gillespy.Model.__init__(self, name="oregonator", volume=volume)
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

        c1x1    = gillespy.Parameter(name='c1x1', expression = param[0])
        c2      = gillespy.Parameter(name='c2', expression = param[1])
        c3x2    = gillespy.Parameter(name='c3x2', expression = param[2])
        c4      = gillespy.Parameter(name='c4', expression = param[3]*2)
        c5x3    = gillespy.Parameter(name='c5x3', expression = param[4])
        
        
        
        self.add_parameter([c1x1, c2, c3x2, c4, c5x3])
        
        # Species
        # Initial values of each species (concentration converted to pop.)
        Y1 = gillespy.Species(name='Y1', initial_value=int(y0in[0]*volume))
        Y2 = gillespy.Species(name='Y2', initial_value=int(y0in[1]*volume))
        Y3 = gillespy.Species(name='Y3', initial_value=int(y0in[2]*volume))
        self.add_species([Y1, Y2, Y3])
        

        

        rxn1 = gillespy.Reaction(name = 'Y2_to_Y1',
                    reactants = {Y2:1},
                    products = {Y1:1},
                    rate = c1x1)
        
        rxn2 = gillespy.Reaction(name = 'Y1_Y2_2_deg',
                    reactants = {Y1:1, Y2:1},
                    products = {},
                    rate = c2)
        
        rxn3 = gillespy.Reaction(name = 'Y1_Y3_form',
                    reactants = {Y1:1},
                    products = {Y1:2, Y3:1},
                    rate = c3x2)
                    
        rxn4 = gillespy.Reaction(name = 'Y1_deg',
                    reactants = {Y1:2},
                    products = {Y1:1},
                    rate = c4)

        rxn5 = gillespy.Reaction(name = 'Y3_to_Y2',
                    reactants = {Y3:1},
                    products = {Y2:1},
                    rate = c5x3)
                    

        
        self.add_reaction([rxn1,rxn2,rxn3,rxn4,rxn5])



if __name__ == "__main__":

    import circadiantoolbox as ctb

    oreg = ctb.Oscillator(model(), param, y0in)
    dsol = oreg.int_odes(10,numsteps=1000)
    #oreg.calc_y0()
    #oreg.limit_cycle()
    #oreg.first_order_sensitivity()
    #oreg.find_prc()
    
    ssa = oregonator_ssa(param=param)
    trajectories = ssa.run(number_of_trajectories=1)
    
    plt.plot(oreg.ts,dsol)
    plt.plot(trajectories[0][:,0], trajectories[0][:,1:]/ssa.volume)





