# Model of a hysteresis-driven negative-feedback oscillator, taken from
# B. Novak and J. J. Tyson, "Design principles of biochemical
# oscillators," Nat. Rev. Mol. Cell Biol., vol. 9, no. 12, pp. 981-91,
# Dec. 2008.
# Figure 3, equation 8.

import casadi as cs

# Model Constants
paramset = [0.05, 1., 4., 0.05, 1., 0.05, 1., 0.1, 2.]
y0in = [  0.6560881 ,   0.85088577,  60.81367555]

NEQ = 2
NP = 9

def model():
    #======================================================================
    # Variable Assignments
    #======================================================================
    X = cs.SX.sym("X")
    Y = cs.SX.sym("Y")

    y = cs.vertcat([X,Y])
    
    #======================================================================
    # Parameter Assignments
    #======================================================================
    k1  = cs.SX.sym("k1")
    Kd  = cs.SX.sym("Kd")
    P   = cs.SX.sym("P")
    kdx = cs.SX.sym("kdx")
    ksy = cs.SX.sym("ksy")
    kdy = cs.SX.sym("kdy")
    k2  = cs.SX.sym("k2")
    Km  = cs.SX.sym("Km")
    KI  = cs.SX.sym("KI")
        
    symparamset = cs.vertcat([k1,Kd,P,kdx,ksy,kdy,k2,Km,KI])
    # Time Variable
    t = cs.SX.sym("t")
    
    
    ode = [[]]*NEQ
    ode[0] = k1*(Kd**P)/((Kd**P) + (Y**P)) - kdx*X
    ode[1] = ksy*X - kdy*Y - k2*Y/(Km + Y + KI*Y**2)
    ode = cs.vertcat(ode)
    
    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=symparamset),
                       cs.daeOut(ode=ode))

    fn.setOption("name","Tyson 2 State Model")
    
    return fn

def create_class():
    from CommonFiles.pBase import pBase
    return pBase(model(), paramset, y0in)


if __name__ == "__main__":
    from CommonFiles.pBase import pBase
    test_base = pBase(model(), paramset, y0in)