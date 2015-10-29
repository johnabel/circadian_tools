# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:53:42 2014
 
@author: John H. Abel

This file will be my repository of classes and functions to call when
solving models. Any general methods will be here.

"""

from __future__ import division
import cPickle as pickle
import numpy as np
import casadi as cs
import pylab as pl
import matplotlib.pyplot as plt
import jha_utilities as jha
import pdb
from scipy import signal
from scipy.interpolate import splrep, splev, UnivariateSpline


class Oscillator(object):
    """
    This circadian oscillator class is for deterministic ODE simulations.
    """
    
    def __init__(self, model, param, y0=None, period_guess=24.):
        """
        Setup the required information.
        ----
        model : casadi.sxfunction
            model equations, sepecified through an integrator-ready
            casadi sx function
        paramset : iterable
            parameters for the model provided. Must be the correct length.
        y0 : optional iterable
            Initial conditions, specifying where d(y[0])/dt = 0
            (maximum) for the first state variable.
        """
        self.model = model
        self.modifiedModel()
        self.neq = self.model.input(cs.DAE_X).size()
        self.npa = self.model.input(cs.DAE_P).size()
        
        self.model.init()
        self.param = param

        self.jacp = self.model.jacobian(cs.DAE_P,0); self.jacp.init()
        self.jacy = self.model.jacobian(cs.DAE_X,0); self.jacy.init()
        
        self.ylabels = [self.model.inputExpr(cs.DAE_X)[i].getName()
                        for i in xrange(self.neq)]
        self.plabels = [self.model.inputExpr(cs.DAE_P)[i].getName()
                        for i in xrange(self.npa)]
        
        self.pdict = {}
        self.ydict = {}
        
        for par,ind in zip(self.plabels,range(0,self.npa)):
            self.pdict[par] = ind
            
        for par,ind in zip(self.ylabels,range(0,self.neq)):
            self.ydict[par] = ind
        
        self.param_dict = {}
        for i in range(self.npa):
            self.param_dict[self.plabels[i]] = self.param[i]
        
        self.inverse_ydict = {v: k for k, v in self.ydict.items()}
        self.inverse_pdict = {v: k for k, v in self.pdict.items()}
        
        self.intoptions = {
            'y0tol'            : 1E-3,
            'bvp_ftol'         : 1E-10,
            'bvp_abstol'       : 1E-12,
            'bvp_reltol'       : 1E-10,
            'sensabstol'       : 1E-11,
            'sensreltol'       : 1E-9,
            'sensmaxnumsteps'  : 80000,
            'sensmethod'       : 'staggered',
            'transabstol'      : 1E-4,
            'transreltol'      : 1E-4,
            'transmaxnumsteps' : 5000,
            'lc_abstol'        : 1E-11,
            'lc_reltol'        : 1E-9,
            'lc_maxnumsteps'   : 40000,
            'lc_res'           : 200,
            'int_abstol'       : 1E-8,
            'int_reltol'       : 1E-6,
            'int_maxstepcount' : 40000
                }

        if y0 is None:
            self.y0 = 5*np.ones(self.NEQ+1)
            self.calcY0(25*period_guess)
        else: self.y0 = np.asarray_chkfinite(y0)     
    
    # shortcuts
    def _phi_to_t(self, phi): return phi*self.y0[-1]/(2*np.pi)
    def _t_to_phi(self, t): return (2*np.pi)*t/self.y0[-1]
    
    
    def modifiedModel(self):
        """
        Creates a new casadi model with period as a parameter, such that
        the model has an oscillatory period of 1. Necessary for the
        exact determinination of the period and initial conditions
        through the BVP method. (see Wilkins et. al. 2009 SIAM J of Sci
        Comp)
        """

        pSX = self.model.inputExpr(cs.DAE_P)
        T = cs.SX.sym("T")
        pTSX = cs.vertcat([pSX, T])
        
        t = self.model.inputExpr(cs.DAE_T)
        sys = self.model.inputExpr(cs.DAE_X)
        ode = self.model.outputExpr()[0]*T
        
        self.modlT = cs.SXFunction(
            cs.daeIn(t=t,x=sys,p=pTSX),
            cs.daeOut(ode=ode)
            )

        self.modlT.setOption("name","T-shifted model")  
        
        
    def int_odes(self, tf, y0=None, numsteps=10000, return_endpt=False):
        """
        This function integrates the ODEs until well past the transients. 
        This uses Casadi's simulator class, C++ wrapped in swig. Inputs:
            tf          -   the final time of integration.
            numsteps    -   the number of steps in the integration is the second argument
        """
        if y0==None: y0 = self.y0
        
        self.integrator = cs.Integrator('cvodes',self.model)
        
        #Set up the tolerances etc.
        self.integrator.setOption("abstol", self.intoptions['int_abstol'])
        self.integrator.setOption("reltol", self.intoptions['int_reltol'])
        self.integrator.setOption("max_num_steps", self.intoptions['int_maxstepcount'])
        self.integrator.setOption("tf",tf)
        
        #Let's integrate
        self.integrator.init()
        self.ts = np.linspace(0,tf, numsteps, endpoint=False)
        self.simulator = cs.Simulator(self.integrator, self.ts)
        self.simulator.init()
        self.simulator.setInput((y0[:]),cs.INTEGRATOR_X0)
        self.simulator.setInput(self.param,cs.INTEGRATOR_P)
        self.simulator.evaluate()
        	
        sol = self.simulator.output().toArray().T
                
        if return_endpt==True:
            return sol[-1]
        else:
            return sol
    
    def burn_trans(self,tf=500.):
        """
        integrate the solution until tf, return only the endpoint
        """
        self.y0 = self.int_odes(tf, return_endpt=True)
        
    
    
    def solve_bvp(self, method='casadi', backup='casadi'):
        """
        Chooses between available solver methods to solve the boundary
        value problem. Backup solver invoked in case of failure
        """
        available = {
            #'periodic' : self.solveBVP_periodic,
            'casadi'   : self.solve_bvp_casadi
            ,'scipy'    : self.solve_bvp_scipy
            }

        y0in = np.array(self.y0)

        return available[method]()
        try: return 0        
        except Exception:
            self.y0 = np.array(y0in)
            try: return available[backup]()
            except Exception:
                self.y0 = y0in
                self.approxY0(tol=1E-4)
                return available[method]()
    
    def solve_bvp_scipy(self, root_method='hybr'):
        """
        Use a scipy optimize function to optimize the BVP function
        """

        # Make sure inputs are the correct format
        paramset = list(self.param)

        
        # Here we create and initialize the integrator SXFunction
        self.bvpint = cs.Integrator('cvodes',self.modlT)
        self.bvpint.setOption('abstol',self.intoptions['bvp_abstol'])
        self.bvpint.setOption('reltol',self.intoptions['bvp_reltol'])
        self.bvpint.setOption('tf',1)
        self.bvpint.setOption('disable_internal_warnings', True)
        self.bvpint.setOption('fsens_err_con', True)
        self.bvpint.init()

        def bvp_minimize_function(x):
            """ Minimization objective. X = [y0,T] """
            # perhaps penalize in try/catch?
            if np.any(x < 0): return np.ones(3)
            self.bvpint.setInput(x[:-1], cs.INTEGRATOR_X0)
            self.bvpint.setInput(paramset + [x[-1]], cs.INTEGRATOR_P)
            self.bvpint.evaluate()
            out = x[:-1] - self.bvpint.output().toArray().flatten()
            out = out.tolist()

            self.modlT.setInput(x[:-1], cs.DAE_X)
            self.modlT.setInput(paramset + [x[-1]], 2)
            self.modlT.evaluate()
            out += self.modlT.output()[0].toArray()[0].tolist()
            return np.array(out)
        
        from scipy.optimize import root

        options = {}

        root_out = root(bvp_minimize_function, np.append(self.y0, self.T),
                        tol=self.intoptions['bvp_ftol'],
                        method=root_method, options=options)

        # Check solve success
        if not root_out.status:
            raise RuntimeError("bvpsolve: " + root_out.message)

        # Check output convergence
        if np.linalg.norm(root_out.qtf) > self.intoptions['bvp_ftol']*1E4:
            raise RuntimeError("bvpsolve: nonconvergent")

        # save output to self.y0
        self.y0 = root_out.x[:-1]
        self.T = root_out.x[-1]
        
    def solve_bvp_casadi(self):
        """
        Uses casadi's interface to sundials to solve the boundary value
        problem using a single-shooting method with automatic differen-
        tiation.
        
        Related to PCSJ code. 
        """

        self.bvpint = cs.Integrator('cvodes',self.modlT)
        self.bvpint.setOption('abstol',self.intoptions['bvp_abstol'])
        self.bvpint.setOption('reltol',self.intoptions['bvp_reltol'])
        self.bvpint.setOption('tf',1)
        self.bvpint.setOption('disable_internal_warnings', True)
        self.bvpint.setOption('fsens_err_con', True)
        self.bvpint.init()
        
        # Vector of unknowns [y0, T]
        V = cs.MX.sym("V",self.neq+1)
        y0 = V[:-1]
        T = V[-1]
        param = cs.vertcat([self.param, T])
        self.bvpint.setInput(x[:-1], cs.INTEGRATOR_X0)
        self.bvpint.setInput(paramset + [x[-1]], cs.INTEGRATOR_P)
        yf = self.bvpint.call(cs.integratorIn(x0=y0,p=param))[0]
        fout = self.modlT.call(cs.daeIn(t=T, x=y0,p=param))[0]
        
        # objective: continuity
        obj = (yf - y0)**2  # yf and y0 are the same ..i.e. 2 ends of periodic fcn
        obj.append(fout[0]) # y0 is a peak for state 0, i.e. fout[0] is slope state 0
        
        #set up the matrix we want to solve
        F = cs.MXFunction([V],[obj])
        F.init()
        guess = np.append(self.y0,24)
        solver = cs.ImplicitFunction('kinsol',F)
        solver.setOption('abstol',self.intoptions['bvp_ftol'])
        solver.setOption('strategy','linesearch')
        solver.setOption('exact_jacobian', False)
        solver.setOption('pretype', 'both')
        solver.setOption('use_preconditioner', True)
        solver.setOption('constraints', (2,)*(self.neq+1))
        solver.setOption('linear_solver_type', 'dense')
        solver.init()
        solver.setInput(guess)
        solver.evaluate()
        
        sol = solver.output().toArray().squeeze()
        
        self.y0 = sol[:-1]
        self.T = sol[-1]


    def dydt(self,y):
        """
        Function to calculate model for given y.
        """
        try:
            out = []
            for yi in y:
                assert len(yi) == self.NEQ
                self.model.setInput(yi,cs.DAE_X)
                self.model.setInput(self.param,cs.DAE_P)
                self.model.evaluate()
                out += [self.model.output().toArray().flatten()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.model.setInput(y,cs.DAE_X)
            self.model.setInput(self.param,cs.DAE_P)
            self.model.evaluate()
            return self.model.output().toArray().flatten()

        
    def dfdp(self,y,p=None):
        """
        Function to calculate model jacobian for given y and p.
        """
        if p is None: p = self.param

        try:
            out = []
            for yi in y:
                assert len(yi) == self.NEQ
                self.jacp.setInput(yi,cs.DAE_X)
                self.jacp.setInput(p,cs.DAE_P)
                self.jacp.evaluate()
                out += [self.jacp.output().toArray()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.jacp.setInput(y,cs.DAE_X)
            self.jacp.setInput(p,cs.DAE_P)
            self.jacp.evaluate()
            return self.jacp.output().toArray()

        
    def dfdy(self,y,p=None):
        """
        Function to calculate model jacobian for given y and p.
        """
        if p is None: p = self.param
        try:
            out = []
            for yi in y:
                assert len(yi) == self.NEQ
                self.jacy.setInput(yi,cs.DAE_X)
                self.jacy.setInput(p,cs.DAE_P)
                self.jacy.evaluate()
                out += [self.jacy.output().toArray()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.jacy.setInput(y,cs.DAE_X)
            self.jacy.setInput(p,cs.DAE_P)
            self.jacy.evaluate()
            return self.jacy.output().toArray()

    
    def approx_y0_T(self, tout=300, burn_trans=True, tol=1E-3):
        """ 
        Approximates the period and y0 to the given tol, by integrating,
        creating a spline representation, and comparing the max values using
        state 0.        
        """
        from jha_utilities import roots
        
        if burn_trans==True:
            self.burn_trans()
        
        states = self.int_odes(tout)
        ref_state = states[:,0]
        time = self.ts
        
        # create a spline representation of the first state, k=4 so deriv k=3
        spl = UnivariateSpline(time, ref_state, k=4, s=0)
        time_spl = np.arange(0,tout,tol)
        
        #finds roots of splines
        roots = spl.derivative(n=1).roots() #der of spline

        # gives y0 and period by finding second deriv.        
        peaks_of_roots = np.where(spl.derivative(n=2)(roots) < 0)
        peaks = roots[peaks_of_roots]
        periods = np.diff(peaks)

        if sum(np.diff(periods)) < tol:
            self.T = np.mean(periods)
            
            #calculating the y0 for each state witha  cubic spline
            self.y0 = np.zeros(self.neq)
            for i in range(self.neq):
                spl = UnivariateSpline(time, states[:,i], k=3, s=0)
                self.y0[i] = spl(peaks[0])
                
        else:
            self.T = -1

    def corestationary(self,guess=None):
        """
        find stationary solutions that satisfy ydot = 0 for stability
        analysis. 
        """
        guess=None
        if guess is None: guess = np.array(self.y0)
        else: guess = np.array(guess)
        y = self.model.inputExpr(cs.DAE_X)
        t = self.model.inputExpr(cs.DAE_T)
        p = self.model.inputExpr(cs.DAE_P)
        ode = self.model.outputExpr()
        fn = cs.SXFunction([y,t,p],ode)
        kfn = cs.ImplicitFunction('kinsol',fn)
        abstol = 1E-10
        kfn.setOption("abstol",abstol)
        kfn.setOption("constraints",(2,)*self.neq) # constain using kinsol to >0
        kfn.setOption("linear_solver_type","dense")
        kfn.setOption("exact_jacobian",True)
        kfn.setOption("u_scale",(100/guess).tolist())
        kfn.setOption("disable_internal_warnings",True)
        kfn.init()
        kfn.setInput(self.param,2)
        kfn.setInput(guess)
        kfn.evaluate()
        y0out = kfn.output().toArray()

        if any(np.isnan(y0out)):
            raise RuntimeError("findstationary: KINSOL failed to find \
                               acceptable solution")
        
        self.ss = y0out.flatten()
        
        if np.linalg.norm(self.dydt(self.ss)) >= abstol or any(y0out <= 0):
            raise RuntimeError("findstationary: KINSOL failed to reach \
                               acceptable bounds")
              
        self.eigs = np.linalg.eigvals(self.dfdy(self.ss))

    def find_stationary(self, guess=None):
        """
        Find the stationary points dy/dt = 0, and check if it is a
        stable attractor (non oscillatory).
        Parameters
        ----------
        guess : (optional) iterable
            starting value for the iterative solver. If empty, uses
            current value for initial condition, y0.
        Returns
        -------
        +0 : Fixed point is not a steady-state attractor
        +1 : Fixed point IS a steady-state attractor
        -1 : Solution failed to converge
        """
        try:
            self.corestationary(guess)
            if all(np.real(self.eigs) < 0): return 1
            else: return 0

        except Exception: return -1


if __name__ == "__main__":
    
    from Models.tyson_model import model, param, EqCount
    
    lap = jha.laptimer()
    
    # initialize with 1s
    tyson = Oscillator(model(), param, np.ones(EqCount))
    print tyson.y0, 'setup time = %0.3f' %(lap() )
    
    # find a spot on LC
    tyson.y0 = np.ones(EqCount)
    tyson.burn_trans()
    print tyson.y0, 'y0 burn time = %0.3f' %(lap() ) 
    
    # or find a max and the approximate period
    tyson.y0 = np.ones(EqCount)
    tyson.approx_y0_T()
    print tyson.y0, tyson.T, 'y0 approx time = %0.3f' %(lap() )
    
    # find the period and y0 using a BVP solution
    tyson.y0 = np.ones(EqCount)
    tyson.solve_bvp(method='scipy')
    print tyson.y0, tyson.T, 'y0 scipy bvp time = %0.3f' %(lap() )
    
    # find the period and y0 using a BVP solution
    tyson.y0 = np.ones(EqCount)
    tyson.solve_bvp(method='casadi')
    print tyson.y0, tyson.T, 'y0 casadi bvp time = %0.3f' %(lap() )

    # find steady state soln (fixed pt)
    tyson.find_stationary()
    print tyson.ss, 'stationary time = %0.3f' %(lap() )


