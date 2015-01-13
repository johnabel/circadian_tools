"""
Written by jh abel, first begun on 24-11-2014.

Collocation module for parameter identification.

"""

# Python imports
from __future__ import division
from itertools import izip

# Import modules
import numpy as np
import casadi as cs
import pdb

#my modules
from jha_utilities import fnlist, spline

# Class for performing collocation analysis
class Collocation:
    """
    Will solve for parameters for a given model, minimizing distance
    between model solution and experimental time-series data.
    """

    def __init__(self, model=None):

        # Set-up the collocation options
        
        self.ydata = {} # data to be fit
        self.ytime = {} # time for the data
        self.yvars = {} # variances for the data
        
        self.collocation_method = 'radau'
        self.deg = 4 # degree of interpolating polynomial
        self.points_per_interval = 1
        self.nk = 10
        self.tf = 24.0 # single period covers everything
        
        # Set up some default placeholders for bounds
        self.PARMAX = 1E+2
        self.PARMIN = 1E-3
        self.XMAX   = 1E+2
        self.XMIN   = 1E-3
        
        # Set up solver options
        
        
        # Add the model to this object
        if model:
            self.AttachModel(model)



    def AttachModel(self, model):
        """
        Attaches a CasADi model to this object, sets the model up for
        its eventual evaluation.
        Call like: AttachModel(model)
        """
        self.model = model()
        self.model.init()
        self.EqCount = self.model.input(cs.DAE_X).size()
        self.ParamCount  = self.model.input(cs.DAE_P).size()
        
        # Some of this syntax is borrowed from Peter St. John.
        self.ylabels = [self.model.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(self.EqCount)]
        self.plabels = [self.model.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(self.ParamCount)]
        
        self.pdict = {}
        self.ydict = {}
        
        for par,ind in zip(self.plabels,range(0,self.ParamCount)):
            self.pdict[par] = ind
            
        for par,ind in zip(self.ylabels,range(0,self.EqCount)):
            self.ydict[par] = ind

        # Model now attached to object along with its contents.
        print "Model attached."



    def AttachExperiment(self, state, time, trajectories, variances=None):
        """
        Attaches experimental data to the method. Submit data as an array. 
        If variances are not give, they will be assumed to be +/-5%.
        """

        if variances is None:
            variances = 0.05*trajectories
        
        #ensure correct sizes
        assert(len(time)==len(trajectories)==len(variances))
        
        self.ydata[self.ylabels[state]]=trajectories
        self.ytime[self.ylabels[state]]=time
        self.yvars[self.ylabels[state]]=variances

        print("Single state "+self.ylabels[state]+" data attached.")



    def DataMatrix(self,data,time,variances=None):
        """
        Calls AttachExperiment for each data trajectory that we have. Must
        be assembled as a matrix with column # being state.
        """
        # confirm time series matches up with data
        assert(len(data[:,0])==len(time))
        
        #confirms data for each SV
        assert(len(data[0,:]==self.EqCount))
        
        #variances for data, if not giver
        if variances is None:
            variances = 0.05*data

        for i in range(len(data[0,:])):
            #each column (each state)
            self.AttachExperiment(i,time,data[:,i],variances[:,i])

        self.Amatrix = np.eye(self.EqCount) 
        # sets up relationships for ys. for example, if each trajectory is a 
        # single state in order, use the identity matrix. If something is a 
        # sum of two states, you'll need two in the rows. For the data matrix
        # case, should always be eye.
        print "Data matrix attached."
    
 
   
    def setup(self):
        """
        Sets up the collocation problem as in the example in casadi. I will
        explain this as it happens.
        
        Casadi explanation:
            
        """
        
        # dimensions
        nx = self.EqCount
        nu = 0 # no control states
      
        tau_root = collocation_point_sets(self.deg, 
                                           method = self.collocation_method)
        # convenience and to match casadi implementation
        deg = self.deg
        nk = self.nk
        tf = self.tf
        
        # size of finite elements 
        h = tf/nk
        
        # Coefficients of the collocation equation
        C = np.zeros((deg+1,deg+1))
        
        # coefficients of the continuity equation
        D = np.zeros(deg+1)
        
        # Collocation point
        tau = cs.ssym("tau")
        
        # All collocation time points
        T = np.zeros((nk, deg+1))
        for k in range(nk):
            for j in range(deg+1):
                T[k,j] = h*(k+tau_root[j])
                
        # For all collocation points
        for j in range(deg+1):
            # Construct Lagrange polynomials to get the polynomial 
            # basis at the collocation point
            L = 1
            for r in range(deg+1):
                if r != j:
                    L *= (tau-tau_root[r])/(tau_root[j]-tau_root[r])
            lfcn = cs.SXFunction([tau],[L])
            lfcn.init()
  
            # Evaluate the polynomial at the final time to get the 
            # coefficients of the continuity equation
            lfcn.setInput(1.0)
            lfcn.evaluate()
            D[j] = lfcn.output()

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            for r in range(deg+1):
                lfcn.setInput(tau_root[r])
                lfcn.setFwdSeed(1.0)
                lfcn.evaluate(1,0)
                C[j,r] = lfcn.fwdSens()
            
            # I believe tg is a time grid
            tg = np.array(tau_root)*h
            for k in range(nk):
                if k == 0:
                    tgrid = tg
                else:
                    tgrid = np.append(tgrid,tgrid[-1]+tg)
        
        # save to self for when we execute the actual collocation
        self.tgrid = tgrid
        self.C = C
        self.D = D
        self.h = h
        
        # weights for the integration of function along lagrange
        # polynomial
        self.E = [0., 0.118463, 0.239314, 0.284445, 0.239314, 0.118463]

        # Set up PARMAX and PARMIN variables. Check if they are a
        # vector, if not resize to number of parameters.
        try:
            assert len(self.PARMAX) == len(self.PARMIN) == self.ParamCount, \
            "Parameter bounds not correct length"
        except TypeError:
            self.PARMAX = [self.PARMAX] * self.ParamCount
            self.PARMIN = [self.PARMIN] * self.ParamCount
            
        # Set up XMAX and XMIN variables. Check if they are a
        # vector, if not resize to number of state variables
        try:
            assert len(self.XMAX) == len(self.XMIN) == self.EqCount, \
            "State variable bounds not correct length"
        except TypeError:
            self.XMAX = [self.XMAX] * self.EqCount
            self.XMIN = [self.XMIN] * self.EqCount
    
 
   
    def set_init_vals(self, x_init=None, p_init=None):
        """
        Sets up initial x (state) and p (parameter) values to feed into the 
        solver. 
        """
        
        # makes spline out of time and data
        sfactor = 5E-3
        self.sy = fnlist([])
        for t, y in izip(self.ytime.values(), self.ydata.values()):
            # Duplicate the first entry.
            tvals = list(t)
            tvals.append(t[0]+self.tf)
            yvals = list(y)
            yvals.append(y[0])
            self.sy += [spline(tvals, yvals, sfactor)]
        
        # Interpolate from measurement data
        if x_init is None: x_init = self.trajectory_estimation()
        else:
            assert x_init.shape == (len(self.tgrid), self.EqCount), \
                "Shape Mismatch"
            assert (np.all(x_init < self.XMAX) &
                    np.all(x_init > self.XMIN)), "Bounds Error"

        self.x_init = x_init
        
        # Interpolate from slopes of measurement data
        # generates a good initial parameter guess
        if p_init is not None: self.p_init = p_init
        else:
            self.p_init = self.ParameterEstimation()
        
        pdb.set_trace()
    

    
    def solve(self, x_init=None, p_init=None):
        self.set_init_vals(x_init=x_init, p_init=p_init)
    
 
   
    def trajectory_estimation(self):
        """
        Generates x_init from y values. Just used PSJ function, minor updates.
        """
        
        from scipy.optimize import minimize
        node_ts = self.tgrid.reshape((self.nk, self.deg+1))[:,0]

        # Loop through start of each finite element (node_ts), solve for
        # optimum state variables from the state constructure
        # measurement functions.
        xopt = []

        bounds = [[xmin, xmax] for xmin, xmax in
                  zip(self.XMIN,self.XMAX)]

        options = {'disp' : False}

        for t in node_ts:
            # Initial guess for optimization. If its the first point,
            # start with [1]*NEQ, otherwise use the result from the
            # previous finite element.
            
            if xopt == []: iguess = np.ones(self.EqCount)
            else: iguess = xopt[-1]

            y = self.sy(t)

            # Inefficient but simple - might need to rethink.
            # Distance weight: to improve identifiability issues, return
            # the solution closest to the previous iterate (or [1]'s)
            dw = 1E-5
            def min_func(x):
                dist = np.linalg.norm(iguess - x)
                ret = sum((self.Amatrix.dot(x) - y)**2) + dw * dist
                return ret
            xopt += [minimize(min_func, iguess, bounds=bounds,
                              method='L-BFGS-B', options=options)['x']]

        x_init_course = np.array(xopt)

        # Resample x_init to get all the needed points
        self.sx = fnlist([])
        
        def flat_factory(x):
            """ Protects against nan's resulting from flat trajectories
            in spline curve """
            return lambda t, d: (np.array([x] * len(t)) if d is 0 else
                                 np.array([0] * len(t)))

        for x in np.array(x_init_course).T:
            tvals = list(node_ts)
            tvals.append(tvals[0] + self.tf)
            xvals = list(x)
            xvals.append(x[0])
            
            if np.linalg.norm(xvals - xvals[0]) < 1E-8:
                # flat profile
                self.sx += [flat_factory(xvals[0])]

            else: self.sx += [spline(tvals, xvals, 0)]

        x_init = self.sx(self.tgrid,0).T

        below = np.where(x_init < self.XMIN)
        above = np.where(x_init > self.XMAX)

        for t_ind, state in zip(*below):
            x_init[t_ind, state] = self.XMIN[state]

        for t_ind, state in zip(*above):
            x_init[t_ind, state] = self.XMAX[state]

        return x_init
    
    
    def parameter_estimation(self):
        """
        Again, borrow from PSJ. Gets a decent estimate of the parameters
        initial values by looking at slopes. I think.
        """
        
        
        
# utility functions below this line ====
        
def collocation_point_sets(collocation_order, method='radau'):
    """
    Returns the necessary collocation points for solving.
    """

    if method == 'legendre':
        coll_pts_dict = {1:[0,0.500000], 2:[0,0.211325,0.788675],
                         3:[0,0.112702,0.500000,0.887298], 
                         4:[0,0.069432,0.330009,0.669991,0.930568],
                         5:[0,0.046910,0.230765,0.500000,0.769235,0.953090]}
    if method == 'radau':
        coll_pts_dict = {1:[0,1.000000], 2:[0,0.333333,1.000000],
                         3:[0,0.155051,0.644949,1.000000], 
                         4:[0,0.088588,0.409467,0.787659,1.000000],
                         5:[0,0.057104,0.276843,0.583590,0.860240,1.000000]}
                         
    return coll_pts_dict[collocation_order]
    
        
def interpolating_polynomial( ):
    pass



if __name__ == "__main__":

    import circadiantoolbox as ctb
    from Models.tyson2statemodel import model, param, y0in, period
    
    # say we are using collocation on the tyson model
    test = Collocation(model=model)

    # Simulate Tyson model
    ODEsol = ctb.CircEval(model(), param, y0in)
    traj_sim = ODEsol.intODEs_sim(y0in,1000)
    time_sim = ODEsol.ts
    ODEsol.find_period(time_sim, traj_sim)
    ODEsol.find_y0PLC(time_sim, traj_sim)
     
    #attach some data from the Tyson model
    time_steps = [0,1000,1300,4000,7000,8134,9999]
    datat = np.zeros(len(time_steps))
    datay = np.zeros([len(time_steps),2])
    for i in range(len(time_steps)):
        ts = time_steps[i]
        datat[i] = ODEsol.t[ts]
        datay[i,:] = ODEsol.y[ts]
    # our randomly selected points
    test.DataMatrix(datay,datat)


    # set up the problem like the casadi example
    test.setup() 
    test.tf = period # since this is not 24 periodic
    
    
    test.solve()





