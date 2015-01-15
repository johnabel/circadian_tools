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
        
        # Storage Dictionary for solver parameters.
        # These options are PSJ, used at various locations throughout
        # the class
        self.NLPdata = {
            'FPTOL'             : 1E-3,
            'ObjMethod'         : 'lsq', # lsq or laplace
            'FPgaurd'           : False,
            'CONtol'            : 0,
            'f_minimize_weight' : 1,
            'stability'         : False # False or numerical weight
        }
        
        # Set up solver options
        # Collocation Solver Options.
        # These options are all passed directly to the CasADi IPOPT
        # solver class, and as such must match the documentation from
        # CasADi
        self.IpoptOpts = {
            'expand_f'                   : True,
            'expand_g'                   : True,
            'generate_hessian'           : True,
            'max_iter'                   : 6000,
            'tol'                        : 1e-6,
            'acceptable_constr_viol_tol' : 1E-4,
            'linear_solver'              : 'ma57',
            'expect_infeasible_problem'  : "yes",
            #'print_level'               : 10      # For Debugging
        }
        
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

    def measurement_matrix(self, matrix, measurement_pars=[]):
        """
        Attaches the matrix A which specifies the measured output states. 
        y(t) = A x(t), where x are the internal states.
        """
        
        height, width = matrix.shape
        assert width == self.EqCount, "Measurement Matrix shape mis-match"
        self.NM = height
        self.Amatrix = matrix

        #self.ydata = self.NM*[[]]
        #self.tdata = self.NM*[[]]
        #self.edata = self.NM*[[]]

        if not measurement_pars: measurement_pars = [False]*self.NM
        else: assert len(measurement_pars) == self.NM, \
                'Measurement Pars shape mis-match'
        
        self.mpars = measurement_pars
        self.NMP = sum(self.mpars)

        self.YMAX = matrix.dot(np.array(self.XMAX))
        self.YMIN = matrix.dot(np.array(self.XMIN))

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

        self.measurement_matrix(np.eye(self.EqCount)) 
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
            self.p_init = self.parameter_estimation()
        
    

    
    def solve(self, x_init=None, p_init=None):
        """
        Runs the overall NLP and solves the problem.
        """
        # set init vals for p and x
        self.set_init_vals(x_init=x_init, p_init=p_init)
        
        # reform model
        self.reform_model()
        
        # initialize everything
        self.initialize()
        
        #set up the collocation solver
        self.collocation_solver_setup()
        
        #solve collocation problem
    
 
   
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
        
        Here we will set up an NLP to estimate the initial parameter
        guess (potentially along with unmeasured state variables) by
        minimizing the difference between the calculated slope at each
        shooting node (a function of the parameters) and the measured
        slope (using a spline interpolant)
        """
        
        # Here we must interpolate the x_init data using a spline. We
        # will differentiate this spline to get intial values for
        # the parameters
        
        node_ts = self.tgrid.reshape((self.nk, self.deg+1))[:,0]
        try:
            f_init = self.sx(self.tgrid,1).T
        except AttributeError:
            # Create interpolation object
            sx = fnlist([])
            def flat_factory(x):
                """ Protects against nan's resulting from flat trajectories
                in spline curve """
                return lambda t, d: (np.array([x] * len(t)) if d is 0 else
                                     np.array([0] * len(t)))
            for x in np.array(self.x_init).T:
                tvals = list(node_ts)
                tvals.append(tvals[0] + self.tf)
                xvals = list(x.reshape((self.nk, self.deg+1))[:,0])
                ## here ##
                xvals.append(x[0])
                
                if np.linalg.norm(xvals - xvals[0]) < 1E-8:
                    # flat profile
                    sx += [flat_factory(xvals[0])]

                else: sx += [spline(tvals, xvals, 0)]

            f_init = sx(self.tgrid,1).T

        # set initial guess for parameters
        logmax = np.log10(np.array(self.PARMAX))
        logmin = np.log10(np.array(self.PARMIN))
        p_init = 10**(logmax - (logmax - logmin)/2)

        f = self.model
        V = cs.MX("V", self.ParamCount)
        par = V

        # Allocate the initial conditions
        pvars_init = np.ones(self.ParamCount)
        pvars_lb = np.array(self.PARMIN)
        pvars_ub = np.array(self.PARMAX)
        pvars_init = p_init

        xin = []
        fin = []

        # For each state, add the (variable or constant) MX
        # representation of the measured x and f value to the input
        # variable list.
        for state in xrange(self.EqCount):
            xin += [cs.MX(self.x_init[:,state])]
            fin += [cs.MX(f_init[:,state])]

        xin = cs.horzcat(xin)
        fin = cs.horzcat(fin)

        # For all nodes
        res = []
        xmax = self.x_init.max(0)
        for ii in xrange((self.deg+1)*self.nk):
            f_out = f.call(cs.daeIn(t=self.tgrid[ii],
                                    x=xin[ii,:].T, p=par))[0]
            res += [cs.sumAll(((f_out - fin[ii,:].T) / xmax)**2)]

        F = cs.MXFunction([V],[cs.sumAll(cs.vertcat(res))])
        F.init()
        parsolver = cs.IpoptSolver(F)

        for opt,val in self.IpoptOpts.iteritems():
            if not opt == "expand_g":
                parsolver.setOption(opt,val)

        parsolver.init()
        parsolver.setInput(pvars_init,cs.NLP_X_INIT)
        parsolver.setInput(pvars_lb,cs.NLP_LBX)
        parsolver.setInput(pvars_ub,cs.NLP_UBX)
        parsolver.solve()
        
        success = parsolver.getStat('return_status') == 'Solve_Succeeded'
        assert success, "Parameter Estimation Failed"

        self.pcost = float(parsolver.output(cs.NLP_COST))

        
        pvars_opt = np.array(parsolver.output( \
                            cs.NLP_X_OPT)).flatten()

        if success: return pvars_opt
        else: return False
        
        
        
    def reform_model(self,sensids=[]):
        """
        Again, a PSJ function with some added goodies
        
        Reform self.model to conform to the standard (implicit) form
        used by the rest of the collocation calculations. Should also
        append sensitivity states (dy/dp_i) given as a list to the end
        of the ode model.
        Monodromy - calculate a symbolic monodromy matrix by integrating
        dy/dy_0,i 
        """
        # Find total number of variables for symbolic allocation
        self.monodromy = bool(self.NLPdata['stability'])
        self.NSENS = len(sensids) + self.EqCount*self.monodromy
        nsvar = self.NSENS * self.EqCount
        self.NVAR = (1 + self.NSENS)*self.EqCount

        # Allocate symbolic vectors for the model
        t     = self.model.inputSX(cs.DAE_T)    # time
        u     = cs.ssym("u", 0, 1)              # control (empty)
        xd_in = self.model.inputSX(cs.DAE_X)    # differential state
        s     = cs.ssym("s", nsvar, 1)          # sensitivities
        xa    = cs.ssym("xa", 0, 1)             # algebraic state (empty)
        xddot = cs.ssym("xd", self.EqCount + nsvar) # differential state dt
        p     = self.model.inputSX(2)           # parameters

        # Symbolic function (from input model)
        ode_rhs = self.model.outputSX()

        # symbolic jacobians
        jac_x = self.model.jac(cs.DAE_X, cs.DAE_X)   
        jac_p = self.model.jac(cs.DAE_P, cs.DAE_X)


        sens_rhs = []
        for index, state in enumerate(sensids):
            s_i = s[index*self.EqCount:(index + 1)*self.EqCount]
            rhs_i = jac_x.mul(s_i) + jac_p[:, state]
            sens_rhs += [rhs_i]

        offset = len(sensids)*self.EqCount
        if self.monodromy:
            for i in xrange(self.EqCount):
                s_i = s[offset + i*self.EqCount:offset + (i+1)*self.EqCount]
                rhs_i = jac_x.mul(s_i)
                sens_rhs += [rhs_i]
            
        sens_rhs = cs.vertcat(sens_rhs).reshape((nsvar,1))
            

        ode = xddot[:self.EqCount] - ode_rhs
        sens = xddot[self.EqCount:] - sens_rhs
        xd = cs.vertcat([xd_in, s])
        tot = cs.vertcat([ode, sens])
        
        self.rfmod = cs.SXFunction([t,xddot,xd,xa,u,p], [tot])

        self.rfmod.init()
    
    
    
    def initialize(self):
        """
        Uses the finished p_init and x_init values to finalize the
        NLPd structure with full matricies on state bounds, parameter
        bounds, etc.
        Requirements: AttachModel, AttachData, SetInitialValues,
        (ParameterEstimation optional)
        """

        NLPd = self.NLPdata
        nsvar = self.EqCount*self.NSENS
        
        nsy = self.EqCount**2 if self.monodromy else 0
        nsp = nsvar - nsy
            

        p_init = self.p_init
        x_init = self.x_init
        xD_init = np.zeros((len(self.tgrid), self.EqCount + nsvar))
        xD_init[:,:self.EqCount] = x_init

        # Set dy_i/dy_0,j = 1 if i=j
        ics = np.eye(self.EqCount).flatten()
        ics = ics[np.newaxis,:].repeat(len(self.tgrid),axis=0)

        if self.monodromy:
            xD_init[:,-nsy:] = ics
            iclist = np.eye(self.EqCount).flatten().tolist()
        else:
            iclist = []

        
        if type(self.XMAX) is np.ndarray: self.XMAX = self.XMAX.tolist()
        if type(self.XMIN) is np.ndarray: self.XMIN = self.XMIN.tolist()

        # Algebraic state bounds and initial guess
        NLPd['xA_min']  = np.array([])
        NLPd['xA_max']  = np.array([])
        NLPd['xAi_min'] = np.array([])
        NLPd['xAi_max'] = np.array([])
        NLPd['xAf_min'] = np.array([])
        NLPd['xAf_max'] = np.array([])
        NLPd['xA_init'] = np.array((self.nk*(self.deg+1))*[[]])

        # Control bounds
        NLPd['u_min']  = np.array([])
        NLPd['u_max']  = np.array([])
        NLPd['u_max']  = np.array([])
        NLPd['u_init'] = np.array((self.nk*(self.deg+1))*[[]])
        
        # Differential state bounds and initial guess
        NLPd['xD_min']  =  np.array(self.XMIN + [-1E6]*nsvar)
        NLPd['xD_max']  =  np.array(self.XMAX + [+1E6]*nsvar)
        NLPd['xDi_min'] =  np.array(self.XMIN + [0]*nsp +
                                    iclist)
        NLPd['xDi_max'] =  np.array(self.XMAX + [0]*nsp +
                                    iclist)
        NLPd['xDf_min'] =  np.array(self.XMIN + [-1E6]*nsvar)
        NLPd['xDf_max'] =  np.array(self.XMAX + [+1E6]*nsvar)
        NLPd['xD_init'] =  xD_init
        # needs to be specified for every time interval
        

        # Parameter bounds and initial guess
        NLPd['p_min']  = np.array(self.PARMIN)
        NLPd['p_max']  = np.array(self.PARMAX)
        NLPd['p_init'] = p_init
        
    
    def collocation_solver_setup(self, warmstart=False):
        """
        Sets up NLP for collocation solution. Constructs initial guess
        arrays, constructs constraint and objective functions, and
        otherwise passes arguments to the correct places. This looks
        really inefficient and is likely unneccessary to run multiple
        times for repeated runs with new data. Not sure how much time it
        takes compared to the NLP solution.
        Run immediately before CollocationSolve.
        """
        
        # Dimensions of the problem
        nx    = self.NVAR # total number of states
        ndiff = nx        # number of differential states
        nalg  = 0         # number of algebraic states
        nu    = 0         # number of controls

        # Collocated variables
        NXD = self.nk*(self.deg+1)*ndiff # differential states 
        NXA = self.nk*self.deg*nalg      # algebraic states
        NU  = self.nk*nu                 # Parametrized controls
        NV  = NXD+NXA+NU+self.ParamCount+self.NMP # Total variables
        self.NV = NV

        # NLP variable vector
        V = cs.msym("V",NV)
          
        # All variables with bounds and initial guess
        vars_lb   = np.zeros(NV)
        vars_ub   = np.zeros(NV)
        vars_init = np.zeros(NV)
        offset    = 0
        
        ## I AM HERE ##
        
        
        #
        # Split NLP vector into useable slices
        #
        # Get the parameters
        P = V[offset:offset+self.ParamCount]
        vars_init[offset:offset+self.ParamCount] = self.NLPdata['p_init']
        vars_lb[offset:offset+self.ParamCount]   = self.NLPdata['p_min']
        vars_ub[offset:offset+self.ParamCount]   = self.NLPdata['p_max']

        # Initial conditions for measurement adjustment
        MP = V[self.NV-self.NMP:]
        vars_init[self.NV-self.NMP:] = np.ones(self.NMP)
        vars_lb[self.NV-self.NMP:] = 0.1*np.ones(self.NMP) 
        vars_ub[self.NV-self.NMP:] = 10*np.ones(self.NMP)


        pdb.set_trace()
        
        offset += self.NP # indexing variable

        # Get collocated states and parametrized control
        XD = np.resize(np.array([], dtype=cs.MX), (self.NK, self.NICP,
                                                   self.DEG+1)) 
        # NB: same name as above
        XA = np.resize(np.array([],dtype=cs.MX),(self.NK,self.NICP,self.DEG)) 
        # NB: same name as above
        U = np.resize(np.array([],dtype=cs.MX),self.NK)

        # Prepare the starting data matrix vars_init, vars_ub, and
        # vars_lb, by looping over finite elements, states, etc. Also
        # groups the variables in the large unknown vector V into XD and
        # XA(unused) for later indexing
        for k in range(self.NK):  
            # Collocated states
            for i in range(self.NICP):
                #
                for j in range(self.DEG+1):
                              
                    # Get the expression for the state vector
                    XD[k][i][j] = V[offset:offset+ndiff]
                    if j !=0:
                        XA[k][i][j-1] = V[offset+ndiff:offset+ndiff+nalg]
                    # Add the initial condition
                    index = (self.DEG+1)*(self.NICP*k+i) + j
                    if k==0 and j==0 and i==0:
                        vars_init[offset:offset+ndiff] = \
                            self.NLPdata['xD_init'][index,:]
                        
                        vars_lb[offset:offset+ndiff] = \
                                self.NLPdata['xDi_min']
                        vars_ub[offset:offset+ndiff] = \
                                self.NLPdata['xDi_max']
                        offset += ndiff
                    else:
                        if j!=0:
                            vars_init[offset:offset+nx] = \
                            np.append(self.NLPdata['xD_init'][index,:],
                                      self.NLPdata['xA_init'][index,:])
                            
                            vars_lb[offset:offset+nx] = \
                            np.append(self.NLPdata['xD_min'],
                                      self.NLPdata['xA_min'])

                            vars_ub[offset:offset+nx] = \
                            np.append(self.NLPdata['xD_max'],
                                      self.NLPdata['xA_max'])

                            offset += nx
                        else:
                            vars_init[offset:offset+ndiff] = \
                                    self.NLPdata['xD_init'][index,:]

                            vars_lb[offset:offset+ndiff] = \
                                    self.NLPdata['xD_min']

                            vars_ub[offset:offset+ndiff] = \
                                    self.NLPdata['xD_max']

                            offset += ndiff
            
            # Parametrized controls (unused here)
            U[k] = V[offset:offset+nu]

        # Attach these initial conditions to external dictionary
        self.NLPdata['v_init'] = vars_init
        self.NLPdata['v_ub'] = vars_ub
        self.NLPdata['v_lb'] = vars_lb

        # Setting up the constraint function for the NLP. Over each
        # collocated state, ensure continuitity and system dynamics
        g = []
        lbg = []
        ubg = []

        # For all finite elements
        for k in range(self.NK):
            for i in range(self.NICP):
                # For all collocation points
                for j in range(1,self.DEG+1):   		
                    # Get an expression for the state derivative
                    # at the collocation point
                    xp_jk = 0
                    for j2 in range (self.DEG+1):
                        # get the time derivative of the differential
                        # states (eq 10.19b)
                        xp_jk += self.C[j2][j]*XD[k][i][j2]
                    
                    # Add collocation equations to the NLP
                    [fk] = self.rfmod.call([0., xp_jk/self.h,
                                            XD[k][i][j], XA[k][i][j-1],
                                            U[k], P])
                    
                    # impose system dynamics (for the differential
                    # states (eq 10.19b))
                    g += [fk[:ndiff]]
                    lbg.append(np.zeros(ndiff)) # equality constraints
                    ubg.append(np.zeros(ndiff)) # equality constraints

                    # impose system dynamics (for the algebraic states
                    # (eq 10.19b)) (unused)
                    g += [fk[ndiff:]]                               
                    lbg.append(np.zeros(nalg)) # equality constraints
                    ubg.append(np.zeros(nalg)) # equality constraints
                    
                # Get an expression for the state at the end of the finite
                # element
                xf_k = 0
                for j in range(self.DEG+1):
                    xf_k += self.D[j]*XD[k][i][j]
                    
                # if i==self.NICP-1:

                # Add continuity equation to NLP
                if k+1 != self.NK: # End = Beginning of next
                    g += [XD[k+1][0][0] - xf_k]
                    lbg.append(-self.NLPdata['CONtol']*np.ones(ndiff))
                    ubg.append(self.NLPdata['CONtol']*np.ones(ndiff))
                
                else: # At the last segment
                    # Periodicity constraints (only for NEQ)
                    g += [XD[0][0][0][:self.NEQ] - xf_k[:self.NEQ]]
                    lbg.append(-self.NLPdata['CONtol']*np.ones(self.NEQ))
                    ubg.append(self.NLPdata['CONtol']*np.ones(self.NEQ))


                # else:
                #     g += [XD[k][i+1][0] - xf_k]
                
        # Flatten contraint arrays for last addition
        lbg = np.concatenate(lbg).tolist()
        ubg = np.concatenate(ubg).tolist()

        # Constraint to protect against fixed point solutions
        if self.NLPdata['FPgaurd'] is True:
            fout = self.model.call(cs.daeIn(t=self.tgrid[0],
                                            x=XD[0,0,0][:self.NEQ],
                                               p=V[:self.NP]))[0]
            g += [cs.MX(cs.sumAll(fout**2))]
            lbg.append(np.array(self.NLPdata['FPTOL']))
            ubg.append(np.array(cs.inf))

        elif self.NLPdata['FPgaurd'] is 'all':
            fout = self.model.call(cs.daeIn(t=self.tgrid[0],
                                            x=XD[0,0,0][:self.NEQ],
                                               p=V[:self.NP]))[0]
            g += [cs.MX(fout**2)]
            lbg += [self.NLPdata['FPTOL']]*self.NEQ
            ubg += [cs.inf]*self.NEQ



        # Nonlinear constraint function
        gfcn = cs.MXFunction([V],[cs.vertcat(g)])


        # Get Linear Interpolant for YDATA from TDATA
        objlist = []
        # xarr = np.array([V[self.NP:][i] for i in \
        #         xrange(self.NEQ*self.NK*(self.DEG+1))])
        # xarr = xarr.reshape([self.NK,self.DEG+1,self.NEQ])
        
        # List of the times when each finite element starts
        felist = self.tgrid.reshape((self.NK,self.DEG+1))[:,0]
        felist = np.hstack([felist, self.tgrid[-1]])

        def z(tau, zs):
            """
            Functon to calculate the interpolated values of z^K at a
            given tau (0->1). zs is a matrix with the symbolic state
            values within the finite element
            """

            def l(j,t):
                """
                Intermediate values for polynomial interpolation
                """
                tau = self.NLPdata['collocation_points']\
                        [self.NLPdata['cp']][self.DEG]
                return np.prod(np.array([ 
                        (t - tau[k])/(tau[j] - tau[k]) 
                        for k in xrange(0,self.DEG+1) if k is not j]))

            
            interp_vector = []
            for i in xrange(self.NEQ): # only state variables
                interp_vector += [np.sum(np.array([l(j, tau)*zs[j][i]
                                  for j in xrange(0, self.DEG+1)]))]
            return interp_vector

        # Set up Objective Function by minimizing distance from
        # Collocation solution to Measurement Data

        # Number of measurement functions
        for i in xrange(self.NM):

            # Number of sampling points per measurement]
            for j in xrange(len(self.tdata[i])):

                # the interpolating polynomial wants a tau value,
                # where 0 < tau < 1, distance along current element.
                
                # Get the index for which finite element of the tdata 
                # values
                feind = get_ind(self.tdata[i][j],felist)[0]

                # Get the starting time and tau (0->1) for the tdata
                taustart = felist[feind]
                tau = (self.tdata[i][j] - taustart)*(self.NK+1)/self.TF

                x_interp = z(tau, XD[feind][0])
                # Broken in newest numpy version, likely need to redo
                # this whole file with most recent versions
                y_model = self.Amatrix[i].dot(x_interp)

                # Add measurement scaling
                if self.mpars[i]: y_model *= MP[sum(self.mpars[:i])]

                # Using relative diff's is probably messing up weights
                # in Identifiability
                diff = (y_model - self.ydata[i][j])

                if   self.NLPdata['ObjMethod'] == 'lsq':
                    dist = (diff**2/self.edata[i][j]**2)

                elif self.NLPdata['ObjMethod'] == 'laplace':
                    dist = cs.fabs(diff)/np.sqrt(self.edata[i][j])
                
                try: dist *= self.NLPdata['state_weights'][i]
                except KeyError: pass
                    
                objlist += [dist]

        # Minimization Objective
        if self.minimize_f:
            # Function integral
            f_integral = 0
            # For each finite element
            for i in xrange(self.NK):
                # For each collocation point
                fvals = []
                for j in xrange(self.DEG+1):
                    fvals += [self.minimize_f(XD[i][0][j], P)]
                # integrate with weights
                f_integral += sum([fvals[k] * self.E[k] for k in
                                   xrange(self.DEG+1)])

            objlist += [self.NLPdata['f_minimize_weight']*f_integral]



        # Stability Objective (Floquet Multipliers)
        if self.monodromy:
            s_final = XD[-1,-1,-1][-self.NEQ**2:]
            s_final = s_final.reshape((self.NEQ,self.NEQ))
            trace = sum([s_final[i,i] for i in xrange(self.NEQ)])
            objlist += [self.NLPdata['stability']*(trace - 1)**2]




        # Objective function of the NLP
        obj = cs.sumAll(cs.vertcat(objlist))
        ofcn = cs.MXFunction([V], [obj])

        self.CollocationSolver = cs.IpoptSolver(ofcn,gfcn)

        for opt,val in self.IpoptOpts.iteritems():
            self.CollocationSolver.setOption(opt,val)

        self.CollocationSolver.setOption('obj_scaling_factor',
                                         len(vars_init))
        
        if warmstart:
            self.CollocationSolver.setOption('warm_start_init_point','yes')
        
        # initialize the self.CollocationSolver
        self.CollocationSolver.init()
          
        # Initial condition
        self.CollocationSolver.setInput(vars_init,cs.NLP_X_INIT)

        # Bounds on x
        self.CollocationSolver.setInput(vars_lb,cs.NLP_LBX)
        self.CollocationSolver.setInput(vars_ub,cs.NLP_UBX)

        # Bounds on g
        self.CollocationSolver.setInput(np.array(lbg),cs.NLP_LBG)
        self.CollocationSolver.setInput(np.array(ubg),cs.NLP_UBG)

        if warmstart:
            self.CollocationSolver.setInput( \
                    self.WarmStartData['NLP_X_OPT'],cs.NLP_X_INIT)
            self.CollocationSolver.setInput( \
                    self.WarmStartData['NLP_LAMBDA_G'],cs.NLP_LAMBDA_INIT)
            self.CollocationSolver.setOutput( \
                    self.WarmStartData['NLP_LAMBDA_X'],cs.NLP_LAMBDA_X)
                    
        
        pdb.set_trace()

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





