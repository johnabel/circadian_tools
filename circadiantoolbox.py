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
import pdb
from scipy import signal
from scipy.interpolate import splrep, splev


class CircEval(object):
    """
    This circadian evaluator class is for deterministic ODE simulations.
    """
    
    def __init__(self,model,param,y0):
        """
        Setup the required information
        """
        self.model = model
        self.EqnCount   = self.model.input(cs.DAE_X).size()
        self.ParamCount = self.model.input(cs.DAE_P).size()
        
        self.model.init()
        self.param = param

        self.jacp = self.model.jacobian(cs.DAE_P,0); self.jacp.init()
        self.jacy = self.model.jacobian(cs.DAE_X,0); self.jacy.init()
        
        self.ylabels = [self.model.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(self.EqnCount)]
        self.plabels = [self.model.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(self.ParamCount)]
        
        self.pdict = {}
        self.ydict = {}
        
        for par,ind in zip(self.plabels,range(0,self.ParamCount)):
            self.pdict[par] = ind
            
        for par,ind in zip(self.ylabels,range(0,self.EqnCount)):
            self.ydict[par] = ind
            
        self.iv_ydict = {v: k for k, v in self.ydict.items()}
        self.iv_pdict = {v: k for k, v in self.pdict.items()}
        self.intoptions = {
                'y0tol'            : 1E-9,
                'sintmethod'       : 'staggered',
                'intabstol'        : 1E-10,
                'intreltol'        : 1E-10,
                'intmaxstepcount'  : 50000,
                }

        if y0 is None:
            self.y0 = np.ones(self.EqnCount)
        else: 
            self.y0 = y0       
            
            
    #========================================================================
    #                   ODE INTEGRATION
    #======================================================================== 
          
    def intODEs(self, y0=None, tf = 1000, numsteps = 1000):
        """
        This function integrates the ODEs until well past the transients. Inputs:
            tf          -   the final time of integration. the number of time steps within [0,tf] is below,
                            set usually to 1000
            numsteps    -   the number of steps in the integration.
        """
        if y0==None: y0 = self.y0

        self.integrator = cs.CVodesIntegrator(self.model)

        #Set up the tolerances etc.
        self.integrator.setOption("abstol", self.intoptions['intabstol'])
        self.integrator.setOption("reltol", self.intoptions['intreltol'])
        self.integrator.setOption("max_num_steps", self.intoptions['intmaxstepcount'])
        self.integrator.setOption("tf",tf)
        
        #Let's integrate
        self.integrator.init()
        self.ts = np.linspace(0,tf,numsteps)
        
        
        self.integrator.setInput((y0[:]),cs.INTEGRATOR_X0)
        self.integrator.setInput(self.param,cs.INTEGRATOR_P)
        self.integrator.evaluate()
        self.integrator.reset()
        
        def out(t):
            self.integrator.integrate(t)
            self.integrator.output().toArray()
            return self.integrator.output().toArray().squeeze()
        
        sol = np.array([out(t) for t in self.ts])
        return sol
    
    def burnTransient(self,tf = 1000, numsteps = 10000):
        """
        This function integrates the ODEs until well past the transients returning only a value
        for LCpoint on the limit cycle.
            tf          -   the final time of integration.
            numsteps    -   the number of steps in the integration.
        """

        self.integrator = cs.CVodesIntegrator(self.model)

        #Set up the tolerances etc.
        self.integrator.setOption("abstol", self.intoptions['intabstol'])
        self.integrator.setOption("reltol", self.intoptions['intreltol'])
        self.integrator.setOption("max_num_steps", self.intoptions['intmaxstepcount'])
        self.integrator.setOption("tf",tf)
        
        #Let's integrate
        self.integrator.init()
        self.ts = np.linspace(0,tf,numsteps)
              
        self.integrator.setInput((self.y0[:]),cs.INTEGRATOR_X0)
        self.integrator.setInput(self.param,cs.INTEGRATOR_P)
        self.integrator.evaluate()
        self.integrator.reset()
        
        def out(t):
            self.integrator.integrate(t)
            self.integrator.output().toArray()
            return self.integrator.output().toArray().squeeze()
        
        LCpoint = np.array([out(tf)]).squeeze()
        self.y0 = LCpoint
	return LCpoint
    
        
    def intODEs_sim(self, tf, y0=None, numsteps=10000):
        """
        This function integrates the ODEs until well past the transients. This uses Casadi's simulator
        class, C++ wrapped in swig. Inputs:
            tf          -   the final time of integration.
            numsteps    -   the number of steps in the integration is the second argument
        """
        if y0==None: y0 = self.y0
        
        self.integrator = cs.CVodesIntegrator(self.model)

        #Set up the tolerances etc.
        self.integrator.setOption("abstol", self.intoptions['intabstol'])
        self.integrator.setOption("reltol", self.intoptions['intreltol'])
        self.integrator.setOption("max_num_steps", self.intoptions['intmaxstepcount'])
        self.integrator.setOption("tf",tf)
        
        #Let's integrate
        self.integrator.init()
        self.ts = np.linspace(0,tf, numsteps)
        
        self.simulator = cs.Simulator(self.integrator, self.ts)
        self.simulator.init()
        self.simulator.setInput((y0[:]),cs.INTEGRATOR_X0)
        self.simulator.setInput(self.param,cs.INTEGRATOR_P)
        self.simulator.evaluate()
	
	self.sol = self.simulator.output().toArray()
        return self.simulator.output().toArray()

    def burnTransient_sim(self, tf=1000, numsteps=10000):
        """
        This function integrates the ODEs until well past the transients. This uses Casadi's simulator
        class, C++ wrapped in swig. Inputs:
            tf          -   the final time of integration.
            numsteps    -   the number of steps in the integration is the second argument
        """
        
        self.integrator = cs.CVodesIntegrator(self.model)

        #Set up the tolerances etc.
        self.integrator.setOption("abstol", self.intoptions['intabstol'])
        self.integrator.setOption("reltol", self.intoptions['intreltol'])
        self.integrator.setOption("max_num_steps", self.intoptions['intmaxstepcount'])
        self.integrator.setOption("tf",tf)
        
        #Let's integrate
        self.integrator.init()
        self.ts = np.linspace(0,tf, numsteps)
        
        self.simulator = cs.Simulator(self.integrator, self.ts)
        self.simulator.init()
        self.simulator.setInput((self.y0[:]),cs.INTEGRATOR_X0)
        self.simulator.setInput(self.param,cs.INTEGRATOR_P)
        self.simulator.evaluate()
        ss= self.simulator.output().toArray()

        self.y0 = ss[-1]


        return ss[-1]


    #========================================================================
    #                   Period Finding, Limit Cycle Identification
    #========================================================================  
    
    def find_period(self,t=None,sol=None,StateVar=None):
        """ This function will find the period of a solution of the system of
        ODEs. If there is no period, it will return a negative value. The
        inputs to this function are:
            
            t           -   the 1D array of time values 
            sol         -   the
            solution array for the state variables. the period is
            calculated from the first state variable, but the zero
            crossings of the other state variables are also tested to
            ensure that the solution is periodic in all state variables
            perCount    -   the number of periods to take before
            determining the standard deviation default value is 8.
                            
        Sequence of operations: 1. remove first 30% of t and sol values
        (cut transient) 2. subtract mean from sol, create index of 0s 3.
        find the times at which it crosses 0 4. determine each period
        length for first perCount periods 5. take standard deviation of
        periods, if less than pertol, accept.  6. check to ensure that
        other parameters are oscillating as well.  7. if too high stdev, or 
        not enough 0s, return -1.  8. else, return period.
                                
                                **period as an output returns [period or
                                error number, stdev] """
        if sol == None:
            sol = self.sol
        if t==None:
            t = self.ts
        
        #takes mean values of each state variable, subtracts mean so that the oscillations occur about 0
	if StateVar:
	    stateindex = self.ydict[StateVar]
        else: stateindex=0
        
        matmean = np.zeros(len(sol[0,:]))
        for i in range(len(sol[0,:])):
            matmean[i]=np.mean(sol[:,i]) 

        sol_eval = sol-matmean

        #Index for where zero is crossed
        zci = np.where(np.diff(np.sign(sol_eval[:,stateindex])))[0]
        
        #if not enough zeros occur, returns a period of -2. this may occur when 
        if len(zci) < 6:
            period = [-2, 'stable']
            return period
        
        #times of 0-crossing
        zerocross_time = np.zeros(len(zci))

        #linear interpolation to find period
        for i in range(len(zci)):
            slope = (sol_eval[zci[i]+1,stateindex]-sol_eval[zci[i],stateindex])/(t[zci[i]+1]-t[zci[i]])
            #(f(x2)-f(x1))/(x2-x1) = b2
            
            zerocross_time[i] = t[zci[i]] - sol_eval[zci[i],stateindex]/slope
                        
        period_array = np.zeros(int(len(zci)/2)-1)
        for i in range(int(len(zci)/2)-1):
            period_array[i] = zerocross_time[2*i+2]-zerocross_time[2*i]
        
        per = np.mean(period_array)
        stdevper = np.std(period_array)
        
        if np.isnan(per):
            period = [-5, 'nan']
            return period
        
        if stdevper > 0.001*per:
            period = [-3, per, stdevper]
            return period
       
        period = [per, stdevper]
        
        self.period = per

        return period
        





    def find_y0PLC(self, t, sol, period=None, StateVar=None):
        """
        Identifies a point on the limit cycle such that the first state variable is at 0,
        sets this to the y0.
        
        Circadian dawn is the time 7hours before the concentration of Per/Cry mRNA peaks.
        0-12 day, 12-24 night
        Inputs:
            t - time. from output of simulator
            sol - from output of simulator
            period - period of the LCO
        Outputs:
            y0dawn, dawn index, tnew, ynew
        """

        if period is None:
            period = self.period
        
        if StateVar:
                stateindex = self.ydict[StateVar]
        else: stateindex=0
    
        pts = 10000 #minimum discretization of single limit cycle
        #accurate to O(log(pts-1))
    
        single_per_ptcount = int(period*len(t)/np.amax(t))
            
        #artificial time is merely a set of points
        tnew = (np.linspace(0,3*pts-1,3*pts)/(3*pts))*t[3*single_per_ptcount]            
        
        ynew = np.zeros([3*pts, len(sol[0,:])])
        
        for i in range(len(sol[0,:])):
            #this creates a spline interpolation for each state variable
            tck = splrep(t[0:3*single_per_ptcount], sol[0:3*single_per_ptcount,i])
            ynew[:,i] = splev(tnew[:],tck,der=0)
    
        maxCryPer_mRNAindex = np.argmax(ynew[pts:2*pts,stateindex])+pts
    
    
        t_dawn = tnew[maxCryPer_mRNAindex] - 7*period/24
        if t_dawn > period:
            t_dawn=t_dawn-period    
            
        dawn_index = np.argmin(np.abs(tnew-t_dawn))            
    
        #this is from a spline interpolation.
        y0dawn = ynew[dawn_index+pts,:]
        
        self.y0dawn = y0dawn
        self.t = tnew
        self.y = ynew
        #return [y0dawn, dawn_index, tnew, ynew]
        
       
    #========================================================================
    #                   Fitting Utilities - useful for parameter estimation
    #========================================================================         
    
    def peak_to_trough_ratio(self,tsol,sol,period, StateVar):
        """
        Determines ratio of peak-to-trough concentration of a state variable.
        """

        stateindex = self.ydict[StateVar]
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        
        peak   = np.amax(sol[0:period_point_count,stateindex])
        trough = np.amin(sol[0:period_point_count,stateindex])
        ratio  = peak/trough

        return ratio

    def peak_to_trough_time(self,tsol,sol,period, StateVar): 
        """
        Determines time from peak to next trough
        """   
        stateindex = self.ydict[StateVar]
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        
        peakindex   = np.argmax(sol[0:period_point_count,stateindex])
        troughindex = np.argmin(sol[peakindex:peakindex+period_point_count,stateindex])
        
        pttt = tsol[troughindex]-tsol[peakindex]

        return pttt
        
    def peak_to_peak_time(self,tsol,sol,period, StateVar, StateVar2):
        """
        Determines time between two different things peaking. I.e., time of state variable 2
        peaking - time of state variable 1 peaking. (How long after peak of 1 does 2 peak?)
        
        """
        stateindex = self.ydict[StateVar]
        stateindex2 = self.ydict[StateVar2]
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        
        peak1index   = np.argmax(sol[0:period_point_count,stateindex])
        peak2index = np.argmax(sol[peak1index:peak1index+period_point_count,stateindex2])
        
        ptpt = tsol[peak2index] #peak2index already has peak1index subtracted

        return ptpt
        
    def fraction_of_max2(self,tsol,sol,period, StateVar, StateVar2):
        """
        Use as many state variables as needed. Fraction of StateVar1 of total StateVars at independent maxes.
        """
        stateindex = self.ydict[StateVar]
        stateindex2 = self.ydict[StateVar2]
        
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        
        #peak of each state var
        peak1 = np.amax(sol[0:period_point_count,stateindex])
        peak2 = np.amax(sol[0:period_point_count,stateindex2])

        
        frac = peak1/(peak1+peak2)
        
        return frac
        
    def fraction_of_max3(self,tsol,sol,period, StateVar, StateVar2, StateVar3):
        """
        Use as many state variables as needed. Fraction of StateVar1 of total StateVars at independent maxes.
        """
        stateindex = self.ydict[StateVar]
        stateindex2 = self.ydict[StateVar2]
        stateindex3 = self.ydict[StateVar3]

        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        
        #Peak of each state var
        peak1 = np.amax(sol[0:period_point_count,stateindex])
        peak2 = np.amax(sol[0:period_point_count,stateindex2])
        peak3 = np.amax(sol[0:period_point_count,stateindex3])

        
        frac = peak1/(peak1+peak2+peak3)
        
        return frac
    
    def fraction_of_sum2(self,tsol,sol,period, StateVar, StateVar2):
        """
        Use as many state variables as needed. Fraction of StateVar1 of total StateVars at max sum.
        """
        stateindex = self.ydict[StateVar]
        stateindex2 = self.ydict[StateVar2]
        
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        
        #index at which sum of 2 state vars peaks
        peakindex   = np.argmax(sol[0:period_point_count,stateindex] + sol[0:period_point_count,stateindex2])
        
        peak1 = sol[peakindex, stateindex]
        peak2 = sol[peakindex, stateindex2]
        
        frac = peak1/(peak1 + peak2)
        return frac
        
    def fraction_of_sum3(self,tsol,sol,period, StateVar, StateVar2, StateVar3):
        """
        Use as many state variables as needed. Fraction of StateVar1 of total StateVars at max sum.
        """
        stateindex = self.ydict[StateVar]
        stateindex2 = self.ydict[StateVar2]
        stateindex3 = self.ydict[StateVar3]
        
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        
        #index at which sum of three state vars peaks
        peakindex   = np.argmax(sol[0:period_point_count,stateindex] +\
                        sol[0:period_point_count,stateindex2] +\
                        sol[0:period_point_count,stateindex3])
        
        peak1 = sol[peakindex, stateindex]
        peak2 = sol[peakindex, stateindex2]
        peak3 = sol[peakindex, stateindex3]
        
        frac = peak1/(peak1+peak2+peak3)
        
        return frac
    
    def c_max(self,tsol,sol,period,StateVar):
        stateindex = self.ydict[StateVar]
        #uses time relative to per peak to get time of peak
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        perpeakindex = np.argmax(sol[0:period_point_count,0])
        
        #index at which sum of state var peaks
        peakindex   = np.argmax(sol[0:period_point_count,stateindex])
        conc = np.amax(sol[0:period_point_count,stateindex])
        return conc
    
    ##Times of maximum
    
    def t_max(self,tsol,sol,period,StateVar):
        stateindex = self.ydict[StateVar]
        #uses time relative to per peak to get time of peak
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        perpeakindex = np.argmax(sol[0:period_point_count,0])
        
        #index at which sum of state var peaks
        peakindex   = np.argmax(sol[0:period_point_count,stateindex])
        time = tsol[peakindex]-tsol[perpeakindex] + period*7/24
        return time
        
    def t_max2(self,tsol,sol,period,StateVar,StateVar2):
        stateindex = self.ydict[StateVar]
        stateindex2 = self.ydict[StateVar2]
        
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        perpeakindex = np.argmax(sol[0:period_point_count,0])
        
        #index at which sum of 2 state vars peaks
        peakindex   = np.argmax(sol[0:period_point_count,stateindex] + sol[0:period_point_count,stateindex2])
        time = tsol[peakindex]-tsol[perpeakindex] + period*7/24
        return time

    def t_max3(self,tsol,sol,period,StateVar,StateVar2,StateVar3):
        stateindex = self.ydict[StateVar]
        stateindex2 = self.ydict[StateVar2]
        stateindex3 = self.ydict[StateVar3]
        
                
        period_point_count = int((period/np.amax(tsol))*len(tsol))+1
        perpeakindex = np.argmax(sol[0:period_point_count,0])
        #index at which sum of 3 state vars peaks
        peakindex   = np.argmax(sol[0:period_point_count,stateindex] + 
                    sol[0:period_point_count,stateindex2] + 
                    sol[0:period_point_count,stateindex3])
        
        time = tsol[peakindex]-tsol[perpeakindex] + period*7/24
        return time

        
    def knockout_period(self, period, ParameterKO):
        """
        A particularly tricky bit of code. Re-does integration and period finding, with one
        of the parameters set to 0.
        """
        
        koparam = np.copy(self.param)

        for i in range(len(ParameterKO[:])):
            pindex = self.pdict[ParameterKO[i]]
            koparam[pindex] = 0

        y0in = 1*np.ones(self.EqnCount)
        
        test = CircEval(self.model, koparam, y0in)
    
        y0LC = test.burnTransient(tf = 1000, numsteps = 1000) 
        sol = test.intODEs_sim(y0LC,250,numsteps=10000)
        tsol = test.ts 
        periodsol = test.findPeriod(tsol, sol,'p')
        period_new = periodsol[0]
        #catches slight oscillations
        if np.var(sol[:,0])<0.001:
            period_new=-3

        frac = period_new/period
        return frac
    
    def FD_period_param_sens(self,period,ParameterS):
        """
        Sensitivity of period to a finite change (+0.05) in parameter value for a certain parameter.
        If nonoscillatory, returns a value of [0, 1] (an error value of 1).
        """

        output = [0, 0]
        dp = 0.1
        sensparam = np.copy(self.param)
        sensparam[self.pdict[ParameterS]] = sensparam[self.pdict[ParameterS]]+dp
        
        new2 = CircEval(self.model, sensparam, self.y0)
    
        y0LC = new2.burnTransient(tf = 1000, numsteps = 1000) 
        sol = new2.intODEs_sim(y0LC,250,numsteps=10000)
        tsol = new2.ts 
        periodsol = new2.findPeriod(tsol, sol,'p')
        period_new = periodsol[0]
        
        if period_new < 0:
            output[1]=1
            return output
        
        sens = (period_new-period)/dp
        
        output[0] = sens
        return output
        
    def PhaseResponse(self,tsol,sol,period,StatePRC,intervals=101, 
                      numsteps = 1000, pa = 1):
        """Phase response curve for arbitrary SV
        Steps:
            1. Take initial conditions y0 from sol at t0, set phase to 0.
            2. Increase conc. --> integrate out 10 periods, to past transient, 
                since LC doesnt change, find
               the point at which the states most closely match the LC. Find 
               phase of that point. subtract the two.
        
        STILL IMPERFECT AND NOT READY TO GO
        """
        dt = tsol[1]-tsol[0]
        SVnum = self.ydict[StatePRC]
        pulse = np.amax(sol[:,SVnum])*pa
        cycle = sol[0:np.round(period/dt)+1,:] 
        tf = 10*period
        
        #discretize time, these points will be where we look at the 
        #phase change
        t_prc = np.zeros(intervals)
        for i in xrange(intervals):
            t_prc[i] = i * (period) / (intervals-1)
        
        response = np.zeros([len(t_prc),2])
        response[:,0] = t_prc
        
        #at each point, find the phase response
        
        #first, set up the integrator
        self.integrator = cs.CVodesIntegrator(self.model)

        #Set up the tolerances etc.
        self.integrator.setOption("abstol", self.intoptions['intabstol'])
        self.integrator.setOption("reltol", self.intoptions['intreltol'])
        self.integrator.setOption("max_num_steps", 
                                  self.intoptions['intmaxstepcount'])
        self.integrator.setOption("tf",tf)
        
        #Let's integrate
        self.integrator.init()
        self.ts = np.linspace(0,tf, numsteps)
        self.simulator = cs.Simulator(self.integrator, self.ts)
        self.simulator.init()
        
        for i in range(len(t_prc)):
            t=t_prc[i]
            
            #set y0 for closest point
            y0pulse = np.copy(cycle[np.argmin(np.abs(t-tsol)),:])
            y0pulse[SVnum] = pulse+y0pulse[SVnum]
            
            #integrate
            self.simulator.setInput((y0pulse[:]),cs.INTEGRATOR_X0)
            self.simulator.setInput(self.param,cs.INTEGRATOR_P)
            self.simulator.evaluate()
            ss= self.simulator.output().toArray()
            end = ss[len(ss)-1]

            diff_sol = cycle-end
            diff_t = np.argmin(np.sum(np.abs(diff_sol), axis=1))
            if np.amin(np.sum(np.abs(diff_sol), axis=1)) > 0.05:
                print('Difference too high in PRC, point unsuitable: '),
                print np.min(np.sum(np.abs(diff_sol), axis=1))
            diff_t = tsol[diff_t]
            ps_arg = np.argmin(np.abs([diff_t-t, 
                                         diff_t-t+period,
                                         diff_t-t-period]))
            phase_shift = [diff_t-t, diff_t-t+period, diff_t-t-period][ps_arg]
            #first is true phase difference, second is if it overruns,
            #third is if it is too far back
            response[i,1] = phase_shift
            #find point where this matches
            #find the time at that point
            #compare it with t
            
        self.prc = response
        
#========================================================================
#               ODE Plotting Utilities
#========================================================================
class ODEplots(object):
    """class for plotting results of deterministic models"""
    
    def __init__(self,tsol,sol,period,model):
        self.tsol = tsol
        self.sol  = sol
        self.period = period
        
        self.model = model
        self.EqnCount   = self.model.input(cs.DAE_X).size()
        self.ParamCount = self.model.input(cs.DAE_P).size()
        
        self.ylabels = [self.model.inputSX(cs.DAE_X)[i].getDescription()
                        for i in xrange(self.EqnCount)]
        self.plabels = [self.model.inputSX(cs.DAE_P)[i].getDescription()
                        for i in xrange(self.ParamCount)]
        
        self.pdict = {}
        self.ydict = {}
        
        for par,ind in zip(self.plabels,range(0,self.ParamCount)):
            self.pdict[par] = ind
            
        for par,ind in zip(self.ylabels,range(0,self.EqnCount)):
            self.ydict[par] = ind
            
        self.iv_ydict = {v: k for k, v in self.ydict.items()}
    
    def fullPlot(self,StateVar, figurenum = 0):
        """
        Plots full range of sol vs. t for a given state variable.
            t           -   The array of time values.
            sol         -   The solution matrix for all state variables.
            period      -   The period as determined by findPeriod.
            StateVarNum -   The state variable number being plotted.
            figurenum   -   The figure number.
        """
        plt.figure(figurenum)
        stateindex = self.ydict[StateVar]
        
        #Identifies and plots two periods
        plt.plot(self.tsol[:],self.sol[:, stateindex],label = StateVar)

        plt.title('State Variable Ocsillation')
        plt.xlabel('Time')
        plt.ylabel('State Variable')
        plt.legend()


    
    def twoCycleTimePlot(self, StateVar, t_dawn = 0,figurenum = 1):
        """
        Identifies and plots two periods of the State Variable selected.
        Inputs:
            t           -   The array of time values.
            sol         -   The solution matrix for all state variables. 
                            Assumed to be past transient.
            period      -   The period as determined by findPeriod.
            StateVarNum -   The state variable number being plotted.
            figurenum   -   The figure number.
            
        """
        plt.figure(figurenum)
        stateindex = self.ydict[StateVar]
        
        pointcount = 2*int(self.period*len(self.tsol)/np.amax(self.tsol))
        
        #Identifies and plots two periods
        plt.plot(self.tsol[range(pointcount)],self.sol[range(pointcount), stateindex],label = StateVar)

        plt.title('State Variable Ocsillation')
        plt.xlabel('Time')
        plt.ylabel('State Variable')
        plt.legend()



    def dawn_2cycle_plot(self, dawn_index, StateVar, figurenum = 3):
        """
        Will plot starting at dawn_index for two periods to get two cycles 
        starting at dawn day 1.
        
        """
        
        circadiantime = False
        nightcolor = False
        plt.figure(figurenum)
        self.tsol = self.tsol - self.tsol[dawn_index]

        stateindex = self.ydict[StateVar]
        tmaxind = len(self.tsol) - (len(self.tsol)/3 - dawn_index)
        #tmaxind = len(t)- (len(t)/3 - dawn_index)
        

        if circadiantime == True:
            self.tsol = self.tsol*24/self.period
        
        #Identifies and plots two periods
        plt.plot(self.tsol[dawn_index:tmaxind],self.sol[dawn_index:tmaxind, 
                 stateindex],label = StateVar)
        if nightcolor ==True:
            plt.axvspan(12, 24, facecolor='#d3d3d3', alpha=0.5)
            plt.axvspan(36, 48, facecolor='#d3d3d3', alpha=0.5)

        plt.title('State Variable Ocsillation')
        plt.xlabel('Time, Circadian Hours')
        plt.ylabel('State Variable')
        plt.legend()




    def limitCyclePlot(self,StateVar,StateVar2, figurenum=2):
        """
        Identifies and plots the limit cycle of the two State Variable 
        selected.
        Inputs:
            t           -   The array of time values.            
            sol         -   The solution matrix for all state variables. 
                            Assumed to be untruncated 
                            (containing the transient).
            period      -   The period as determined by findPeriod.
            StateVar1&2 -   The state variable number being plotted.
            figurenum   -   The figure number.
            
        """
        plt.figure(figurenum)
        stateindex = self.ydict[StateVar]
        stateindex2 = self.ydict[StateVar2]
        pointcount = int(self.period*2*len(self.tsol)/np.amax(self.tsol))
        plt.plot(self.sol[range(pointcount), stateindex], 
                          self.sol[range(pointcount), stateindex2])
        plt.title('Limit Cycle Oscillations')
        plt.xlabel(StateVar)
        plt.ylabel(StateVar2)

    def period_param_sens_plot(self,Param):
        return 0


        
#============================================:============================
#                   Data tools - for manipulating data
#======================================================================== 



def psave(filename):
    savefile = open(filename)
    pickler = pickle.Pickler(savefile)
    pickler.clear_memo()
    pickler.dump(filename)
    del pickler
    del savefile






