# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:19:56 2014

@author: john abel
Network inference module for circadian systems.
"""

from __future__ import division
import time
import itertools
import numpy  as np
import scipy as scp
import matplotlib.pyplot as plt
import minepy as mp
import statstools.tsa

import pdb



#class initiate with set of time series, cell count, adjacency matrix
#can add class.mut_info, class.MIC, class.MIC.ROC?

class network(object):
    """
    Class to analyze time series data and infer networks.    
    JH Abel    
    """

    def __init__(self,xs,t=None):
        """xs should contain multiple time series for analysis.
        t is an optional series of time points"""
        if t is not None:
            self.t = t
        self.xs = xs
        
        self.nodecount = len(xs[0,:])
        
    def mutual_info(self):
        """calculates mutual information between nodes.
        does not rely on scoop"""
        
        for i in xrange(self.nodecount):
            for j in xrange(self.nodecount):
                
                # set up time series, infer
                x1


    def scoop_mutual_info(self):
        """calculates mutual information between nodes.
        uses scoop parallelization"""
        
        aa = [range(self.nodecount),range(self.nodecount)]
        inds = list(itertools.product(*aa))
        self.mutualinfo = scoop_mi(inds)[:,:3]
        

















#if you want to scoop your inference, import scoop and write its own function
#in your program

#parts not in the class

def mutual_information(ts1,ts2,max_lag,noverlap,window=None):
    """
    Mutual information function, set up the same way as migram.m
    This is windowed, if you don't want it windowed then leave:
        window = None
    """
    if window==None:
        window = len(ts1)
    
    nints = np.fix((len(ts1)-noverlap)/(window-noverlap)) #interval count
    L = range(-max_lag,max_lag+1,1) #range of lags
    C = np.zeros([2*max_lag+1, nints])
    
    #set up lagged arrays, as in migram.m
    X = np.zeros([len(ts1),max_lag+1])
    Y = np.zeros([len(ts2),max_lag+1])

    for i in range(max_lag+1):
        X[i:,i] = ts1[:len(ts1)-i]
        Y[i:,i] = ts2[:len(ts2)-i]
    X = np.fliplr(X)
    
    #Now, collect mutual informations
    #-max lag : 0
    ccount=0
    Xi=np.zeros([window,nints])
    Yi=np.zeros([window,nints])

    for i in range(nints):
        inds = i*(window-noverlap)
        indf = inds+1*window
        Yi[:,i] = (Y[inds:indf,0])
    
    for i in range(len(X[0,:])):
        for j in range(nints):
            inds = j*(window-noverlap)
            indf = inds+1*window
            Xi[:,j] = (X[inds:indf,i])
        C[ccount,:] = MIcalc(Xi,Yi)
        ccount=ccount+1
        
    #0 : max lag
    Xi=np.zeros([window,nints])
    Yi=np.zeros([window,nints])
    for i in range(nints):
        inds = i*(window-noverlap)
        indf = inds+1*window
        Xi[:,i] = (X[inds:indf,-1])

    for i in range(1,len(Y[0,:]),1):
        for j in range(nints):
            inds = j*(window-noverlap)
            indf = inds+1*window
            Yi[:,j] = (Y[inds:indf,i])
        
        C[ccount,:] = MIcalc(Xi,Yi)
        ccount=ccount+1
    
    nx = len(ts1)
    t = np.arange(1,nx,nx/len(Xi[0,:])) #matching t in migram
    return [C, L, t]


def MIcalc(x,y,nbins=10):
    "does the actual mutual information calculation"
    #scale the matrices to be from 0 to 1
    #note: added 2e-15 values are to avoid division by zero errors
    maxsX = np.max(x,axis=0)
    minsX = np.min(x,axis=0)
    rngX  = maxsX - minsX + 2e-15
    x = np.nan_to_num((x-minsX)/(rngX))
    
    maxsY = np.max(y,axis=0)
    minsY = np.min(y,axis=0)
    rngY  = maxsY - minsY + 2e-15
    y = np.nan_to_num((y-minsY)/(rngY))
    
    #separate into bins. let's have 20 bins..
    #rounding fixes floating point error
    x = np.floor(np.around(x,14)*nbins)+1
    y = np.floor(np.around(y,14)*nbins)+1
    
    #now, calculate probabilities
    Z = np.zeros(len(x[0,:]))
    for i in range(len(x[0,:])):
        Pxy = scp.sparse.coo_matrix((np.ones(len(x[:,i])), (x[:,i],y[:,i])), 
                                    shape=[np.max(x[:,i])+1,np.max(y[:,i])+1])

        Px = Pxy.sum(axis=0)
        Py = Pxy.sum(axis=1)
        Pxy = Pxy/Pxy.sum()
        Px = Px/Px.sum()
        Py = Py/Py.sum()
        
        #Information theoretic entropies
        Hx = -np.matrix.sum(np.nan_to_num(np.multiply(Px,np.log(Px+2e-15))))
        Hy = -np.matrix.sum(np.nan_to_num(np.multiply(Py,np.log(Py+2e-15))))
        Hxy = -np.matrix.sum(np.nan_to_num(np.multiply(Pxy.todense(),
                                             np.log(Pxy.todense()+2e-15))))
                             
        MI = Hx + Hy - Hxy
        #output the mutual info
        Z[i] = MI
    return Z

def scoop_mi(inds):
    """
    exists so you can call mutual information with a list of indicies,
    and ultimately parallelize the calculation
    """
    c1 = inds[0]
    c2 = inds[1]
    ts1 = ts_data[:,c1]
    ts2 = ts_data[:,c2]
    [C,L, t] = mutual_information(ts1,ts2,max_lag,noverlap,window=window)
    
    #KIRSTEN METHOD
    kirstenIm = np.zeros(len(C[0,:]))
    for k in range(len(C[0,:])):
        kirstenIm[k] = np.max(C[:,k])
    
    #my method: take the mean mut info at the time lag where mutual info is
    #on average maximum
    john = np.max(np.mean(C,axis=1))
    return [c1, c2, kirstenIm, john]
    
    