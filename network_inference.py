# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:19:56 2014

@author: john abel
Network inference module for circadian systems.
"""

from __future__ import division
import itertools
import numpy  as np
import scipy as scp
from scipy import sparse
from scipy import signal
import statsmodels.tsa.filters as filters
import matplotlib.pyplot as plt
import PlotOptions as plo
import minepy as mp
import numpy.random as random
import networkx as nx

from scoop import futures

import pdb



#class initiate with set of time series, cell count, adjacency matrix
#can add class.mut_info, class.MIC, class.MIC.ROC?

class network(object):
    """
    Class to analyze time series data and infer network connections.    
    JH Abel    
    """

    def __init__(self, data, sph=None, t=None, loc=None):
        """
        data should contain multiple time series for analysis. vertical is a 
        time series for a single node, horizontal is contant time different 
        nodes.
        t is an optional series of time points to get the time series correct
        """
        self.sph = {'raw' : sph}
        if t is not None:
            if len(t) != len(data[:,0]):
                print ('ERROR: '
                    +'Time series data and time array do not match in size.')
            else:
                self.t = {'raw': t}
                self.sph = {'raw': 1/(t[1]-t[0])}
        
        self.data = {'raw' : data}
        self.nodecount = len(data[0,:])
        
        if loc is not None:
            # is it a dict?
            if type(loc) is dict:
                self.locations = loc
            # or is it a numpy array
            elif type(loc) is np.ndarray:
                loc_dict = {}
                for i in xrange(len(loc)):
                    loc_dict[i] = (loc[i,1], loc[i,2])
                self.locations = loc_dict
            else:
                print "Location data type not recognized."
                
    
    def resample(self,des,sph = None):
        """
        des is desired samples/hour, sph is data samples/hour
        This is always for downsampling, so that the simulated data will have
        the same sph as experimental data.
        
        As experimental bioluminescence data is an integration, we do that 
        here as well, summing the values between time points.
        
        We assume that samples are evenly spaced.
        """

        if sph is None:
            sph = self.sph['raw']
        self.sph['resample'] = des
        #data_pre_fix = np.copy(self.data)
        data_resample = np.zeros([np.floor(sph*len(self.data['raw'])/des),
                                self.nodecount])
        t_resample = np.zeros([np.floor(sph*len(self.data['raw'])/des),1])
        for i in range(len(data_resample)):
            data_resample[i,:] = np.sum(self.data['raw'][
                    int(i*des/sph):int((1+i)*des/sph),:],axis=0)
            t_resample[i] = self.t['raw'][int((i)*des/sph)]+sph
            
        self.data['resample'] = data_resample
        self.t['resample'] = t_resample
    
    def mutual_info(self,use_sph='raw'):
        """
        kirsten's MI
        calculates mutual information between nodes.
        does not rely on scoop"""
        
        mutinfo = np.zeros([self.nodecount,self.nodecount])
        
        sph = self.sph[use_sph]
        max_lag = 6*sph
        noverlap = 26*sph
        window = 30*sph
        
        c1 = range(self.nodecount)
        c2 = range(self.nodecount)
        
        for i in c1:
            x1 = self.data[use_sph][:,i]
            for j in c2:
                x2 = self.data[use_sph][:,j]
                [C,L,t] = mutual_information(x1, x2, int(max_lag),noverlap, window = window)

                kirstenIm = np.zeros(len(C[0,:]))
                for k in range(len(C[0,:])):
                    kirstenIm[k] = np.max(C[:,k])

                MI = sum(abs(kirstenIm))
                
                mutinfo[i,j] = MI
        if hasattr(self,'mi'):
            self.mi[use_sph] = mutinfo
        else: self.mi = {use_sph : mutinfo}

    def parallel_mutual_info(self,use_sph='raw'):
        """
        Kirsten's MI
        calculates mutual information between nodes.
        uses scoop parallelization
        Needs more args passed to scoop mi        
        """
        
        aa = [range(self.nodecount),range(self.nodecount)]
        inds = list(itertools.product(*aa))
        MI = scoop_mi(inds)[:,:3]
        
        mutinfo = np.reshape(MI,[self.nodecount,self.nodecount])
        
        if hasattr(self,'mi'):
            self.mi[use_sph] = mutinfo
        else: self.mi = {use_sph : mutinfo}
        
        
    def MIC(self,use_sph='raw'):
        """
        Serial calculation of MIC for an array of trajectories.
        """
        c1 = range(self.nodecount)
        c2 = range(self.nodecount)
        
        mic = np.zeros([self.nodecount,self.nodecount])
        for i in c1:
            x1 = self.data[use_sph][:,i]
            for j in c2:
                if i<=j:
                    x2 = self.data[use_sph][:,j]
                    mic[i,j] = mp.minestats(x1,x2)['mic']
                else: 
                    mic[i,j] = mic[j,i]
        
        if hasattr(self,'mic'):
            self.mic[use_sph] = mic
        else: self.mic = {use_sph : mic}
    
    def parallel_MIC(self,use_sph='raw'):
        """
        Parallel calculation of MIC for an array of trajectories. It calls 
        the function below it, scoop_mp, which is parallelizable.
        """
        # index list setup
        inds = list(itertools.combinations_with_replacement(
                    range(self.nodecount),2))
        # takes correct data to pass to MIC
        self.data['mic']= self.data[use_sph]
        info = list(futures.map(self.scoop_mp,inds))
        
        # fill in a connectivity matrix
        mic = np.zeros([self.nodecount,self.nodecount])
        for i in range(len(info)):
            mic[info[i][0],info[i][1]] = info[i][2]
            mic[info[i][1],info[i][0]] = info[i][2]
            
        if hasattr(self,'mic'):
            self.mic[use_sph] = mic
        else: self.mic = {use_sph : mic}

    def ipython_MIC(self,use_sph='raw'):
        """
        Parallel calculation of MIC for an array of trajectories. It calls 
        the function below it, ipython_mp, which is parallelizable.
        """
        from IPython import parallel as pll        
        
        # checks to make sure Client may be run
        try: client = pll.Client()
        except IOError, e:
            print "Cannot run parallel calculation, no engines running."
            print e
            return
        
        import time
        start_time = time.time()
        print client.ids
        # index list setup
        inds = list(itertools.combinations_with_replacement(
                    range(self.nodecount),2))
                    
        # takes correct dataset to pass to MIC
        self.data['mic']= self.data[use_sph]
        
        # feed data correctly
        feed_data = []
        for i in inds:
            # incredibly obnoxious to get into correct format
            feed_data.append(np.vstack((self.data['mic'][:,i[0]], self.data['mic'][:,i[1]])).T)# transform to get it to be veritcal
        
        self.feed_data_mic = feed_data
        
        #parallel component
        dview = client[:] # use all nodes alotted
        #include necessary imports for function
        dview.execute('import minepy as mp')
            
        info = dview.map_sync(ipython_mp,feed_data)
        
        end_time = time.time()-start_time
        print end_time
        
        # fill in a connectivity matrix
        mic = np.zeros([self.nodecount,self.nodecount])
        for i in range(len(inds)):
            mic[inds[i][0],inds[i][1]] = info[i]
            mic[inds[i][1],inds[i][0]] = info[i]
            
        if hasattr(self,'mic'):
            self.mic[use_sph] = mic
        else: self.mic = {use_sph : mic}
        
    def scoop_mp(self,inds):
        """
        Function that is only called to parallelize the MIC calculation. This
        function is parallelizable and only takes indexes as input.
        """
        [c1,c2] = inds
        ts1 = self.data['mic'][:,c1]
        ts2 = self.data['mic'][:,c2]
        return [c1,c2,mp.minestats(ts1,ts2)['mic']]
        
    def pearson_r2(self):
        pass
    
    def create_adjacency(self,method='def', sph='raw', thresh=0.90, adj=None):
        """method def allows you to define an adjacency matrix, method mic 
        allows you to generate one from mic, method mi allows you to generate
        one from mi, r2 from r2."""
        
        #defining one        
        if method=='def':
            if hasattr(self,'adj'):
                self.adj[method] = adj
            else: self.adj = {method : adj}

        if method=='mic':
            adj = np.floor(self.mic[sph]+1-thresh) #floors values below thresh
            if hasattr(self,'adj'):
                self.adj['%.2f' % (thresh)] = adj
            else: self.adj = {'%.2f' % (thresh) : adj}

    def simulate_from_adjacency(self, model='SDS', adj='def'):
        """allows you to simulate the network from a certain adjacency
        matrix and a given model"""
        pass


    def detrend_constant(self,data='raw'):
        """
        detrends by subtracting a constant
        """
        detrended = detrend_constant(self.data[data])
        self.data['detrend_cons'] = detrended
    
    def detrend_hodrick_prescott(self,data='raw',
                                 smoothing_parameter = 100000):
        """
        detrends using a hodrick-prescott filter. 
        data: which data set you want to h-p detrend. defaults to 'raw'
        smoothing_parameter: hp smoothing parameter, defaults to 10000
        """
        detrended = detrend_hodrick_prescott(
                self.data[data], smoothing_parameter = smoothing_parameter)            
        
        self.data['detrend_hp'] = detrended



    def hilbert_transform(self,detrend='detrend_cons'):
        """applys a hilbert transform to determine instantaneous phase.
        data should be detrended before applying this filter method"""
        
        # make sure it is detrended
        try: hil_data = self.data[detrend]
        except: 
            print "Constant detrend being used."
            self.detrend_constant()
            hil_data = self.data['detrend_cons']
        
        # setup phase array        
        theta = np.zeros(hil_data.shape)
        
        #take transform
        for i in xrange(self.nodecount):
            hil_trans = signal.hilbert(hil_data[:,i])
            theta[:,i] = np.arctan2(np.imag(hil_trans),np.real(hil_trans))
            
        self.theta = theta
    
    def networkx_graph(self,adj=None, colors=None):
        """creates a networkx graph object for further analysis"""
        
        # initialize the object
        G = nx.Graph()
        G.add_nodes_from(range(self.nodecount))
        
        if adj is not None:
            edge_list = np.vstack(np.where(self.adj['def']!=0)).T
            G.add_edges_from(edge_list)
        
        self.nx_graph = G
                
    def netowrkx_plot(self,ax=None):
        """plots a networkx graph"""
        
        try:
            G = self.networkx_graph
        except: 
            print "Networkx graph must be generated before it is plotted."

        if ax is None:
            ax = plt.subplot()
        
        #Turn off the ticks
        plt.tick_params(\
                axis='both',       # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off',
                left='off',
                right='off',
                labelleft='off')

        nx.draw_networkx_nodes(G, 
                               pos=self.locations,
                               ax = ax)
        nx.draw_networkx_edges(G,pos,width=1)
        
        pass
        



#if you want to scoop your inference, import scoop and write its own function
#in your program

#parts not in the class
def detrend_constant(data):
    """
    detrends by subtracting a constant
    """
    detrended = np.zeros(data.shape)
    nodecount = data.shape[1]
    
    for i in xrange(nodecount):
        detrended[:,i] = signal.detrend(data[:,i],type='constant')
    
    return detrended

def detrend_hodrick_prescott(data,smoothing_parameter = 100000):
    """
    detrends using a hodrick-prescott filter
    """
    detrended = np.zeros(data.shape)
    nodecount = data.shape[1]
    
    for i in xrange(nodecount):
        cyc,trend = filters.hpfilter(data[:,i],
                                     smoothing_parameter)
        detrended[:,i] = cyc
        
    return detrended
    
def mutual_information(ts1,ts2,max_lag,noverlap,window=None):
    """
    Mutual information function, set up the same way as migram.m
    This is windowed, if you don't want it windowed then leave:
        window = None
    """
    if window==None:
        window = len(ts1)
    
    nints = np.fix((len(ts1)-noverlap)/(window-noverlap)) #interval count
    L = range(int(-max_lag),int(max_lag+1)) #range of lags
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
    """does the actual mutual information calculation"""
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
        Pxy = sparse.coo_matrix((np.ones(len(x[:,i])), (x[:,i],y[:,i])), 
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

def scoop_mi(inds, ts_data, max_lag, noverlap, window):
    """
    exists so you can call mutual information with a list of indicies,
    and ultimately parallelize the calculation
    """
    pass

def ipython_mp(data):
    """
    Function that is only called to parallelize the MIC calculation. This
    function is parallelizable and only takes indexes as input.
    """
    return mp.minestats(data[:,0],data[:,1])['mic']

def generate_trajectories(adjacency,tf=100,inc=0.05):
    pass

def ROC(adj, infer, ints = 1000):
    """
    Compares adj, the connectivity matrix, to infer, the inferred mutual info
    matrix, to determine the ROC curve. Default of cutoff range from 0 to 1, 
    1000 intervals.
    
    Returns: 
        false positive rate, sensitivity, false negative rate, specificity 
    """
    
    if np.shape(adj) != np.shape(infer):
        print 'ERROR: Shapes of adjacency matrices do not match.'
        return
    
    TP = (adj != 0).sum()
    TN = (adj == 0).sum()
    cellcount = len(adj)
    
    roc = np.zeros([ints+1,5])
    for i in xrange(len(roc)):
        criteria = i/(len(roc)-1)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        active = np.floor(infer+criteria)
        
        for c1 in xrange(cellcount):
            for c2 in xrange(cellcount):
                if c1==c2:
                    pass
                else:                    
                    if adj[c1,c2]==0:
                        if active[c1,c2]==0:
                            tn = tn+1
                        else:
                            fp = fp+1
                    else:
                        if active[c1,c2]>=1:
                            tp = tp+1
                        else:
                            fn = fn+1
        
        fpr = fp/TN #false positive rate (fall-out)
        tpr = tp/TP #true positive rate (sensitivity)
        fnr = fn/TP #false negative rate 
        tnr = tn/TN #true negative rate (specificity)
        
        
        roc[i,:] = [criteria,fpr,tpr,fnr,tnr]
    return roc

























if __name__ == "__main__":
    
    
    #make up a data series
    time = np.array(range(0,1000))
    
    data = np.zeros([1000,3])
    data[:,0] = random.rand(1000)
    data[:,1] = random.rand(1000) + 4*np.sin(0.1*time)
    data[:,2] = 1.6*random.rand(1000) + 4*np.sin(0.1*time+0.8)
    
    net = network(data,t=time)
    
    # changes samples per hour
    net.resample(2)
    
    net.parallel_MIC(use_sph='resample')
    net.create_adjacency(sph='resample',thresh=0.50)
    
    plt.figure()
    plt.pcolormesh(net.mic['resample'])
    plt.figure()
    plt.pcolormesh(net.adj['0.50'])
    
    
    
    
