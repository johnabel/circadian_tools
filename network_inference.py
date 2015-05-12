# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:19:56 2014

@author: john abel
Network inference module for circadian systems.
"""

from __future__ import division
import itertools
import numpy  as np
from scipy import sparse
from scipy import signal, optimize
from scipy.interpolate import UnivariateSpline
import statsmodels.tsa.filters as filters
import matplotlib.pyplot as plt
import PlotOptions as plo
import minepy as mp
import numpy.random as random
import networkx as nx
import pywt

from jha_utilities import corrsort, cmap_rb

from scoop import futures

import pdb




class network(object):
    """
    Class to analyze time series data and infer network connections.    
    JH Abel
    
    Things it can go:
        detrend                     - hodrick-prescott, linear
        infer                       - MIC, kirsten's MI (bad), corrsort
        create adjacency matrices   - either input them or generate from MIC
        find instantaneous phase    - either with hilbert transform, dwt
        
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
        
        if (t is None and sph is not None):
            # defining t from sph
            t = np.arange(0,sph*len(data),sph).astype(float)
            self.t = {'raw': t}

        
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
                
    
    def resample(self,des,sph = None, method = 'spline', s=0):
        """
        des is desired samples/hour, sph is data samples/hour
        This is always for downsampling, so that the simulated data will have
        the same sph as experimental data.
        
        As experimental bioluminescence data is an integration, we do that 
        here as well, summing the values between time points.
        
        We assume that samples are evenly spaced.
        
        Sum: adding counts to get an hourly count, assumed even number of sph
             per each new sampling time
        Spline: interpolating
        """
        if method == 'sum':
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
            
        if method == 'spline':
            if sph is None:
                sph = self.sph['raw']
            self.sph['resample'] = des
            
            t_resample = np.arange(self.t['raw'].min(),self.t['raw'].max(),des)
            data_resample = np.zeros([len(t_resample),self.nodecount])
            for i in xrange(self.nodecount):
                spline = UnivariateSpline(self.t['raw'], 
                                          self.data['raw'][:,i],
                                          s=s)
                data_resample[:,i] = spline(t_resample)
            
            self.t['resample'] = t_resample
            self.data['resample'] = data_resample

        
        
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

    def ipython_MIC(self,use_data='raw'):
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
        self.data['mic']= self.data[use_data]
        
        # feed data correctly
        feed_data = []
        for i in inds:
            # incredibly obnoxious to get into correct format
            feed_data.append(np.vstack((self.data['mic'][:,i[0]], 
                                        self.data['mic'][:,i[1]])).T)
                                        # transform to get it to be veritcal
        
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
            self.mic[use_data] = mic
        else: self.mic = {use_data : mic}
        
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
            np.fill_diagonal(adj,0)
            if hasattr(self,'adj'):
                self.adj['%.2f' % (thresh)] = adj
            else: self.adj = {'%.2f' % (thresh) : adj}

    def simulate_from_adjacency(self, model='SDS', adj='def'):
        """allows you to simulate the network from a certain adjacency
        matrix and a given model"""
        pass


    def detrend(self,detrend='constant',data='raw', 
                est_period = 24, smoothing_parameter = 100000,
                dwt_bin_desired = 3):
        """
        detrend='constant'
        detrends by subtracting a constant
        
        -or-
        
        detrend='hodrick'
        detrends using a hodrick-prescott filter. 
        data: which data set you want to h-p detrend. defaults to 'raw'
        smoothing_parameter: hp smoothing parameter, defaults to 10000
        
        -or-
        
        detrend='dwt'
        detrends using a DWT. select which bin you want out of it. For sph=1,
        this is the third bin
        """
        if detrend == 'constant':
            detrended = detrend_constant(self.data[data])
            self.data['detrend_cons'] = detrended
            
        if detrend == 'hodrick':        
            detrended = detrend_hodrick_prescott(
                            self.t[data],
                            self.data[data], 
                            smoothing_parameter = smoothing_parameter, 
                            est_period = est_period
                            )            
            self.data['detrend_hp'] = detrended
        
        if detrend == 'dwt':
            detrended = detrend_dwt(
                            self.t[data],
                            self.data[data],
                            dwt_bin_desired
                            )
            self.data['detrend_dwt'] = detrended

    def hilbert_transform(self,detrend='detrend_cons', cells='all'):
        """applys a hilbert transform to determine instantaneous phase.
        data should be detrended before applying this filter method.
        method is only applied to listed cells"""
        
        if cells == 'all':
            cells = np.arange(self.nodecount)        
        
        # make sure it is detrended
        try: hil_data = self.data[detrend][:,cells]
        except: 
            print "Constant detrend being used in Helbert transform."
            self.detrend()
            hil_data = self.data['detrend_cons'][:,cells]
        
        # setup phase array        
        theta = np.zeros(hil_data.shape)
        
        #take transform
        for i in xrange(len(cells)):
            hil_trans = signal.hilbert(hil_data[:,i])
            theta[:,i] = np.arctan2(np.imag(hil_trans),np.real(hil_trans))
            
        self.theta = theta
    
    def unwrap_theta(self,discont=2.5):
        """unwraps theta into theta_unwrap"""
        self.theta_unwrap = np.unwrap(self.theta,discont=discont,axis=0)

        
    def bioluminescence_to_spline(self,data='raw',sph='raw',s=0):
        """uses a univariate cubic spline interpolant on single-cell 
        bioluminescence data. returns univariateSpline objects in a dict"""
        spline_dict = {}
        
        for i in xrange(self.nodecount):
            spline_dict[i] = UnivariateSpline(self.t[sph],
                                            self.data[data][:,i],s=s)
        try: self.spline_dict[data] = spline_dict
        except:
            self.spline_dict = {data:spline_dict}
            
    def theta_to_spline(self,sph='raw',s=0):
        """uses a univariate cubic spline interpolant on single-cell 
        UNWRAPPED phase. returns univariateSpline objects in a dict"""
        try: self.theta_unwrap
        except: self.unwrap_theta()        
        
        spline_dict = {}
        
        for i in xrange(self.nodecount):
            spline_dict[i] = UnivariateSpline(self.t['raw'],
                                            self.theta_unwrap[:,i],s=s)
        try: self.spline_dict['theta'] = spline_dict
        except:
            self.spline_dict = {'theta':spline_dict}
            
    def kuramoto_param(self, cells = 'all'):
        """ calculates the kuramoto parameter for all data"""
        
        if cells == 'all':
            cells = np.arange(self.nodecount)
        
        kuramoto = np.zeros(len(self.theta[:,0]))
        mean_phase = np.zeros(len(self.theta[:,0]))
        for i in xrange(len(self.theta[:,0])):
            kuramoto[i] = radius_phase(self.theta[i,:])[0]
            mean_phase[i] = radius_phase(self.theta[i,:])[1]
        
        self.kuramoto = kuramoto
        self.mean_phase = mean_phase
        
    def times_of_period(self,data='detrend_dwt', cells = 'all',
                     interpolating_time_steps = 10000, start_cut = 24, end_time = 120):
        """ returns period times as lists for each individual cell in the 
        list of cells. does so by interpolating a series of data (for example,
        the dwt detrended data) 
        
        end_cut is how much time is removed from the end in order to not mess 
        up the periods with low signal, here, we use 48 hours
        
        Note: if you use the raw data you will get weird values since it is 
        not detrended
        """
        
        #make a dict of splines
        
        try: splines_dict = np.array(self.spline_dict[data].values())
        except: 
            self.bioluminescence_to_spline(data=data)
            splines_dict = np.array(self.spline_dict[data].values())
        
        # only use the splines indicated by "cells"
        if cells == 'all':
            cells = np.arange(self.nodecount)
        splines_dict = splines_dict[cells]
        
        periods_list = []
        interpolation_times = np.linspace(start_cut,
                                          end_time,#self.t['raw'].max()-end_cut, 
                                       interpolating_time_steps)
                                    
                                       
        # find periods now from the max of the dwt signal
        for i in xrange(len(cells)):
            inds = signal.argrelmax(splines_dict[i](interpolation_times), 
                                    order=2)
            ptimes = interpolation_times[inds]
            periods = np.diff(ptimes)
            periods_list.append(periods)
        
        
        self.periods_list = periods_list
        
    def plot_splines(self,s_name='bioluminescence', derivative=0, 
                     ax=None, t=None,scaling=1, colors=None, **kwargs):
        """plots the bioluminescence trances as splines. 
        can select derivative to plot also. defaults to plotting 
        bioluminescence as splines.
        
        scaling = 1 yields rad/time
        """
        
        try: spline = self.spline_dict[s_name]
        except: 
            self.bioluminescence_to_spline()
            spline = self.spline_dict['bioluminescence']
        
        if ax is None: ax = plt.subplot()
        
        if t is None: t = self.t['raw']
        
        for i in xrange(self.nodecount):
            if colors is None:
                color = 'gray'
            else: color= colors[i]
            ax.plot(t, scaling*spline[i].derivative(derivative)(t), 
                    color=color, **kwargs)
                    
    def corrsort_mic(self):
        try:
            mat, ord = corrsort(self.mic['raw']*2-1)
        except:
            print "MIC not yet calculated!"
        
        #returns sorted matrix, order
        self.corrsort = [mat, ord]
    
    def corrsort_color_groups(self,division_inds,colors):
        """creates color groups based on corrsort matrices. convenience for 
        plotting"""
        color = ['k']*self.nodecount
        
        divisions_all= np.append([0],division_inds)
        divisions_all=np.append(divisions_all,self.nodecount)
        

        for i in range(len(divisions_all)-1):
            inds = self.corrsort[1][divisions_all[i]:divisions_all[i+1]]
            for j in inds:
                color[j] = colors[i]
        return color
        
    def clustering_coeff(self):
        """celculates the clustering coefficient"""
        
        print 'The average clustering coefficient is: ',
        print nx.average_clustering(self.nx_graph)
        self.c_delta_avg = nx.average_clustering(self.nx_graph)
    
    def average_path_length(self):
        """celculates the average path length"""
        if nx.number_connected_components(self.nx_graph) > 1:
            print 'Unconnected bits exist.'
            G = self.nx_graph.subgraph(nx.shortest_path(self.nx_graph.to_undirected(),10))
            print 'The average shortest path length is: ',
            self.l_avg = nx.average_shortest_path_length(G)
            print nx.average_shortest_path_length(G)
            return
        else: 
            print 'The average shortest path length is: ',
            self.l_avg = nx.average_shortest_path_length(self.nx_graph)
            print nx.average_shortest_path_length(self.nx_graph)
    
    def generate_bioluminescence_gif(self, name = '', delete_pngs = True):
        """ uses imagemagick to make a gif of the bioluminescence oscillating.
        requires locations be defined. """
        import subprocess

        
        max_x = np.max(np.array(self.locations.values())[:,0])
        min_x = np.min(np.array(self.locations.values())[:,0])
        
        max_y = np.max(np.array(self.locations.values())[:,1])
        min_y = np.min(np.array(self.locations.values())[:,1])
        
        max_luminescence = np.max(self.data['raw'])
        min_luminescence = np.min(self.data['raw'])
        
        map = np.zeros([max_x-min_x+1,max_y-min_y+1])
        
        for ti in xrange(len(self.data['raw'])):
            time = self.t['raw'][ti]
            print time
            
            for ci in xrange(self.nodecount):
                map[self.locations[ci][0]-min_x,self.locations[ci][1]-min_y] = \
                                    self.data['raw'][ti,ci]
            map = (map)/(max_luminescence-min_luminescence)
            
            plo.PlotOptions()
            plt.figure()
            ax = plt.subplot(111)
            colb = ax.pcolormesh(map,cmap = 'RdBu_r',vmax=1.00,vmin=0.00)
            ax.tick_params(\
                axis='both',       # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off',
                left='off',
                right='off',
                labelleft='off')
            ax.set_title('Time: '+'{0:03d}'.format(time)+' hr')
            plt.colorbar(colb)
            ax.set_xlim([0,max_y-min_y+1])
            ax.set_ylim([0,max_x-min_x+1])            
            fig = "frame_%04d"%(ti)
            plt.savefig('temp/'+fig+'.png',format='png')
            plt.clf()
            
        command = ('convert',
           'temp/frame*png',
           '+dither',
           '-layers',
           'Optimize',
           '-colors',
           '256',
           '-depth',
           '8',
           'temp/bioluminescence'+name+'.gif')

        subprocess.check_call(command)
        
        if delete_pngs is True:
            print "Deletion must be run manually."
        
        
    def population_dft(self,cells = 'all', window = 'hamming',
                       data = 'raw'):
        """ 
        Performs a discrete fourier transform on the population mean 
        bioluminescence. Uses a hamming window by default to scale data.
        cells: indexes of all cells to perform this on. if cells = 'all',
               all cells are used
        
        """

        if cells == 'all':
            cells = np.arange(self.nodecount)
        fdata = self.data[data][:,cells].mean(1)

        time_step = 1.0/self.sph[data]
        
        if window == 'hamming':
            fdata = fdata*signal.hamming(len(fdata))
        
        freqs = np.fft.fftfreq(fdata.size, time_step)
        
        power = np.abs(np.fft.fft(fdata))**2
        
        return_dict = {
                        'cells':cells,
                        'fdata':fdata,
                        'freqs':freqs,
                        'power':power
                        }
                        
        self.dft_population_dict = return_dict
    
    def sc_dft(self,cells='all', window = 'hamming',
               data = 'raw', remove_DC = True):
        """        
        Performs a discrete fourier transform on the single cell 
        bioluminescence. Uses a hamming window by default to scale data.
        cells: indexes of all cells to perform this on. if cells = 'all',
               all cells are used
        """
        if cells == 'all':
            cells = np.arange(self.nodecount)
        fdata = self.data[data][:,cells]
        
        if window == 'hamming':
            fdata = ((fdata.T)*signal.hamming(len(fdata))).T
        
        time_step = 1.0/self.sph[data]
        freqs = np.fft.fftfreq(fdata.shape[0], time_step)
        
        powers = np.zeros([freqs.shape[0],len(cells)])
        
        for i in xrange(len(cells)):
            if remove_DC == True: fdata[:,i]=fdata[:,i] - fdata[:,i].mean() 
            powers[:,i] = np.abs(np.fft.fft(fdata[:,i]))**2
        
        return_dict = {
                        'cells':cells,
                        'fdata':fdata,
                        'freqs':freqs,
                        'powers':powers
                        }
        self.dft_single_cell_dict = return_dict
    
    def power_circ_bin(self, low_per = 18.0, high_per = 32.0):
        """
        Uses sc dft to identify the power in each cell's circadian bin.
        DC (frq = 0 component) should be removed.
        """
        if not 'cells' in self.dft_single_cell_dict:
            print "DFT has not been performed."
            return
        cells = self.dft_single_cell_dict['cells']
        freqs = self.dft_single_cell_dict['freqs']
        low_freq = 1.0/high_per
        high_freq = 1.0/low_per
        
        circ_frac = np.zeros(len(cells))
        
        for i in range(len(cells)):
            powers = self.dft_single_cell_dict['powers'][:,i]
            total_power = powers[freqs>0].sum()
            sum_low = powers[freqs>low_freq].sum()
            sum_high = powers[freqs>high_freq].sum()
            circ_power = sum_low-sum_high
            circ_frac[i] = circ_power/total_power
        
        self.dft_single_cell_dict['circ_bin_power'] = circ_frac
    
    def ls_periodogram(self, cells='all', data = 'raw', 
                       period_low=None, period_high=None, res=None, norm=True,
                       remove_dwt_trend=False, dwt_max_bin=None):
        """ calculates the lomb-scargle normalized periodogram for 
        bioluminescence of individual cells. creates and attaches array of 
        frequencies, pgram magnitudes, and significance."""
        
        if period_low is None:
            nyquist_freq = self.sph[data]/2
            period_low = 1/nyquist_freq
        if period_high is None:
            period_high = 64# max(self.t['raw'])/4
        if res is None:
            res = (period_high-period_low)*10
        
        # select cell traces
        if cells == 'all':
            cells = np.arange(self.nodecount)
        lsdata = self.data[data][:,cells]
        
        pgrams = np.zeros([res,cells.size])        
        sigs = np.zeros([res,cells.size])
        
        for i in range(len(cells)):
            cell_data = lsdata[:,i]
            
            #removes dwt lowest bin, the trend, and dwt bin for 64-128 hour
            # periods, due to some leakage?
            if remove_dwt_trend:
                cell_data = sum(dwt_breakdown(self.t[data], 
                                    cell_data)['components'][:dwt_max_bin],0)
                
            cell_pers, cell_pgram, cell_sig = \
                                periodogram(self.t[data], cell_data, 
                                            period_low = period_low,
                                            period_high = period_high, 
                                            res = res, norm = True)
            pgrams[:,i] = cell_pgram
            sigs[:,i] = cell_sig
        
        self.periodogram = {}
        self.periodogram['cells'] = cells
        self.periodogram['periods'] = cell_pers
        self.periodogram['pgrams'] = pgrams
        self.periodogram['sigs'] = sigs
    

    def find_peaktimes(self, data='raw', t='raw', tstart=0, tend=24, order=5):
        """ finds and returns list of peaks between tstart and tend using 
        the scipy signal argrelmax function
        NOTE: can only currently identify a single peaktime, 0 and 2+ peaks
        return a peaktime of 0"""
        
        ptimes_ref = np.zeros(self.nodecount)
        for i in xrange(self.nodecount):
            inds_ref = signal.argrelmax(self.data[data][tstart:tend,i],
                                        order=order)[0]
            if len(inds_ref)==1:
                ptimes_ref[i] = self.t[t][inds_ref+tstart]
        
        return ptimes_ref
    
    def ls_rhythmic_cells2(self, p = 0.05, circ_low=18, circ_high=32):
        """TEST FUNCTION determines which cells are rhythmic from the lomb-scargle
        periodogram, by comaparing significance to our p-value"""
        
        try:
            cells = self.periodogram['cells']
            periods = self.periodogram['periods']
            pgrams = self.periodogram['pgrams']
        except:
            print "ERROR: ls_periodogram must be calculated before testing."
            return
        
        period_inds = np.where(
                        np.logical_and(circ_low <= self.periodogram['periods'], 
                                       self.periodogram['periods'] <= circ_high))
        
        # find the critical z-score (= pgram score)
        z = -np.log(1-(1-p)**(1/len(self.t['raw'])))
        
        #set up empty lists of rhythmic cells and max power periods
        rhythmic_cells = np.zeros(len(cells))
        cell_periods = np.zeros(len(cells))
        for i in range(len(cells)):
            pgrams_all = pgrams[:,i]
            pgrams_in_range = pgrams[period_inds,i].flatten()
            if (pgrams_in_range > z).any():
                
                # enforces that highest peak must be in circadian range
                all_peak_locs = signal.argrelmax(pgrams_all, order = 2)
                all_peaks = pgrams_all[all_peak_locs]
                range_peak_locs = signal.argrelmax(pgrams_in_range, order=2)
                range_peaks = pgrams_in_range[range_peak_locs]

                if np.max(all_peaks) == np.max(range_peaks):                  
                    rhythmic_cells[i] = 1
                    cell_periods[i] = periods[period_inds[0][np.argmax(pgrams_in_range[0])]]
        
        #period_of_oscillatory_cells = np.argma
        self.periodogram['rhythmic_'+str(p)] = rhythmic_cells
        self.periodogram['zcrit_p'+str(p)] = z
        self.periodogram['cell_period_peaks'] = cell_periods
        self.rc = np.where(rhythmic_cells > 0.1)[0]
        self.nrc = np.where(rhythmic_cells < 0.1)[0]
    
    def rhythmic_cells_given(self,cellnames):
        """ vania and anne provide which cells are found to be rhythmic,
        this adds them as self.vrc, and the non-rhythmic cells as self.vnrc"""
        
        cellnames = cellnames[~np.isnan(cellnames)] #remove nans
        self.vrc = cellnames.astype(int)
        nodenames = xrange(self.nodecount)
        self.nvrc =np.array(list(set(nodenames) - set(cellnames)))
        
    def rhythmic_comparison(self):
        """provides self.crc, common rhythmic cells, and net.cnrc, common
        nonrhythmic cells"""
        self.crc = np.array(list(set(self.rc) & set(self.vrc))).astype(int)
        self.ncrc = np.array(list(set(self.nrc) & set(self.nvrc))).astype(int)
        
    def networkx_graph(self,adj=None):
        """creates a networkx graph object for further analysis"""
        
        # initialize the object
        G = nx.Graph()
        G.add_nodes_from(range(self.nodecount))
        
        if adj is not None:
            edge_list = np.vstack(np.where(self.adj[adj]!=0)).T
            G.add_edges_from(edge_list)
        
        self.nx_graph = G
    
    def networkx_plot(self,ax=None, colors=None, invert_y=False, **kwargs):
        """plots a networkx graph"""
        
        plo.PlotOptions()
        try:
            G = self.nx_graph
        except: 
            print "ERROR: Networkx graph must be generated before it is plotted."
            return
            
        if ax is None:
            ax = plt.subplot()
    
        
        #Turn off the ticks
        ax.tick_params(\
                axis='both',       # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off',
                left='off',
                right='off',
                labelleft='off')
                
        if invert_y==True:
            plt.gca().invert_yaxis()
            
        nx.draw_networkx_nodes(G, 
                               pos=self.locations,
                               node_size=10,
                               node_color=colors, ax=ax, **kwargs)
        nx.draw_networkx_edges(G,pos=self.locations,width=0.5,alpha=0.1,ax=ax,**kwargs)
        
    
    def find_path_lengths(self, adj = 'def',num=500):
        """ takes a networkx network and returns the probability of finding a 
        connection between any two nodes of AT LEAST _physical_ length delta,
        NOT as in Eguiluz et al., 2005. essentially, gives a path length 
        distribution (same as node degree distribution)"""
        
        #get length max from subtracting max and min of each row and 
        length_max = np.linalg.norm(np.array(self.locations.values()).max(0) 
                        - np.array(self.locations.values())[:,0].min(0))

        deltas = np.linspace(0,length_max,num=num)
        
        path_lengths = []
        
        adjacency = self.adj[adj]
        # assemble path lengths
        for i in xrange(self.nodecount):
            for j in xrange(self.nodecount):
                if adjacency[i,j] > 0:
                    i_pos = np.array(self.locations[i])
                    j_pos = np.array(self.locations[j])
                    path_lengths.append(np.linalg.norm(i_pos - j_pos))
        
        prob_list = []
        for i in xrange(len(deltas)):
            prob = (path_lengths > deltas[i]).sum()/len(path_lengths)
            prob_list.append(prob)
        self.path_lengths = path_lengths

    def plot_heatmap2(self,cells,heats=None,ax=None,cmap='YlGn',
                     xtitle='',max_heats=1,min_heats=0):
        """ heatmap generator for any of the things Erik wants. Feed in the 
        heats that match up to the cells
        cells: list of locations
        masked array method
        """
        
        if heats is not None:
            # ensure lengths are ok
            assert len(cells) == len(heats)
        
        if ax is None:
            ax = plt.subplot()
            
        max_x = np.max(np.array(self.locations.values())[:,0])
        min_x = np.min(np.array(self.locations.values())[:,0])
        
        max_y = np.max(np.array(self.locations.values())[:,1])
        min_y = np.min(np.array(self.locations.values())[:,1])
        
        greatest_diff = np.max([(max_x-min_x+1), (max_y-min_y+1)])
        
        
        
        # the map we will eventually plot
        map = np.ones([greatest_diff,greatest_diff])*-1E-10 #default to no ROI

        
        for ci in xrange(self.nodecount):
            # all ROI receive a value for if they are not rhythmic
            map[self.locations[ci][0]-min_x,self.locations[ci][1]-min_y] =\
                            -0.5

        if len(heats)>0:          
            for hi in xrange(len(cells)):
                ci = cells[hi]
                map[self.locations[ci][0]-min_x,self.locations[ci][1]-min_y] =\
                                heats[hi]
        
        
        map1 = np.ma.masked_array(map,map>0) #mask all non-negative values
        map2 = np.ma.masked_array(map,map<0) 
        
        colb1 = ax.pcolormesh(map1,cmap = 'Greys',vmax=0,vmin=-1,
                              rasterized=True)
        #cb1 = plt.colorbar(colb1)
        
        colb2 = ax.pcolormesh(map2,cmap=cmap,vmax=max_heats,vmin=min_heats,
                              rasterized=True)
        #cb2 = plt.colorbar(colb2)
        
        ax.tick_params(\
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            left='off',
            right='off',
            labelleft='off')
        ax.set_title(xtitle)
        ax.set_xlim([0,len(map)])
        ax.set_ylim([0,len(map)])
        return map,colb1,colb2 #returns for colorbar
        
#if you want to scoop your inference, import scoop and write its own function
#in your program
def ipython_periodogram(nets,**kwargs):
    """quickly takes periodogram for a list of networks"""
    from IPython import parallel as pll        
        
    # checks to make sure Client may be run
    try: client = pll.Client()
    except IOError, e:
        print "Cannot run parallel calculation, no engines running."
        print e
        return
    
    import time
    start_time = time.time()
   
    #parallel component
    dview = client[:] # use all nodes alotted
    #include necessary imports for function
    dview.execute('import network_inference as ni; reload(ni)')
        
    dview.map_sync(ipp,nets)
    
    end_time = time.time()-start_time
    print end_time
    
def ipp(net,**kwargs):
    """attached to ipython_periodogram"""
    net.ls_periodogram(**kwargs)

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

def detrend_hodrick_prescott(t,data,smoothing_parameter = None, 
                             est_period = 24.0):
    """
    detrends using a hodrick-prescott filter.
    smoothing_parameter previously used was 10000
    """
    detrended = np.zeros(data.shape)
    nodecount = data.shape[1]
    
    if smoothing_parameter == None:
        # As recommended by Ravn, Uhlig 2004, a calculated empirically 
        num_periods = (t.max() - t.min())/est_period
        points_per_period = len(t)/num_periods
        smoothing_parameter = 0.05*points_per_period**4
    
    for i in xrange(nodecount):
        cyc,trend = filters.hpfilter(data[:,i],
                                     smoothing_parameter)
        detrended[:,i] = cyc
        
    return detrended

def ipython_mp(data):
    """
    Function that is only called to parallelize the MIC calculation. This
    function is parallelizable and only takes indexes as input.
    """
    return mp.minestats(data[:,0],data[:,1])['mic']

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

def detrend_dwt(t, data, bin_num):
    """Uses a dwt to detrend into the selected bin range"""
    
    # TODO need some conversion ebtween x and t to get the bins situated correctly

    data_dict = np.zeros([len(dwt_breakdown(t, data[:,0])['components'][bin_num]),len(data[0,:])])
    
    #print "Discrete wavelet detrend to isolate periods on:",
    #print dwt_breakdown(t, data[:,0])['period_bins'][bin_num]
    for i in xrange(len(data[0,:])):
        data_dict[:,i] = dwt_breakdown(t, data[:,i])['components'][bin_num]
    
    return data_dict

def dwt_breakdown(x, y, wavelet='dmey', nbins=np.inf, mode='sym'):
    """ Function to break down the data in y into multiple frequency
    components using the discrete wavelet transform """

    lenx = len(x)

    # Restrict to the maximum allowable number of bins
    if lenx < 2**nbins: nbins = int(np.floor(np.log(len(x))/np.log(2)))

    dx = x[1] - x[0]
    period_bins = [(2**j*dx, 2**(j+1)*dx) for j in xrange(1,nbins+1)]

    details = pywt.wavedec(y, wavelet, mode, level=nbins)
    cA = details[0]
    cD = details[1:][::-1]

    # Recover the individual components from the wavelet details
    rec_d = []
    for i, coeff in enumerate(cD):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, wavelet)[:lenx])

    rec_a = pywt.waverec([cA] + [None]*len(cD), wavelet)[:lenx]

    return {
        'period_bins' : period_bins,
        'components' : rec_d,
        'approximation' : rec_a,
    }

def periodogram(x, y, period_low=2, period_high=35, res=200, norm=True):
    """ calculate the periodogram at the specified frequencies, return
    periods, pgram. if norm = True, normalized pgram is returned """
    
    periods = np.linspace(period_low, period_high, res)
    # periods = np.logspace(np.log10(period_low), np.log10(period_high),
    #                       res)
    freqs = 2*np.pi/periods
    try: pgram = signal.lombscargle(x, y, freqs)
    # Scipy bug, will be fixed in 1.5.0
    except ZeroDivisionError: pgram = signal.lombscargle(x+1, y, freqs)
    
    # significance (see press 1994 numerical recipes, p576)
    
    
    if norm == True:
        var = np.var(y)
        pgram_norm = pgram/var
        significance =  1- (1-np.exp(-pgram_norm))**len(x)
        return periods, pgram_norm, significance
    else:
        return periods, pgram


def estimate_period(x, y, period_low=1, period_high=100, res=200):
    """ Find the most likely period using a periodogram """
    periods, pgram = periodogram(x, y, period_low=period_low,
                                 period_high=period_high, res=res)
    return periods[pgram.argmax()]

def phase_plot_polar(phases, ax=None, color='f', nbins=20, **kwargs):
    """
    Makes a basic polar plot of circadian phases. Really not very difficult.
    """
    if ax is None:
        ax = plt.subplot(polar='True')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    N = nbins
    bottom = 0.0
    max_height = 1.0
    
    theta = np.linspace(-np.pi, np.pi, N, endpoint=True)

    radii = np.zeros(theta.shape)
    for i in range(len(radii)-1):
        radii[i] = max_height*np.sqrt(
            np.all([phases < theta[i+1], phases > theta[i]], axis=0).sum()/len(phases))
    
    width = (2*np.pi) / N
    
    bars = ax.bar(theta, radii, width=width, bottom=bottom,**kwargs)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(color)
        bar.set_alpha(0.8)
    
    #ax00.plot([0,pre_20_ang],[0,0.5*pre_20_len],color='k')
    
    ax.axes.spines['polar'].set_visible(False)
    ax.set_yticks([])
    #ax00.set_xticks([0,np.pi/2,np.pi,np.pi*3/2])
    ax.set_yticklabels([])
    ax.set_xticklabels(['0','','6','','12','','18',''])
    
def radius_phase(phases):#, cells='all'):
    """ returns radius, phase, number of cells"""
    
    if len(phases)    ==0:      
        return 0, 0, 0
    
    #if cells == 'all':
    #        cells = np.arange(len(phases))
    
    #phases = phases[cells]
    m = (1/len(phases))*np.sum(np.exp(1j*phases))
    length = np.abs(m)
    ang = np.angle(m)
    
    return length, ang, len(phases)












if __name__ == "__main__":
    
    
    #make up a data series
    time = np.array(range(0,1000))
    
    data = np.zeros([1000,3])
    data[:,0] = random.rand(1000)
    data[:,1] = random.rand(1000) + 4*np.sin(0.1*time)
    data[:,2] = 1.6*random.rand(1000) + 4*np.sin(0.1*time+0.8)
    
    net = network(data,t=time)
    
    # changes samples per hour
    net.resample(0.3204)

    
    
    
