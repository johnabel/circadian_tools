# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:07:05 2015

@author: john
"""
from __future__ import division

# third-party packages
import numpy  as np
import casadi as cs
import pdb
import matplotlib.pyplot as plt
import scipy.signal as signal
import PlotOptions as plo
import matplotlib.colors as colors
import matplotlib.cm as cmx

# my packages
import circadiantoolbox as ctb
import gillespy as gp

#define Bode plotting

def bode(G,f=np.arange(.01,100,.01),desc=None,color='k'):
    """NOT angular frequency, absolute frequency, units 1/time"""
    
    jw = 2*np.pi*f*1j
    y = np.polyval(G.num, jw) / np.polyval(G.den, jw)
    mag = 20.0*np.log10(abs(y))
    phase = np.arctan2(y.imag, y.real)*180.0/np.pi % 360

    #plt.semilogx(jw.imag, mag)
    #plt.semilogx(f,mag,label=desc,color=color)

    return mag, phase

def bode_plot(magnitudes,f=np.arange(.01,100,.01),ax = None, grid = False,
               **kwargs):
    """Takes array of bode magnitudes and plots a correct"""
    
    if ax is None:
        ax = plt.subplot()

    ax.plot(f,magnitudes,**kwargs)
    ax.set_xscale('log')
    if grid == True: ax.grid()

def bode_range(magnitudes,f=np.arange(.01,100,.01),ax = None, grid = False,
               **kwargs):
    """ Takes array of bode magnitudes and plots a shaded range"""
    
    if ax is None:
        ax = plt.subplot()
        
    maxes = magnitudes.max(1)
    mins = magnitudes.min(1)
    
    ax.fill_between(f,maxes,mins,**kwargs)
    ax.set_xscale('log')
    if grid == True: ax.grid()
    

def amplitude_comparison(ts1, ts2, freq, time_step, window = 'hamming', ref = None):
    """
    Compares amplitude for frequency freq between time series ts1 and ts2,
    by using the power spectrum.
    """
    
    if window == 'hamming':
        ts1 = ts1*signal.hamming(len(ts1))
        ts2 = ts2*signal.hamming(len(ts2))
        #ref = ref*signal.hamming(len(ref))
    
    amp1 = np.sqrt(np.abs(np.fft.fft(ts1))**2)
    amp2 = np.sqrt(np.abs(np.fft.fft(ts2))**2)
    
    if ref is not None:
        ref_amp = np.sqrt(np.abs(np.fft.fft(ref))**2)
        amp1 = amp1-ref_amp
        amp2 = amp2-ref_amp     
    
    freqs = np.fft.fftfreq(ts2.size, time_step)
    
    freq_index = np.argmin(np.abs(freq-freqs))    
    
    rel_amp = amp2[freq_index]/amp1[freq_index]

    if rel_amp < 0: 
        print "ERROR: Amplitude of x2 indistinguishable from unpurturbed."
        print "Amplitude of x2 = ",
        print amp2[freq_index]+ref_amp[freq_index]
        print "Reference amplitude = ",
        print ref_amp[freq_index]
           
    return rel_amp
