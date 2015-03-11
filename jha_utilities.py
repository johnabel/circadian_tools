"""
A mixed file containing some utilities that I find useful in solving 
circadian problems.

jha
"""

#import modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import (splrep, splint, fitpack, splev,
                               UnivariateSpline, dfitpack,
                               InterpolatedUnivariateSpline)


def roots(data,times=None):
    """
    Takes a set of data and finds the roots of it. Uses a spline 
    interpolation to get the root values.
    """

    if times is None:
        #time intervals set to one
        times = np.arange(len(data))

    #fits a spline centered on those indexes
    s = UnivariateSpline(times, data, s=0)

    return s.roots()


class fnlist(list):
    # from Peter's things
    def __call__(self,*args,**kwargs):
        return np.array([entry(*args,**kwargs) for entry in self])


class spline:
    """ Periodic data interpolation object used by Collocation. Probably
    could stand an update """
    def __init__(self,tvals,yvals,sfactor):
        self.max = np.array(yvals).max()
        self.min = np.array(yvals).min()
        self.amp = self.max-self.min
        # scaled y (0->1)
        self.yscaled = (yvals - self.min)/self.amp
        smooth = sfactor*(len(tvals) - np.sqrt(2*len(tvals)))
        spl = splrep(tvals,self.yscaled,s=smooth,per=True)
        self.spl = spl

    def __call__(self,s,d=0):
        if d == 0:
            return self.amp*(splev(s, self.spl, der=d)) + self.min
        else:
            return self.amp*(splev(s, self.spl, der=d))

def bode(G,f=np.arange(.01,100,.01),desc=None,color=None):

    jw = 2*np.pi*f*1j
    y = np.polyval(G.num, jw) / np.polyval(G.den, jw)
    mag = 20.0*np.log10(abs(y))
    phase = np.arctan2(y.imag, y.real)*180.0/np.pi % 360

    #plt.semilogx(jw.imag, mag)
    plt.semilogx(f,mag,label=desc,color=color)

    return mag, phase


if __name__ == "__main__":

    #test roots
    times = np.arange(0,10,0.1)
    xvals = np.sin(times)
    sine_roots = roots(xvals, times=times)
    print 'The roots of sine are:'
    print sine_roots
    print 'Root finding successful.'
