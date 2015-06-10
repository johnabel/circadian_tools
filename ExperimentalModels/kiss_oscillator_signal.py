"""
Created on Tue Jan 13 13:01:35 2014

@author: John H. Abel

Model of an electrochemical oscillator based on Wickramasinghe, 
Mrugacz, and Kiss (doi:10.1039/c1cp21429b) equations 7-9.

This is also the first model build using native gillespy, so let's see how
it goes. - jha
"""

# common imports
from __future__ import division

# python packages
import numpy as np
import casadi as cs
import gillespy as gl
import matplotlib.pyplot as plt
from numpy import fft

modelversion = 'kiss_oscillator_2011'

# constants and equations setup
neq = 4
npa  = 10
param = [20, 20, 15, 1600, 0.3, 6E-5, 0.001, 0.01, 0.0102, 1]
freq = 100
y0in = [-1.29374616,  0.67413515, -1.62914683,  0.87060893]
period = 13.92

def kiss_model_ode():
    
    # For two oscillators
    e1 = cs.ssym('e1')
    tht1 = cs.ssym('tht1')

    e2 = cs.ssym('e2')
    tht2 = cs.ssym('tht2')

    state_set = cs.vertcat([e1, tht1, e2, tht2])

    # Parameter Assignments
    A1Rind1 = cs.ssym('A1Rind1')
    A2Rind2 = cs.ssym('A2Rind2')
    V       = cs.ssym('V')
    Ch      = cs.ssym('Ch')
    a       = cs.ssym('a')
    b       = cs.ssym('b')
    c       = cs.ssym('c')
    gam1    = cs.ssym('gam1')
    gam2    = cs.ssym('gam2')
    AR      = cs.ssym('AR') 
    
    # AR is coupling strength, = 50.5 for model in paper

    param_set = cs.vertcat([A1Rind1, A2Rind2, V, Ch, a, b, c,
                            gam1, gam2, AR])

    # Time
    t = cs.ssym('t')

    ode = [[]]*neq
    
    # oscillator 1
    ode[0] = (V+100*np.sin(10*t)-e1)/A1Rind1 - \
       ( Ch*cs.exp(0.5*e1)/(1+Ch*cs.exp(e1)) + a*cs.exp(e1) )*(1-tht1) -\
       (e1-e2)/AR
    ode[1] = (1/gam1)*(\
            (cs.exp(0.5*e1)/(1+Ch*cs.exp(e1)))*(1-tht1) -\
            b*Ch*cs.exp(2*e1)*tht1/(c*Ch+cs.exp(e1))\
            )
    
    # oscillator 2
    ode[2] = (V-e2-np.sin(freq/(2*np.pi)*t))/A2Rind2 - \
       ( Ch*cs.exp(0.5*e2)/(1+Ch*cs.exp(e2)) + a*cs.exp(e2) )*(1-tht2) -\
       (e2-e1)/AR
    ode[3] = (1/gam1)*(\
            (cs.exp(0.5*e2)/(1+Ch*cs.exp(e2)))*(1-tht2) -\
            b*Ch*cs.exp(2*e2)*tht2/(c*Ch+cs.exp(e2))\
            )

    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t,x=state_set,p=param_set), 
            cs.daeOut(ode=ode))

    fn.setOption("name","kiss_oscillator_2011")

    return fn






if __name__ == "__main__":

    """
    Test suite for the model. We will run with both casadi deterministic 
    and gillespy stochastic simulations.
    """

    import circadiantoolbox as ctb
    import PlotOptions as plo

    # create deterministic circadian object
    kiss_odes = ctb.CircEval(kiss_model_ode(), param, y0in)
    kiss_odes.burnTransient_sim()
    kiss_odes.intODEs_sim(200)
    
    # plot deterministic solutions
    plo.PlotOptions()
    plt.figure()
    plt.plot(kiss_odes.ts,kiss_odes.sol[:,0],label = 'e1, V oscillatory')
    plt.plot(kiss_odes.ts,kiss_odes.sol[:,2],label = 'e2')
    plt.title('High-Freq. Signal Propagates as described by TF')
    plt.xlabel('t')
    plt.ylabel('e')
    plt.legend(loc=0)
    plt.tight_layout(**plo.layout_pad)
    
    # fft setup data
    tf=200
    step=2.00020002E-02
    Fs = 1/step # sampling freq
    t = np.arange(0, tf, step=step)
    
    #freq domain
    connections=np.zeros((2,1))
    n = len(t)
    df = Fs/n
    fft_freqs = np.arange(-Fs/2,Fs/2,df)
    
    # find pure signal
    fx2 = np.sin(10*t)
    Fk2 = fft.fftshift(fft.fft(fft.fftshift(fx2)))
    freq_index = np.argmax(np.abs(Fk2))
    
    for out in range(2):
        exp_ind = 2*out
        
        fx = kiss_odes.sol[:,exp_ind]
        Fk = fft.fftshift(fft.fft(fft.fftshift(fx)))
        
        plt.plot(fft_freqs,np.abs(Fk), label=str(out))
        connections[out] = np.abs(Fk[freq_index])**2
    

    
    plt.show()
    

