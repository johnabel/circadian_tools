import numpy as np

from lmfit import minimize, Parameters
import pdb

from Amplitude import Amplitude, gaussian_phase_distribution
from DecayingSinusoid import DecayingSinusoid, SingleModel




class StochDecayEstimator(object):

    def __init__(self, x, ys, base, vary_amp=False, **kwargs):
        """ Class to estimate the decay rate (phase diffusivity) for a
        stochastic model simulation. Assumes the population starts
        completely synchronized at y[0] == base.y0[:-1] """

        assert len(x) == ys.shape[0], "Incorrect Dimensions, x"
        assert base.neq == ys.shape[1], "Incorrect Dimensions, y"

        self.x = x
        self.ys = ys
        self.base = base
        self._kwargs = kwargs
        self.vary_amp = vary_amp

        self.base.__class__ = Amplitude
        self.base._init_amp_class()

        amp, phase, baseline = base._cos_components()
        self._cos_dict = {
            'amp'      : amp,
            'phase'    : phase,
            'baseline' : baseline,
        }

        self.masters = [self._run_single_state(i) for i in
                        xrange(base.neq)]

        sinusoid_param_keys = ['decay', 'period']
        
        self.sinusoid_params = {}
        for param in sinusoid_param_keys:
            vals = np.array([master.averaged_params[param].value for
                             master in self.masters])
            self.sinusoid_params[param] = np.average(vals)


        xbar_param = Parameters()
        xbar_param.add('decay', value=self.sinusoid_params['decay'],
                       min=0)
                       
        # WHAT IS HAPPENING HERE AAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHH
        self.result = minimize(self._minimize_function, xbar_param)
        
        self.decay = self.result.params['decay'].value
        self.x_bar = self._calc_x_bar(self.result.params)
        ss_res = ((self.ys - self.x_bar)**2).sum()
        ss_tot = ((self.ys - self.ys.mean(0))**2).sum()
        self.r2 = 1 - ss_res/ss_tot


    def _run_single_state(self, i):

        start_ind = np.abs(self.x - self.base.T).argmin()
        # skips first period?

        imaster = DecayingSinusoid(
            self.x[start_ind:], self.ys[start_ind:,i], max_degree=0,
            outlier_sigma=10, decay_units='1/hrs', **self._kwargs)

        imaster._estimate_parameters()
        imaster.models = [SingleModel(imaster.x, imaster.y, 1)]
        imodel = imaster.models[0]
        imodel.create_parameters(imaster)
        imodel.params['amplitude'].value = self._cos_dict['amp'][i]
        imodel.params['amplitude'].vary = self.vary_amp
        imodel.fit()
        imaster._fit_models()
        imaster._calculate_averaged_parameters()
        return imaster

        
    def _calc_x_bar(self, param):
        """ Calculate an estimated x_bar given a guess for the phase
        diffusivity, in units of 1/hrs """

        d_hrs = param['decay'].value
        d_rad = d_hrs * self.base.T / (2 * np.pi)

        # Initial population starts with mu, std = 0
        phase_pop = gaussian_phase_distribution(0., 0., d_rad)
        self.base.phase_distribution = phase_pop

        basexbar = self.base.x_bar(2 * np.pi * self.x /
                               self.sinusoid_params['period'])
        basexbar
        return basexbar

    def _minimize_function(self, param):
        """ Function to minimize via lmfit """
        return (self.ys - self._calc_x_bar(param)).flatten()




if __name__ == "__main__":
    from CommonFiles.Models.degModelFinal import create_class
    from CommonFiles.Models.DegModelStoch import simulate_stoch
    # from CommonFiles.Models.Oregonator import create_class, simulate_stoch
    base_control = create_class()

    vol = 250
    periods = 10
    ntraj = 200

    ts_c, traj_control = simulate_stoch(base_control, vol,
                                        t=periods*base_control.y0[-1],
                                        traj=ntraj,
                                        increment=base_control.y0[-1]/100)

    Estimator = StochDecayEstimator(ts_c, traj_control.mean(0),
                                    base_control)


    print "Amplitude Decay Estimate: {0:0.3f}".format(Estimator.decay)
    print "DecayingSinusoid Estimate: {0:0.3f}".format(
        Estimator.sinusoid_params['decay'])


    import matplotlib.pylab as plt
    plt.plot(ts_c, traj_control.mean(0))
    plt.gca().set_color_cycle(None)
    plt.plot(ts_c, Estimator.x_bar, '--')

    plt.show()

