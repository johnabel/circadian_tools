import numpy as np
import cPickle as pickle
import pylab as plt
import pdb

# Load and process stochastic trajectories
data = pickle.load(open('stochastic_even_data.p', 'rb'))
# data = pickle.load(open('stochastic_sync_data.p', 'rb'))

control = data['control'].squeeze()
ts_cont = control[:,0]
ys_cont = control[:,1:].reshape((len(ts_cont), -1, 2))/data['vol']

pdb.set_trace()

pre = data['pre'].squeeze()
ts_pre = pre[:,0]
ys_pre = pre[:,1:].reshape((len(ts_pre), -1, 2))/data['vol']

pulse = data['pulse'].squeeze()
ts_pulse = pulse[:,0] + ts_pre[-1]
ys_pulse = pulse[:,1:].reshape((len(ts_pulse), -1, 2))/data['vol']

post = data['post'].squeeze()
ts_post = post[:,0] + ts_pulse[-1]
ys_post = post[:,1:].reshape((len(ts_post), -1, 2))/data['vol']

pulse = [ts_pulse[0], ts_pulse[-1]]

ts_pert = np.hstack([ts_pre[:-1], ts_pulse[:-1], ts_post])
ys_pert = np.vstack([ys_pre[:-1], ys_pulse[:-1], ys_post])

ts_cont = ts_cont - pulse[1]
ts_pert = ts_pert - pulse[1]


# Import deterministic model



# # Control movie
# for i in xrange(ys_cont.shape[0]):
#     fig = plt.figure(dpi=70, figsize=(6, 2.2))
#     gs = plt.GridSpec(1,3)
#     ax_ss = fig.add_subplot(gs[0])
#     ax_ss.plot(model.sol[:,0], model.sol[:,1], 'k')
#     ax_ss.plot(ys_cont[:i,:,0].mean(1), ys_cont[:i,:,1].mean(1), 'r--')
#     ax_ss.plot(ys_cont[i,:,0], ys_cont[i,:,1], 'go')
#     ax_ss.plot(ys_cont[i,:,0].mean(0), ys_cont[i,:,1].mean(0), 'ro')
#     ax_ss.set_xlim([0, 0.8])
#     ax_ss.set_xlabel('X')
#     ax_ss.set_ylabel('Y')
#     ax_ss.set_ylim([0, 5.0])
# 
#     ax_ts = fig.add_subplot(gs[1:], sharey=ax_ss)
#     ax_ts.set_xlabel(r'$\hat{t}$')
#     ax_ts.plot(ts_cont[:i] - ts_cont[0], ys_cont[:i, :, 1].mean(1), 'r--')
#     ax_ts.plot(ts_cont[i] - ts_cont[0], ys_cont[i, :, 1].mean(0), 'ro')
#     ax_ts.set_ylim([0, 5.0])
#     ax_ts.set_xlim([0, 6*np.pi])
#     ax_ts.set_xticks([0, 2*np.pi, 4*np.pi, 6*np.pi])
#     ax_ts.set_xticklabels([r'$0$', r'$2\pi$', r'$4\pi$',
#                            r'$6\pi$'])
# 
#     fig.tight_layout(**layout_pad)
#     fig.savefig('cont_plot/cont' + "%03d" % (i,) + '.png',
#                 bbox_inches='tight')
#     plt.close(fig)

# Desync movie
pulse_inds = range(len(ts_pre), len(ts_pre) + len(ts_pulse))
for i in xrange(ys_pert.shape[0]):
    fig = plt.figure(dpi=70, figsize=(6, 2.2))
    gs = plt.GridSpec(1,3)
    ax_ss = fig.add_subplot(gs[0])
    ax_ss.plot(model.sol[:,0], model.sol[:,1], 'k')
    ax_ss.plot(ys_pert[:i,:,0].mean(1), ys_pert[:i,:,1].mean(1), 'r--')
    ax_ss.plot(ys_pert[i,:,0], ys_pert[i,:,1], 'go')
    ax_ss.plot(ys_pert[i,:,0].mean(0), ys_pert[i,:,1].mean(0), 'ro')
    ax_ss.set_xlim([0, 0.8])
    ax_ss.set_xlabel('X')
    ax_ss.set_ylabel('Y')
    ax_ss.set_ylim([0, 5.0])

    ax_ts = fig.add_subplot(gs[1:], sharey=ax_ss)
    ax_ts.set_xlabel(r'$\hat{t}$')
    ax_ts.plot(ts_pert[:i] - ts_pert[0], ys_pert[:i, :, 1].mean(1), 'r--')
    ax_ts.plot(ts_pert[i] - ts_pert[0], ys_pert[i, :, 1].mean(0), 'ro')
    ax_ts.set_ylim([0, 5.0])
    ax_ts.set_xlim([0, 6*np.pi])
    ax_ts.set_xticks([0, 2*np.pi, 4*np.pi, 6*np.pi])
    ax_ts.set_xticklabels([r'$0$', r'$2\pi$', r'$4\pi$', r'$6\pi$'])

    fig.tight_layout(**layout_pad)

    if i in pulse_inds:
        ax_ss.patch.set_facecolor(lighten_color('y', 0.5))
        ax_ts.patch.set_facecolor(lighten_color('y', 0.5))

    fig.savefig('sync_plot/sync' + "%03d" % (i,) + '.png',
                bbox_inches='tight')
    # fig.savefig('desync_plot/desync' + "%03d" % (i,) + '.png',
    #             bbox_inches='tight')
    plt.close(fig)
