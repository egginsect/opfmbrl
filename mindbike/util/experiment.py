import matplotlib as mpl
import socket
if socket.gethostname() != "Davids-MacBook-Air-2.local":
    mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import os
from time import sleep
import autograd.numpy as np
from mindbike.util.misc import curry, plot_weights_stats
from mindbike.util.conslist import imap, tails, repeat, iunzip, chain
from mindbike.dynamical_vae import look_both_ways
from mindbike.probability import flat
import cPickle as pickle
import shutil

def show_mat(ax, matrix, title):
    ax.set_ylabel(title)
    ax.matshow(np.concatenate(matrix, axis=1), cmap='Greys', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])

def draw_time(ax, time_fraction):
    ax.plot([time_fraction, time_fraction], [0, 1], transform=ax.transAxes,
            lw=3, color='b')

@curry
def write_axes(vect_to_frame, skip, steps_ahead, params, g,
               render, observations, write_params, past_future_dists, axes):
    rs = np.random.RandomState(1)
    past_zs_dist, future_zxs_dist = past_future_dists
    past_zs = past_zs_dist.sample(rs)
    past_xs = map(lambda z: render(z).sample(rs), past_zs)
    future_zs, future_xs = iunzip(future_zxs_dist.sample(rs))
    xs = chain(past_xs, future_xs)
    zs = chain(past_zs, future_zs)

    total_time_steps = steps_ahead

    mean_xs = imap(lambda x: np.mean(x, axis=0), xs)
    std_xs  = imap(lambda x: np.std( x, axis=0), xs)

    observation_frames = map(vect_to_frame, observations[:total_time_steps:skip])
    x_mean_stills      = map(vect_to_frame, mean_xs[     :total_time_steps:skip])
    x_std_stills       = map(vect_to_frame, std_xs[      :total_time_steps:skip])

    ax_observations, ax_x_mean, ax_x_std, ax_z = axes[:4]
    ax_observations.cla(), ax_x_mean.cla(), ax_x_std.cla(), ax_z.cla()

    show_mat(ax_observations, observation_frames,  'Actual rollout')
    show_mat(ax_x_mean,       x_mean_stills,       'Predictive mean')
    show_mat(ax_x_std,        x_std_stills,        'Predictive uncertainty')

    time_fraction = float(len(past_zs)) / total_time_steps
    for ax in axes[:4]:
        draw_time(ax, time_fraction)

    ax_z.set_ylabel('Latent states')
    ax_z.set_xticks([])
    ax_z.set_yticks([])
    plot_lines_with_uncertainty(ax_z, np.stack(zs[:total_time_steps]))

    if write_params:
        ax_weights, ax_weight_grads = axes[4:]
        ax_weights.cla(), ax_weight_grads.cla()
        plot_weights_stats(ax_weights, params, ['dynamics', 'rendering', 'recognition'])
        ax_weights.set_ylabel('weights')
        plot_weights_stats(ax_weight_grads, g, ['dynamics', 'rendering', 'recognition'])
        ax_weight_grads.set_ylabel('grads')

def write_data(filename, params):
    # Write to a temp file then move, in case something is already reading from the current file
    tmp_filename = filename + '.tmp'
    with open(tmp_filename, 'w') as f:
        pickle.dump(params, f)
    shutil.move(tmp_filename, filename)

def read_data(filename):
    while not os.path.isfile(filename):
        sleep(0.1)
    with open(filename) as f:
        return pickle.load(f)

def plot_lines_with_uncertainty(ax, all_zs):
    # all_zs are of shape (time, samples, latent-dim) 
    for zs, color in zip(all_zs.T, ['blue', 'green', 'orange', 'red']):
        ax.plot(zs.T, alpha=0.02, lw=1, color=color)

def plot(filename, vect_to_frame, build_conditionals,
         gen_single_run, steps_ahead=10, skip=2,
         num_samples=10, plot_params=True, save_dir=None,
         num_frames=5):
    write_axes_ = write_axes(vect_to_frame, skip, steps_ahead)
    def ax_writers(params, g):
        observations = gen_single_run()  # A conslist
        stacked_observations = imap(lambda x : np.stack([x] * num_samples), observations)
        actions = repeat(None)
        update, render, recognize = build_conditionals(params)
        past_future_dists = look_both_ways(update, render, recognize, flat(), actions, stacked_observations)
        write_params = chain([plot_params], repeat(False))
        return imap(write_axes_(params, g, render, observations),
                    write_params, past_future_dists[:num_frames])
    if not save_dir:
        plt.ion()
    if plot_params:
        fig = plt.figure()
        num_axes = 6
        axes = [fig.add_subplot(num_axes, 1, i+1) for i in range(num_axes)]
    else:
        fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    # fig.tight_layout()
    while True:
        params = read_data(filename)
        for i, ax_writer in enumerate(ax_writers(*params)):
            ax_writer(axes)
            if save_dir:
                fig_fname = save_dir + '/{}.png'.format(i)
                print "Saving", fig_fname
                fig.savefig(fig_fname)
            else:
                plt.draw()
                plt.pause(0.01)

        if save_dir:
            break
