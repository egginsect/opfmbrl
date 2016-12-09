from __future__ import division
import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad

def EM(init_params, episodes):
    """params are a tuple: (pi, A, B)
    pi : (num_states,)                          initial state probs
    A  : (num_actions, num_states, num_states)  transition probs
    B  : (num_state, num_renderings)            rendering probs
    """

    def EM_update(params):
        log_params = map(np.log, params)
        expected_stats = grad(log_partition_function)(log_params, episodes)  # E step
        return map(normalize, expected_statistics)                           # M step

    def different(params1, params2):
        return not all(map(np.allclose, params1, params2))

    def fixed_point(f, x0):
        x1 = f(x0)
        while different(x0,x1):
            x0, x1 = x1, f(x1)
        return x1

    return fixed_point(EM_update, init_params)

def normalize(a):
    def replace_zeros(a):
        return np.where(a > 0., a, 1.)

    return a / replace_zeros(a.sum(-1, keepdims=True))

def log_partition_function(natural_params, episodes):
    log_p_init, log_p_dynamics, log_p_render = natural_params
    def single_episode_log_partition_function(episode):
        log_p_state = log_p_init
        for action, rendering in episode:
            log_p_state = (logsumexp(log_p_state[:, None]
                                     + log_p_dynamics[action], axis=0)
                           + log_p_render[:, rendering])

        return logsumexp(log_p_state)

    return sum(map(single_episode_log_partition_function, episodes))
