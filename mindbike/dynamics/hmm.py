from __future__ import division
import numpy as np
from scipy.misc import logsumexp


def EM(init_params, episodes):
    """params are a tuple: (pi, A, B)
    pi : (num_states,)                          initial state probs
    A  : (num_actions, num_states, num_states)  transition probs
    B  : (num_state, num_renderings)            rendering probs
    """

    def EM_update(params):
        return M_step(*E_step(params, episodes))

    def different(params1, params2):
        return not all(map(np.allclose, params1, params2))

    def fixed_point(f, x0):
        x1 = f(x0)
        while different(x0,x1):
            x0, x1 = x1, f(x1)
        return x1

    return fixed_point(EM_update, init_params)

def E_step(params, episodes):
    def single_episode_counts(episode):
        actions, renderings = map(np.array, zip(*episode))
        messages = pass_messages(params, actions, renderings)
        return expected_stats(params, actions, renderings, messages)

    all_episodes_counts = map(single_episode_counts, episodes)
    return map(sum, zip(*all_episodes_counts))

def normalize(a):
    def replace_zeros(a):
        return np.where(a > 0., a, 1.)

    return a / replace_zeros(a.sum(-1, keepdims=True))

def M_step(*counts_arrays):
    # Add some pseudo counts for stability
    return map(normalize, map(lambda x : x + 1e-3, counts_arrays))

def pass_messages(params, inputs, outputs):
    pil, Al, Bl = map(np.log, params)
    T = len(inputs)  # number of time steps
    N = len(pil)     # number of states

    def messages_forward_log(pil, Al, inputs, loglikes):
        alphal = np.zeros((T+1, N))

        alphal[0] = pil
        for t in xrange(T):
            alphal[t+1] = logsumexp(alphal[t] + Al[inputs[t]].T, axis=1) + loglikes[t]

        return alphal

    def messages_backward_log(Al, inputs, loglikes):
        betal = np.zeros((T+1, N))

        for t in xrange(T-1,-1,-1):
            betal[t] = logsumexp(Al[inputs[t]] + betal[t+1] + loglikes[t], axis=1)

        return betal

    def compute_loglikes(Bl, outputs):
        return Bl[:, outputs].T  # shape (T, N)

    loglikes = compute_loglikes(Bl, outputs)
    alphal = messages_forward_log(pil, Al, inputs, loglikes)
    betal = messages_backward_log(Al, inputs, loglikes)

    return alphal, betal, loglikes

def expected_stats(params, inputs, outputs, messages):
    pil, Al, Bl = map(np.log, params)
    alphal, betal, loglikes = messages
    Nactions, Nobs = Al.shape[0], Bl.shape[1]

    loglike = logsumexp(alphal[0] + betal[0])
    expected_states = np.exp(alphal + betal - loglike)
    expected_pairs = np.exp(
        alphal[:-1,:,None] + (betal[1:,None,:] + loglikes[:,None,:])
        + (Al - loglike)[inputs])

    assert np.allclose(expected_states.sum(1), 1.)
    assert np.allclose(expected_pairs.sum((1,2)), 1.)

    E_init = expected_states[0]
    E_obscounts = np.concatenate([
        expected_states[1:][outputs == obs].sum(0)[None,...]
        for obs in range(Nobs)]).T
    E_transcounts = np.concatenate([
        expected_pairs[inputs == action].sum(0)[None,...]
        for action in range(Nactions)])

    return E_init, E_transcounts, E_obscounts
