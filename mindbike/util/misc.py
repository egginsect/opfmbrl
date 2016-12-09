from autograd.core import getval
import autograd.numpy as np
from functools import partial
from inspect import getargspec
from mindbike.probability import diag_gaussian, gaussian
import itertools as it
import mindbike.util.conslist as cl

def curry(f, N=None):
    if N is None:
        N = len(getargspec(f).args)

    def curried_f(*args):
        num_unbound = N - len(args)
        if num_unbound == 0:
            return f(*args)
        else:
            return curry(partial(f, *args), N=num_unbound)

    return curried_f

def flatten(value):
    # value can be any nested thing ((), array, [] ) etc
    # returns numpy array
    if isinstance(getval(value), np.ndarray):
        def unflatten(vector):
            return np.reshape(vector, value.shape)
        return np.ravel(value), unflatten

    elif isinstance(getval(value), float):
        return np.array([value]), lambda x : x[0]

    elif isinstance(getval(value), tuple):
        if not value:
            return np.array([]), lambda x : ()
        flattened_first, unflatten_first = flatten(value[0])
        flattened_rest, unflatten_rest = flatten(value[1:])
        def unflatten(vector):
            N = len(flattened_first)
            return (unflatten_first(vector[:N]),) + unflatten_rest(vector[N:])

        return np.concatenate((flattened_first, flattened_rest)), unflatten

    elif isinstance(getval(value), list):
        if not value:
            return np.array([]), lambda x : []

        flattened_first, unflatten_first = flatten(value[0])
        flattened_rest, unflatten_rest = flatten(value[1:])
        def unflatten(vector):
            N = len(flattened_first)
            return [unflatten_first(vector[:N])] + unflatten_rest(vector[N:])

        return np.concatenate((flattened_first, flattened_rest)), unflatten

    else:
        raise Exception("Don't know how to flatten type {}".format(type(value)))

def scalar_mul(value, scalar):
    vect, unflatten = flatten(value)
    return unflatten(vect * scalar)

def getattrs(obj, attrs):
    return map(partial(getattr, obj), attrs)

def format_structure(struct):
    return "".join(" {:>8.2f}".format(x) for x in flatten(struct)[0])

def mean_and_std_error(xs):
    N = len(xs)
    mu = sum(xs) / N
    std_error = np.sqrt(sum([(x - mu)**2 for x in xs]) / N**2)
    return mu, std_error

def weights_scales(scale, layer_sizes):
    return [(scale * np.ones((D_prev, D_cur)) / np.sqrt(D_prev), scale * np.ones(D_cur))
            for D_prev, D_cur in zip(layer_sizes[:-1], layer_sizes[1:])]

def weights_ones(layer_sizes):
    return [(np.ones((D_prev, D_cur)), np.ones(D_cur))
            for D_prev, D_cur in zip(layer_sizes[:-1], layer_sizes[1:])]

def mlp(x, weights):
    for W, b in weights[:-1]:
        x = np.tanh(np.dot(x, W) + b)

    W_final, b_final = weights[-1]
    return np.dot(x, W_final) + b_final

def cleave(vect):
    # splits in half over the final dimension
    halfway = vect.shape[-1] / 2
    return vect[..., :halfway], vect[..., halfway:]

@curry
def mlp_gaussian(params, x):
    return diag_gaussian(*mlp_mu_std(params, x))

def mlp_mu_std(params, x):
    mu, log_std = cleave(mlp(x, params))
    log_std = 2 * np.tanh(0.5 * log_std)  # Prevent from becoming unstable by limiting range
    std = np.exp(log_std)
    mu = mu * std
    return mu, std

def get_stats(x):
    if isinstance(x, np.ndarray):
        return "shape : {:<12}".format(np.shape(x)) + \
            "".join(["{:>5} : {:<7.2f}".format(f.__name__, f(x))
                        for f in [np.min, np.max, np.mean, np.std]])
    else:
        return "\n" + "\n".join(map(get_stats, x)) + "\n"

def repeat_gaussian(G, N):
    return gaussian(J = np.tile(G.J, (N, 1, 1)),
                    h = np.tile(G.h, (N, 1)))

def list_concat(lists):
    return list(it.chain(*lists))

def prepend_name(name_value_pairs, name):
    return [(name + prev_name, value) for prev_name, value in name_value_pairs]

@curry
def flatten_with_names(name_lists, tree):
    if not name_lists:
        return [("", tree)]
    cur_names, remaining_names = name_lists[0], name_lists[1:]
    descendents_by_child = map(flatten_with_names(remaining_names), tree)
    children_with_names = cl.imap(prepend_name, descendents_by_child, cur_names)
    return list_concat(children_with_names)

def plot_weights_stats(ax, weights, h1_labels):
    h2_labels = cl.imap("_{}".format, cl.count())
    h3_labels = ['_W', '_b']
    named_weights = flatten_with_names([h1_labels, h2_labels, h3_labels], weights)
    all_labels, all_weights = zip(*named_weights)
    ax.boxplot(all_weights, labels=all_labels)
