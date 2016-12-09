import autograd.numpy as np
from autograd import grad
import autograd.scipy.stats.norm as norm
from mindbike.util.conslist import conslistgen
from collections import namedtuple
np.seterr(over='raise')
np.seterr(invalid='raise')

Distribution = namedtuple('Distribution', ['sample', 'log_density'])

# Write tests for autograd:
# Solve with broadcasting
# Indexing with ellipses
# Ellipses in einsum

def broadcasting_jacobian(fun):
    # Jacobian over final dimension only
    def jac_fun(x):
        out_size = fun(x).shape[-1]
        jac = [grad(lambda x_ : np.sum(fun(x_)[..., i]))(x)[..., None, :]
               for i in range(out_size)]
        return np.concatenate(jac, axis=(x.ndim - 1))

    return jac_fun

def swap_final_axes(A):
    return np.swapaxes(A, -1, -2)

def multiply_gaussians(G1, G2):
    return gaussian(J = G1.J + G2.J, h = G1.h + G2.h)

def ekf(update, marginal):
    mean_dist = update(marginal.mu)
    jac = broadcasting_jacobian(lambda z : update(z).mu)(marginal.mu)
    propagated_sigma = triple_mat_prod(marginal.sigma, swap_final_axes(jac))
    new_marginal = gaussian(mu    = mean_dist.mu,
                            sigma = mean_dist.sigma + propagated_sigma)

    def backwards_conditional(z):
        J_from_future = triple_mat_prod(mean_dist.J, jac)
        h_from_future = (  np.einsum('...ij,...ik,...k->...j', jac, mean_dist.J, z - mean_dist.mu)
                         + np.einsum('...ij,...j->...i', J_from_future, marginal.mu))
        return gaussian(J = marginal.J + J_from_future,
                        h = marginal.h + h_from_future)

    return new_marginal, backwards_conditional

def triple_mat_prod(A, B):
    # Compute B.T * A * B, with broadcasting
    return np.einsum('...ji,...jk,...kl->...il', B, A, B)

def handle_shapes(mu, sigma):
    shape = mu.shape
    N = mu.size
    if sigma.shape[:-1] != shape:
        sigma = np.tile(sigma, shape[:-1] + (1, 1))
    assert sigma.shape[:-1] == shape, "Sigma shape {}, mu shape {}".format(sigma.shape, shape)
    return shape, N, sigma

class gaussian(object):
    def __init__(self, J=None, h=None, mu=None, sigma=None):
        # Assumes that either (J, h) or (mu, sigma) are specified
        if h is not None:
            self.shape, self.N, J = handle_shapes(h, J)
        else:
            self.shape, self.N, sigma = handle_shapes(mu, sigma)

        self._J = J
        self._h = h
        self._mu = mu
        self._sigma = sigma
        self._sigma_chol = None

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol

    @property
    def J(self):
        if self._J is None:
            self._J = np.linalg.inv(self._sigma)
        return self._J

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = np.linalg.inv(self._J)
        return self._sigma

    @property
    def h(self):
        if self._h is None:
            self._h = np.linalg.solve(self._sigma, self._mu)
        return self._h

    @property
    def mu(self):
        if self._mu is None:
            self._mu = np.linalg.solve(self._J, self._h)
        return self._mu

    def sample(self, rs):
        return self.mu + np.einsum('...ij,...j->...i', self.sigma_chol, rs.randn(*self.shape))

    def log_density(self, x):
        delta = x - self.mu
        return - 0.5 * np.sum(np.einsum('...i,...ij,...j->...', delta, self.J, delta)) \
               - np.sum(np.log(np.diagonal(self.sigma_chol, axis1=-1, axis2=-2))) \
               - self.N * 0.5 * np.log(2 * np.pi)

class diag_gaussian(object):
    def __init__(self, mu, std):
        assert std.shape == mu.shape
        self.mu = mu
        self.std = std
        self.N = mu.size

    @property
    def J(self):
        return np.make_diagonal(1 / self.std**2, axis1=-1, axis2=-2)

    @property
    def sigma(self):
        return np.make_diagonal(self.std**2, axis1=-1, axis2=-2)

    @property
    def h(self):
        return self.mu / self.std**2

    def sample(self, rs):
        return self.mu + rs.randn(*self.mu.shape) * self.std

    def log_density(self, x):
        return np.sum(norm.logpdf(x, self.mu, self.std))

def std_gaussian(shape):
    return diag_gaussian(np.zeros(shape), np.ones(shape))

def zero_mean_gaussian(std):
    return diag_gaussian(np.zeros(std.shape), std)

class delta(object):
    def __init__(self, x0):
        self.x0 = x0

    def sample(self, rs):
        return self.x0

    def log_density(self, x):
        return 0.0  # Actually inf, but doesn't depend on parameters so gradient is zero

class flat(object):
    def __init__(self):
        self.J = 0.0
        self.h = 0.0

    def sample(self, rs):
        raise Exception("Can't sample from improper prior")

    def log_density(self, x):
        return 0.0  # Arbitrary constant

def compose(conditional_dist, dist):
    # (a -> Distribution(b)), Distribution(a) -> Distribution((a,b)
    def sample(rs):
        x = dist.sample(rs)
        y = conditional_dist(x).sample(rs)
        return x, y

    def log_density(x_y):
        x, y = x_y
        return dist.log_density(x) + conditional_dist(x).log_density(y)

    return Distribution(sample, log_density)

def chain_backwards(all_p_cond_backward, p_final):
    return flip(chain(list(all_p_cond_backward)[::-1], p_final))

def chain(all_p_cond, p_init):
    # [a -> Distribution(a)], Distribution(a) -> Distribution([a])
    @conslistgen
    def sample(rs):
        x = p_init.sample(rs)
        yield x
        for p_cond in all_p_cond:
            x = p_cond(x).sample(rs)
            yield x

    def log_density(xs):
        x0, x_rest = xs[0], xs[1:]
        log_prob = p_init.log_density(x0)
        x_prev = x0
        for x, p_cond in zip(x_rest, all_p_cond):
            log_prob += p_cond(x_prev).log_density(x)
            x_prev = x

        return log_prob

    return Distribution(sample, log_density)

def dangling_chain(all_p_cond):
    # [a -> Distribution(a)] -> (a -> Distribution([a]))
    def conditional_chain(x0):
        def sample(rs):
            x = x0
            xs = []
            for p_cond in all_p_cond:
                x = p_cond(x).sample(rs)
                xs.append(x)

            return xs

        def log_density(xs):
            log_prob = 0.0
            x_prev = x0
            for x, p_cond in zip(xs, all_p_cond):
                log_prob += p_cond(x_prev).log_density(x)
                x_prev = x

            return log_prob

        return Distribution(sample, log_density)

    return conditional_chain

def prod(dists):
    # [Distribution(a)] -> Distribution([a])
    def sample(rs):
        return [d.sample(rs) for d in dists]

    def log_density(xs):
        return sum([d.log_density(x) for d, x in zip(dists, xs)])

    return Distribution(sample, log_density)

def reshape(dist, do_reshape, undo_reshape):
    def sample(rs):
        return do_reshape(dist.sample(rs))

    def log_density(xs):
        return dist.log_density(undo_reshape(xs))

    return Distribution(sample, log_density)

def flip(reverse_sequence):
    # Distribution([a]) -> Distribution([a])
    def do_flip(seq):
        return list(seq)[::-1]

    return reshape(reverse_sequence, do_flip, do_flip)
