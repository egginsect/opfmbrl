from mindbike.probability import compose, chain, chain_backwards, ekf, flip, multiply_gaussians
from mindbike.util.misc import curry
from mindbike.util.conslist import cons, imap, izip, iunzip, head, tails, inits, scanl

@curry
def generative_step(update, render, action, zx_prev):
    z_prev, _ = zx_prev
    p_z = update(action, z_prev)
    return compose(render, p_z)

@curry
def generative_dist(update, render, p_z0, actions):
    p_zx_0 = compose(render, p_z0)
    return chain(imap(generative_step(update, render), actions), p_zx_0)

@curry
def filtering_step(update, recognize, prev_filtrate, action, observation):
    z_marginal, _ = prev_filtrate
    p_z_observation = recognize(observation)
    p_z_dynamics, backwards_conditional = ekf(update(action), z_marginal)
    new_z_marginal = multiply_gaussians(p_z_observation, p_z_dynamics)
    return new_z_marginal, backwards_conditional

def running_filter(update, recognize, p_z0, actions, observations):
    x0, x_rest = observations[0], observations[1:]
    z_marginal_0 = multiply_gaussians(p_z0, recognize(x0))
    z_marginal_rest, backwards_conditionals = iunzip(scanl(
        filtering_step(update, recognize), (z_marginal_0, None), actions, x_rest))
    z_marginals = cons(z_marginal_0, z_marginal_rest)
    prev_z_dists = imap(chain_backwards, inits(backwards_conditionals), z_marginals)
    return prev_z_dists, z_marginals

variational_dist = lambda *args: running_filter(*args)[0][-1]

def variational_lower_bound(rs, p, q, observations):
    zs = q.sample(rs)
    return p.log_density(zip(zs, observations)) - q.log_density(zs)

def look_both_ways(update, render, recognize, p_z0, actions, observations):
    past_dists, z_marginals = running_filter(update, recognize, p_z0, actions, observations)
    future_dists = imap(generative_dist(update, render), z_marginals, tails(actions))
    return izip(past_dists, future_dists)
