import autograd.numpy.random as npr
import autograd.numpy as np
from autograd import grad

a0 = 0.00
v0 = 0.1
observation_noise = 0.0
optimal_speed = 1.0


class CruiseControlWorld(object):
    def __init__(self):
        self.v = 0

    def reset(self, seed=0):
        self.v = 0

    def update(self, a):
        self.v = self.v + a0 + a
        v_obs = self.v + npr.randn() * observation_noise
        reward = - (self.v - v0)**2
        return [reward, v_obs]


def react(policy, belief_state):
    alpha, beta = policy
    _, v_believed = belief_state
    return alpha + beta * v_believed


def fit_dynamics_model(experiences):
    def state_estimator(belief_state, observation):
        return observation
    return CruiseControlWorld(), state_estimator


def gradient_ascent(learning_rate, num_iters):
    def optimize(f, x0):
        grad_fun = grad(f)
        x = x0.copy()
        for i in range(num_iters):
            x += learning_rate * grad_fun(x)
        return x
    return optimize


hypers = dict(
    dynamics=CruiseControlWorld(),

    react=react,
    fit_dynamics=fit_dynamics_model,
    optimize_policy=gradient_ascent(1e-2, 10),

    init_belief_state=np.zeros(2),
    init_policy=np.zeros(2),

    steps_per_day=10,
)


if __name__ == '__main__':
    def callback(day, experience):
        print "Day {} reward: {}".format(day, np.mean(map(reward, experience[-3:])))

    from big_picture import universe
    live_life, reward = universe(**hypers)
    live_life(100, callback)
