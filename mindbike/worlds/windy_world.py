"""
Continuous 1-D world. Can move forward, backward or not at all, and you can be
involuntarily moved too. Goal is to reach and remain in a particular spot.
"""

import autograd.numpy as np
from autograd import grad
from copy import copy

from mindbike.big_picture import World, WorldModel
from mindbike.distributions import NormalDist, NormalTransitionOperator
from mindbike.util.optimizers import adam
from mindbike.neuralnet import make_nn_funs

class WindyWorld(World):
    def __init__(self, params):
        self.params = params  # (init_state_params, dynamics_params, rendering_params)
        self.initial_state_params, _, self.rendering_params = params

    def initial_state_dist(self):
        # initial_state_params are init_mean, init_var
        return NormalDist(*self.initial_state_params)

    def transition_operator(self, action):
        return NormalTransitionOperator(additive = np.tanh(action) * 1.3,
                                        multiplicative = 1.0,
                                        noise = 0.001)

    def rendering_dist(self, state):
        return NormalDist(state, self.rendering_params)

class WindyWorldModel(WindyWorld, WorldModel):
    def __init__(self, params, recognition_params, history=()):
        self.params = params   # (init_state_params, dynamics_params, rendering_params)
        self.history = history
        self.initial_state_params, self.dynamics_params, self.rendering_params = params
        self.recognition_params = recognition_params

    def recognize(self, rendering):
        return NormalDist(rendering, self.recognition_params)

policy_layer_sizes = [1, 10, 1]
num_policy_params, policy_predictions = make_nn_funs(policy_layer_sizes)

class WindyWorldPolicy(Policy):
    def __init__(self, params):
        self.params = params

    @staticmethod
    def num_policy_params():
        return num_policy_params

    def react(self, belief):
        return policy_predictions(self.params, belief.mean)[0]
        #return self.params[0] + self.params[1] * belief.mean # + self.params[1] * belief.var

def make_optimizers(num_iters, step_size):

    def optimize_world_model(world_model, experience):
        dynamics_params = copy(world_model.dynamics_params)
        #actions, renderings = zip(*experience)
        #for prev_rendering, action, rendering in \
        #    zip(renderings[:-1], actions[1:], renderings[1:]):
        #    dynamics_params[prev_rendering, action, rendering] += 1

        return WindyWorldModel((world_model.initial_state_params,
                                dynamics_params,
                                world_model.rendering_params),
                                world_model.recognition_params)

    def optimize_policy(evaluate_policy, init_policy):
        def evaluate_policy_params(policy_params, i):
            # i optionally indexes a minibatch
            cur_policy = WindyWorldPolicy(policy_params)
            return -evaluate_policy(cur_policy)
        updated_policy_params = adam(grad(evaluate_policy_params), copy(init_policy.params),
                                     step_size=step_size, num_iters=num_iters)
        return WindyWorldPolicy(updated_policy_params)

    return optimize_world_model, optimize_policy
