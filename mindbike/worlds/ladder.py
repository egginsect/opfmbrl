"""
Discrete, 1-D world. Can move forward, backward or not at all, and you can be
involuntarily moved too. Goal is to reach and remain in a particular spot.
"""

import numpy as np
import numpy.random as npr

from mindbike.big_picture import World, WorldModel, Policy
from mindbike.distributions import CatDist, CatTransitionOperator
from mindbike.dynamics.hmm import EM

class LadderWorld(World):
    def __init__(self, params):
        self.params = params # (init_state_params, dynamics_params, rendering_params)
        # Ideal recognition parameters. Just for plotting
        _, _, rendering_params = params
        self.recognition_params = ideal_recognition_params(rendering_params)

    def initial_state_dist(self):
        # initial_state_params dims are (cur_state, observed_state)
        init_state_params, _, _ = self.params
        return CatDist(init_state_params)

    def transition_operator(self, action):
        # dynamics_params dims are (action, cur_state, next_state)
        _, dynamics_params, _ = self.params
        return CatTransitionOperator(dynamics_params[action, :, :])

    def rendering_dist(self, state):
        # rendering_params dims are (cur_state, observed_state)
        _, _, rendering_params = self.params
        return CatDist(rendering_params[state, :])


class LadderWorldModel(LadderWorld, WorldModel):
    def __init__(self, params, history=()):
        self.params = params # (init_state_params, dynamics_params, rendering_params)
        self.history = history
        _, _, rendering_params = params
        self.recognition_params = ideal_recognition_params(rendering_params)

    def recognize(self, rendering):
        # TODO: replace this with a learned recognition model
        return CatDist(self.recognition_params[rendering, :])

def ideal_recognition_params(rendering_params):
    normalize = lambda X: X / X.sum(-1, keepdims=True)
    return normalize(normalize(rendering_params).T)

class LadderWorldPolicy(Policy):
    def __init__(self, params):
        self.params = params

    def react(self, belief):
        # params dims are (cur_state, action)
        sampled_state = belief.sample()
        reaction_dist = CatDist(self.params[sampled_state, :])
        return reaction_dist.sample()


def make_optimizers(num_iters, evals_per_iter, step_size):
    def optimize_world_model(world_model, experience):
        new_history = world_model.history + (experience,)
        return LadderWorldModel(EM(world_model.params, new_history), new_history)

    def optimize_policy(evaluate_policy, policy):
        def evaluate_policy_multiple_runs(test_policy):
            return sum(map(evaluate_policy, [test_policy] * evals_per_iter))

        cur_reward = evaluate_policy_multiple_runs(policy)
        for i in range(num_iters):
            test_policy_params = policy.params + step_size * npr.randn(*policy.params.shape)
            test_policy_params = np.maximum(1e-6, test_policy_params)  # ensure positive
            test_policy = LadderWorldPolicy(test_policy_params)
            test_reward = evaluate_policy_multiple_runs(test_policy)
            if test_reward > cur_reward:
                cur_reward = test_reward
                policy = test_policy

        return policy

    return optimize_world_model, optimize_policy
