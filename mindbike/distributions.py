import autograd.numpy as np
import autograd.numpy.random as npr

class Dist(object):
    def sample(self):
        # returns a sample
        # TODO: pass around seeds so that everything can be deterministic
        raise Exception("Class is abstract")

    def update(self, other):
        # return a new distribution which is the product of self and other
        raise Exception("Class is abstract")

class CatDist(Dist):
    def __init__(self, unnormalized_probs):
        assert np.all(unnormalized_probs >= 0.0), unnormalized_probs
        assert np.sum(unnormalized_probs) > 0.0 # TODO: consider relaxing these
        self.probs = unnormalized_probs / np.sum(unnormalized_probs)
        self.N = len(self.probs)

    def sample(self):
        return npr.choice(range(self.N), p=self.probs)

    def update(self, other):
        assert type(other) == CatDist, "Wrong type: {}".format(type(other))
        return CatDist(self.probs * other.probs)

class TransitionOperator(object):
    def propagate_dist(self, prev_dist):
        # return a distribution over the next state
        raise Exception("Class is abstract")

    def propagate_state(self, prev_state):
        # return a distribution over the next state
        raise Exception("Class is abstract")

class CatTransitionOperator(TransitionOperator):
    def __init__(self, unnormalized_transition_probs):
        # dims are (prev_state, next_state)
        self.transition_probs = (unnormalized_transition_probs /
                                 np.sum(unnormalized_transition_probs,
                                        axis=1, keepdims=True))

    def propagate_dist(self, prev_dist):
        return CatDist(np.dot(prev_dist.probs, self.transition_probs))

    def propagate_state(self, prev_state):
        return CatDist(self.transition_probs[prev_state, :])

class NormalDist(Dist):
    # TODO: Make this multivariate.  Use natural parameterization?
    def __init__(self, mean, var):
        assert np.all(var >= 0.0)
        self.mean = mean
        self.var = var

    def sample(self):
        return npr.randn() * np.sqrt(self.var) + self.mean

    def update(self, other):
        assert type(other) == NormalDist
        new_var = 1.0 / ( 1.0 / self.var + 1.0 / other.var)
        new_mean = new_var * (self.mean / self.var + other.mean / other.var)
        return NormalDist(new_mean, new_var)

class NormalTransitionOperator(TransitionOperator):
    def __init__(self, multiplicative, additive, noise):
        # dims are (prev_state, next_state)
        self.multiplicative = multiplicative
        self.additive = additive
        self.noise = noise

    def propagate_dist(self, prev_dist):
        assert type(prev_dist) == NormalDist
        # TODO: check this is correct.
        new_var = self.noise + np.dot(np.dot(self.multiplicative, prev_dist.var), self.multiplicative)
        return NormalDist(self.additive + np.dot(self.multiplicative, prev_dist.mean), new_var)

    def propagate_state(self, prev_state):
        # prev_state is a matrix of shape d
        return NormalDist(self.additive + np.dot(self.multiplicative, prev_state), self.noise)
