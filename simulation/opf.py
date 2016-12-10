import cvxpy as cvx
import numpy as np
class OPF(object):
    def __init__(self, params, constraints = None):
        self.params = params
        self.vars = dict()
        self.vars['g'] = cvx.Variable(params['numG'], self.params['T'])
        self.vars['q'] = cvx.Variable(params['numC'], self.params['T'])
        self.vars['c'] = cvx.Variable(params['numC'], self.params.T)
        if constraints:
            self.constraints = constraints
        else:
            self.constraints = list()
    def constructProblem(self, withRandom):
        pass
    def generatorLocal(self, construction):
        A = self.params['A'] + construction*np.random.uniform(-0.5, 1, len(self.params['A']))
        B = self.params['B'] + construction*np.random.uniform(-1, 1, len(self.params['A']))

        for i in xrange(self.params['numG']):
            self.loss = self.loss + cvx.sum_entries(A * (self.vars['g'][i, :]) ** 2 + B * (self.vars['g'][i, :]))
            Pmax = self.params['Pmax'] + np.random.randn()
            self.constraints.append(self.vars['g'][i, :] <= Pmax)

    def capacitorLocal(self):
        pass

    def encodeConstraint(self):
        pass

    @classmethod
    def decodeConstraint(self):
        pass
