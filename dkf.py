from __future__ import absolute_import
from __future__ import print_function
from builtins import range
from os.path import dirname, join
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.misc import logsumexp

from autograd.optimizers import adam
from rnn import sigmoid, concat_and_multiply, rnn_predict, create_rnn_params
from gru import init_gru_params

import autograd.scipy.stats.norm as norm
from vae import diag_gaussian_log_density, sample_diag_gaussian, \
    relu, neural_net_predict, nn_predict_gaussian, init_net_params
import pickle

def likelihood_mean_field(emitparams, latents, data):
    p_mean, p_log_std = nn_predict_gaussian(emitparams, latents)
    return diag_gaussian_log_density(data, p_mean, p_log_std)

def getRNNLatentState(params, input):
    def update_rnn(input, hiddens):
        return np.tanh(concat_and_multiply(params['rnn']['change'], input, hiddens))

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['rnn']['init hiddens'], num_sequences, axis=0)
    output = [hiddens_to_output_probs(hiddens)]

    for input in inputs:  # Iterate over time steps.
        hiddens = update_rnn(input, hiddens)
        output.append((hiddens[:,:hiddens.shape[1]/2], hiddens[:,hiddens.shape[1]/2:]))
    return zip(*output)
def getGRUTranstionDist(latents, input, rs):
    pass
def createDKFparams(dataDims, hiddenDims, param_scale=0.01):
    params = dict()
    params['rnn'] = create_rnn_params(sum(dataDims.values()), hiddenDims['rnn']*2, hiddenDims['rnn'])
    params['emission'] = init_net_params(param_scale, (hiddenDims['rnn'], hiddenDims['emission'][0],
                                                       hiddenDims['emission'][1], dataDims['x']*2))
    params['transion'] = init_gru_params(hiddenDims['rnn']+dataDims['u']+dataDims['a'], hiddenDims['rnn'], hiddenDims['rnn'])
    params['policy'] = init_net_params(param_scale, (hiddenDims['rnn']+dataDims['x'], hiddenDims['policy'][0],
                                                       hiddenDims['policy'][1], dataDims['a']*2))
    return params

def dkf_lower_bd(params, input, rs):
    q_means, q_log_stds = getRNNLatentState(params, input)
    latents = sample_diag_gaussian(q_means, q_log_stds, rs)
    p_means, p_log_stds = getGRUTranstionDist(latents, input, rs)
    temporalKL = computeTemporalKL(q_means, q_log_stds, p_means, p_log_stds)
    likelihoodx = emission_dist(params, input, latents)
    likelihooda = policy_dist(params, input, latents)
    return likelihoodx + likelihooda - temporalKL

if __name__ == '__main__':
    with open('powerData.pkl') as f:
        X = pickle.load(f)
    inputDim = X.shape[1]
    seqLen = 100
    numSeq = 5
    fakeData = np.random.randn(seqLen,numSeq,inputDim)
    print(fakeData.shape)
    dataDims={'x':30,'u':1,'a':10}
    hiddenDims={'rnn':20, 'emission':(10,10), 'policy':(10,10), 'gru':None}
    params = createDKFparams(dataDims, hiddenDims)
    print(params)
