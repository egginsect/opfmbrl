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
import pdb

def likelihood_mean_field(emitparams, latents, data):
    p_mean, p_log_std = nn_predict_gaussian(emitparams, latents)
    return diag_gaussian_log_density(data, p_mean, p_log_std)

softplus = lambda x: np.log(1/np.exp(x))

def getRNNLatentState(params, data, withSoftPlus=False):
    def update_rnn(input, hiddens):
        return np.tanh(concat_and_multiply(params['rnn']['change'], input, hiddens))
    inputs = np.concatenate(data.values(), axis=2)
    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['rnn']['init hiddens'], num_sequences, axis=0)
    def computeMuSigma(hiddens, withSoftPlus):
        if withSoftPlus:
            #With softplus, output mu and sigma
            return softplus(neural_net_predict(params['rnn']['mu'], hiddens)), \
                   softplus(neural_net_predict(params['rnn']['sigma'], hiddens))
        else:
            #without softplus, split hiddent state to mu and logstd
            return hiddens[:,:hiddens.shape[1]/2], hiddens[:,hiddens.shape[1]/2:]
    output = [computeMuSigma(hiddens, withSoftPlus)]
    for input in inputs:  # Iterate over time steps.
        hiddens = update_rnn(input, hiddens)
        output.append(computeMuSigma(hiddens, withSoftPlus))
    return zip(*output)

def getGRUTranstionDist(params, data):
    inputs = np.concatenate([data[k] for k in ['a','u']], axis=2)
    def update_gru(input, hiddens):
        update = sigmoid(concat_and_multiply(params['transion']['update'], input, hiddens))
        reset = sigmoid(concat_and_multiply(params['transion']['reset'], input, hiddens))
        hiddens = (1 - update) * hiddens + update * sigmoid(
            concat_and_multiply(params['transion']['hiddenOut'], input, hiddens * reset))
        return hiddens

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['transion']['predict'], hiddens)
        return output - logsumexp(output, axis=1, keepdims=True)  # Normalize log-probs.

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['transion']['init hiddens'], num_sequences, axis=0)

    output = [(hiddens[:,:hiddens.shape[1]/2], hiddens[:,hiddens.shape[1]/2:])]
    for input in inputs:  # Iterate over time steps.
        hiddens = update_gru(input, hiddens)
        output.append((hiddens[:,:hiddens.shape[1]/2], hiddens[:,hiddens.shape[1]/2:]))
    return zip(*output)


def createDKFparams(dataDims, hiddenDims, param_scale=0.01):
    params = dict()
    params['rnn'] = create_rnn_params(sum(dataDims.values()), hiddenDims['rnn']*2, hiddenDims['rnn'])
    params['emission'] = init_net_params(param_scale, (hiddenDims['rnn'], hiddenDims['emission'][0],
                                                       hiddenDims['emission'][1], dataDims['x']*2))
    params['transion'] = init_gru_params(dataDims['u']+dataDims['a'], hiddenDims['rnn']*2, hiddenDims['rnn']*2)
    params['policy'] = init_net_params(param_scale, (hiddenDims['rnn']+dataDims['x'], hiddenDims['policy'][0],
                                                       hiddenDims['policy'][1], dataDims['a']*2))
    return params

def computeTemporalKL(q_means, q_log_stds, p_means, p_log_stds):
    def KLgaussian(q_mean, q_log_std, p_mean, p_log_std):
        mu_diff = p_mean-q_mean
        return (logsumexp(p_log_std, axis=1)-logsumexp(q_log_std, axis=1)-1
                +np.sum(np.exp(q_log_std)/np.exp(p_log_std), axis=1)
                +np.sum(mu_diff**2/np.exp(p_log_std), axis=1))
    temporalKLs=map(KLgaussian, q_means, q_log_stds, p_means, p_log_stds)
    return temporalKLs

def emissionDist(params, input, latents):
    p_meanAndstd = map(nn_predict_gaussian, [params['emission']]*input['x'].shape[0], latents[1:])
    return map(diag_gaussian_log_density, input['x'], *zip(*p_meanAndstd))

def policyDist(params, input, latents):
    print(type(input))
    slices = [slice for slice in input['x']]
    combinedInput = map(np.hstack, zip(slices, latents))
    p_meanAndstd = map(nn_predict_gaussian, [params['policy']] * input['x'].shape[0], combinedInput)
    return map(diag_gaussian_log_density, input['a'], *zip(*p_meanAndstd))

def dkf_lower_bd(params, input):
    q_means, q_log_stds = getRNNLatentState(params, input)
    latents = map(sample_diag_gaussian, q_means, q_log_stds)
    p_means, p_log_stds = getGRUTranstionDist(params, input)
    temporalKL = computeTemporalKL(q_means, q_log_stds, p_means, p_log_stds)
    likelihoodx = emissionDist(params, input, latents)
    likelihooda = policyDist(params, input, latents)
    return (np.mean(sum(likelihooda)) + np.mean(sum(likelihoodx)) - np.mean(sum(temporalKL)))/input['x'].shape[2]

if __name__ == '__main__':
    with open('powerData.pkl') as f:
        X = pickle.load(f)
    inputDim = 101
    seqLen = 100
    numSeq = 5
    fakeData = np.random.randn(seqLen,numSeq,inputDim)
    print(fakeData.shape)
    dataDims={'x':80,'u':1,'a':20}
    inputs = {'x':fakeData[:,:,:dataDims['x']], 'u':fakeData[:,:,dataDims['x']:dataDims['x']+dataDims['u']],
              'a':fakeData[:,:,:dataDims['a']]}
    hiddenDims={'rnn':20, 'emission':(10,10), 'policy':(10,10), 'transition':None}
    params = createDKFparams(dataDims, hiddenDims)
    q_means, q_log_stds = getRNNLatentState(params, inputs)
    #pdb.set_trace()
    p_means, p_log_stds = getGRUTranstionDist(params, inputs)
    latents = map(sample_diag_gaussian, q_means, q_log_stds)
    temporalKL = computeTemporalKL(q_means, q_log_stds, p_means, p_log_stds)
    bd = dkf_lower_bd(params, inputs)
    #likelihoodx = emissionDist(params, input, latents)
    #pdb.set_trace()
    likelihooda = policyDist(params, inputs, latents)
    pdb.set_trace()
    out = emissionDist(params, inputs, latents)
    for item in inputs['x']:
        print(item.shape)
    print(params)
