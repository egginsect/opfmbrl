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
import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm
from vae import diag_gaussian_log_density, sample_diag_gaussian, \
    relu, neural_net_predict, nn_predict_gaussian, init_net_params
import pickle
import pdb
import csv
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
    output = []
    for input in inputs:  # Iterate over time steps.
        hiddens = update_rnn(input, hiddens)
        output.append(computeMuSigma(hiddens, withSoftPlus))
    return zip(*output)

def getGRUTranstionDist(params, data, latents):
    inputs = np.concatenate(map(lambda x:np.expand_dims(x, axis=0), latents), axis=0)
    if('a' in data and 'u' in data):
        inputs = np.concatenate([data[k] for k in ['a','u']]+[inputs], axis=2)
    def update_gru(input, hiddens):
        update = sigmoid(concat_and_multiply(params['transion']['update'], input, hiddens))
        reset = sigmoid(concat_and_multiply(params['transion']['reset'], input, hiddens))
        hiddens = (1 - update) * hiddens + update * sigmoid(
            concat_and_multiply(params['transion']['hiddenOut'], input, hiddens * reset))
        return hiddens

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['transion']['init hiddens'], num_sequences, axis=0)

    output = []
    for input in inputs:  # Iterate over time steps.
        hiddens = update_gru(input, hiddens)
        output.append((hiddens[:,:hiddens.shape[1]/2], hiddens[:,hiddens.shape[1]/2:]))
    return zip(*output)


def createDKFparams(dataDims, hiddenDims, param_scale=0.01):
    params = dict()
    params['rnn'] = create_rnn_params(sum(dataDims.values()), hiddenDims['rnn']*2, hiddenDims['rnn'])
    params['emission'] = init_net_params(param_scale, (hiddenDims['rnn'], hiddenDims['emission'][0],
                                                       hiddenDims['emission'][1], dataDims['x']*2))
    params['transion'] = init_gru_params(dataDims['u']+dataDims['a']+hiddenDims['rnn'], hiddenDims['rnn']*2, hiddenDims['rnn']*2)
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
    p_meanAndstd = map(nn_predict_gaussian, [params['emission']]*input['x'].shape[0], latents)
    return map(diag_gaussian_log_density, input['x'], *zip(*p_meanAndstd))

def policyDist(params, input, latents):
    if 'a' in input:
        slices = [slice for slice in input['x']]
        combinedInput = map(np.hstack, zip(slices, latents))
        p_meanAndstd = map(nn_predict_gaussian, [params['policy']] * input['x'].shape[0], combinedInput)
        return map(diag_gaussian_log_density, input['a'], *zip(*p_meanAndstd))
    else:
        return [0]

def dkf_lower_bd(params, inputs, outputs, anneal=True, iter=None):
    q_means, q_log_stds = getRNNLatentState(params, inputs)
    latents = map(sample_diag_gaussian, q_means, q_log_stds)
    p_means, p_log_stds = getGRUTranstionDist(params, inputs, latents)
    temporalKL = computeTemporalKL(q_means, q_log_stds, p_means, p_log_stds)
    likelihoodx = emissionDist(params, outputs, latents)
    likelihooda = policyDist(params, outputs, latents)
    #print(np.mean(sum(likelihooda)))
    #print(np.mean(sum(likelihoodx)))
    #print(np.mean(sum(temporalKL)))
    #print('===========================')
    if(anneal):
        return (1-np.exp(-iter/100))*(np.mean(sum(likelihooda)) + np.mean(sum(likelihoodx))) - np.mean(sum(temporalKL))
    else:
        return np.mean(sum(likelihooda)) + np.mean(sum(likelihoodx)) - np.mean(sum(temporalKL))
if __name__ == '__main__':
    #with open('powerData.pkl') as f:
    #    X = pickle.load(f)
    with open('pendulous.pkl') as f:
        X = pickle.load(f)
    inputDim = 121
    seqLen = 200
    numSeq = 1
    step_size = 0.0001
    fakeData = np.random.randn(seqLen,numSeq,inputDim)
    print(fakeData.shape)

    frame_to_vect = lambda frame: np.reshape(np.arctanh(2.0 * frame - 1.0), 121)
    vect_to_frame = lambda vect: np.reshape(0.5 * np.tanh(vect) + 0.5, (11, 11))

    dataDims={'x':80,'u':1,'a':20}
    dataDims={'x':121,'u':0,'a':0}
    X = map(frame_to_vect, X)
    X = np.concatenate(map(lambda x:np.expand_dims(x, axis=0), X), axis=0)
    inputs = {'x':np.expand_dims(X, axis=0)}
    outputs = {'x':inputs['x'][:,1:,:]}
    inputs['x'] = inputs['x'][:,:-1,:]
    #inputs = {'x': X}
    #inputs = {'x':fakeData[:,:,:dataDims['x']], 'u':fakeData[:,:,dataDims['x']:dataDims['x']+dataDims['u']],
    #          'a':fakeData[:,:,:dataDims['a']]}
    hiddenDims={'rnn':20, 'emission':(10,10), 'policy':(10,10), 'transition':None}
    params = createDKFparams(dataDims, hiddenDims)
    def training_loss(params, iter):
        return -dkf_lower_bd(params, inputs, outputs, True, iter)/(inputs['x'].shape[1]*inputs['x'].shape[0])
    def training_loss_noAnneal(params):
        return -dkf_lower_bd(params, inputs, outputs, anneal=False) / (inputs['x'].shape[1])
    def print_perf(params, iter, grad):
        if iter % 10 == 0:
            bound = training_loss_noAnneal(params)
            print("{:15}|{:20}".format(iter, bound))
        with open(r'dkfTrainTest.csv', 'a') as f:
            btrain = np.mean(training_loss_noAnneal(params))
            if btrain>1:
                btrain=1
            if btrain<-1:
                btrain=-1
            writer = csv.writer(f)
            writer.writerow([btrain])
    training_loss_grad = grad(training_loss)
    #pdb.set_trace()
    trained_params = adam(training_loss_grad, params, step_size=0.05, num_iters=1000, callback=print_perf)
    def plotTrainingCurve():
        X=np.genfromtxt(r'dkfTrainTest.csv',delimiter=',')
        t=np.arange(X.shape[0])
        plt.clf()
        plt.plot(t,X)
        #plt.plot(t,X[:,1])
        #plt.legend(['Train', 'Test'])
        plt.savefig('trainingCurvedkf.jpg')
    plotTrainingCurve()
