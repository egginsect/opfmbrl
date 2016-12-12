# Implements auto-encoding variational Bayes.

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.optimizers import adam
import pickle
import sys
from tsne import *
import csv
import matplotlib.pyplot as plt
import pdb
def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal cGaussian.
    D = np.shape(params)[-1] / 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std, rs=npr.RandomState(0)):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def bernoulli_log_density(targets, unnormalized_logprobs):
    # unnormalized_logprobs are in R
    # Targets must be -1 or 1
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*targets)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 0.5 * (np.tanh(x) + 1)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def neural_net_predict(params, inputs, activation=relu):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""
    for W, b in params[:-1]:
        outputs = batch_normalize(np.dot(inputs, W) + b)  # linear transformation
        inputs = activation(outputs)                            # nonlinear transformation
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb
    return outputs

def nn_predict_gaussian(params, inputs):
    # Returns means and diagonal variances
    return unpack_gaussian_params(neural_net_predict(params, inputs))

def generate_from_prior(gen_params, num_samples, noise_dim, rs):
    latents = rs.randn(num_samples, noise_dim)
    return sigmoid(neural_net_predict(gen_params, latents))

def p_images_given_latents(gen_params, images, latents):
    p_mean, p_log_std = nn_predict_gaussian(gen_params, latents)
    return diag_gaussian_log_density(images, p_mean, p_log_std)
    # preds = neural_net_predict(gen_params, latents)
    # return bernoulli_log_density(images, preds)

def vae_lower_bound(gen_params, rec_params, data, rs):
    # We use a simple Monte Carlo estimate of the KL
    # divergence from the prior.
    q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
    latents = sample_diag_gaussian(q_means, q_log_stds, rs)
    q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
    p_latents = diag_gaussian_log_density(latents, 0, 1)
    likelihood = p_images_given_latents(gen_params, data, latents)
    return np.mean(p_latents + likelihood - q_latents)

def partitionTrainTest(X):
    Idx = np.arange(X.shape[0])
    truncate = np.ceil(X.shape[0]*0.7)
    return X[Idx[:truncate],:], X[Idx[truncate:],:]

def visualizeLatentState(X, rs, gen_params, rec_params):
    q_means, q_log_stds = nn_predict_gaussian(rec_params, X)
    latents = sample_diag_gaussian(q_means, q_log_stds, rs)
    gen = sigmoid(neural_net_predict(gen_params, latents))
    gen = gen[:,:gen.shape[1]/2]
    print(gen.shape)
    print(X.shape)
    #yTrain =y[:genTrain.shape[0],:]
    #yTest = y[genTrain.shape[0]:,:]
    #pdb.set_trace
    pdb.set_trace()
    y = tsne(np.vstack((X,gen*100)))
    plt.figure()
    plt.clf()
    plt.scatter(y[:gen.shape[0],0],y[:gen.shape[0],1],color='red')
    plt.scatter(y[gen.shape[0]:,0],y[gen.shape[0]:,1],color='blue')
    plt.legend(['X', 'Xdecoded'],)
    plt.savefig('hidden.jpg')

if __name__ == '__main__':
    # Model hyper-parameters
    with open('powerData.pkl') as f:
        X = pickle.load(f)
    X=np.array(X)
    #pdb.set_trace()
    Xtrain, Xtest =  partitionTrainTest(X)
    latent_dim = 8
    data_dim = X.shape[1] # How many pixels in each image (28x28).
    gen_layer_sizes = [latent_dim, 300, 200, data_dim * 2]
    rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]

    # Training parameters
    param_scale = 0.01
    batch_size = 200
    num_epochs = 2000
    step_size = 0.001

    init_gen_params = init_net_params(param_scale, gen_layer_sizes)
    init_rec_params = init_net_params(param_scale, rec_layer_sizes)
    combined_init_params = (init_gen_params, init_rec_params)

    num_batches = int(np.ceil(len(Xtrain) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    seed = npr.RandomState(0)
    def objective(combined_params, iter):
        data_idx = batch_indices(iter)
        gen_params, rec_params = combined_params
        return -vae_lower_bound(gen_params, rec_params, Xtrain[data_idx], seed) / len(Xtrain[data_idx])
    def trainloss(combined_params, iter):
        gen_params, rec_params = combined_params
        return -vae_lower_bound(gen_params, rec_params, Xtrain, seed) / len(Xtrain)

    def testloss(combined_params, iter):
        gen_params, rec_params = combined_params
        return -vae_lower_bound(gen_params, rec_params, Xtest, seed) / len(Xtest)

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)
    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
    def print_perf(combined_params, iter, grad):
        if iter % 10 == 0:
            gen_params, rec_params = combined_params
            bound = np.mean(objective(combined_params, iter))
            print("{:15}|{:20}".format(iter//num_batches, bound))
            fake_data = generate_from_prior(gen_params, 20, latent_dim, seed)
        with open(r'vaeTrainTest.csv', 'a') as f:
            btrain = np.mean(trainloss(combined_params, iter))
            btest = np.mean(testloss(combined_params, iter))
            writer = csv.writer(f)
            writer.writerow([btrain, min(btest,100)])
    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, combined_init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)
    visualizeLatentState(X, seed, *combined_init_params)
    def plotTrainingCurve():
        X=np.genfromtxt(r'vaeTrainTest.csv',delimiter=',')
        t=np.arange(X.shape[0])
        plt.clf()
        plt.plot(t,X[:,0])
        plt.plot(t,X[:,1])
        plt.legend(['Train', 'Test'])
        plt.savefig('trainingCurve.jpg')
    plotTrainingCurve()
