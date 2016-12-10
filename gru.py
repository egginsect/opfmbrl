"""Implements the long-short term memory character model.
This version vectorizes over multiple examples, but each string
has a fixed length."""

from __future__ import absolute_import
from __future__ import print_function
from builtins import range
from os.path import dirname, join
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.misc import logsumexp

from autograd.optimizers import adam
from rnn import string_to_one_hot, one_hot_to_string,\
                build_dataset, sigmoid, concat_and_multiply


def init_gru_params(input_size, state_size, output_size,
                     param_scale=0.01, rs=npr.RandomState(0)):
    def rp(*shape):
        return rs.randn(*shape) * param_scale

    return {'init hiddens': rp(1, state_size),
            'update':       rp(input_size + state_size + 1, state_size),
            'reset':      rp(input_size + state_size + 1, state_size),
            'hiddenOut':  rp(input_size + state_size + 1, state_size),
            'predict':      rp(state_size + 1, output_size)}

def gru_predict(params, inputs):
    def update_gru(input, hiddens):
        update  = sigmoid(concat_and_multiply(params['update'], input, hiddens))
        reset  = sigmoid(concat_and_multiply(params['reset'], input, hiddens))
        hiddens = (1-update)*hiddens+update*sigmoid(concat_and_multiply(params['hiddenOut'], input, hiddens*reset))
        return hiddens

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        return output - logsumexp(output, axis=1, keepdims=True) # Normalize log-probs.

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)

    output = [hiddens_to_output_probs(hiddens)]
    for input in inputs:  # Iterate over time steps.
        hiddens = update_gru(input, hiddens)
        output.append(hiddens_to_output_probs(hiddens))
    return output

def gru_log_likelihood(params, inputs, targets):
    logprobs = gru_predict(params, inputs)
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])
    return loglik / (num_time_steps * num_examples)


if __name__ == '__main__':
    num_chars = 128

    # Learn to predict our own source code.
    text_filename = join(dirname(__file__), 'gru.py')
    train_inputs = build_dataset(text_filename, sequence_length=30,
                                 alphabet_size=num_chars, max_lines=60)

    init_params = init_gru_params(input_size=128, output_size=128,
                                   state_size=40, param_scale=0.01)

    def print_training_prediction(weights):
        print("Training text                         Predicted text")
        logprobs = np.asarray(gru_predict(weights, train_inputs))
        for t in range(logprobs.shape[1]):
            training_text  = one_hot_to_string(train_inputs[:,t,:])
            predicted_text = one_hot_to_string(logprobs[:,t,:])
            print(training_text.replace('\n', ' ') + "|" +
                  predicted_text.replace('\n', ' '))

    def training_loss(params, iter):
        return -gru_log_likelihood(params, train_inputs, train_inputs)

    def callback(weights, iter, gradient):
        if iter % 10 == 0:
            print("Iteration", iter, "Train loss:", training_loss(weights, 0))
            print_training_prediction(weights)

    # Build gradient of loss function using autograd.
    training_loss_grad = grad(training_loss)

    print("Training GRU...")
    trained_params = adam(training_loss_grad, init_params, step_size=0.1,
                          num_iters=1000, callback=callback)

    print()
    print("Generating text from GRU...")
    num_letters = 30
    for t in range(20):
        text = ""
        for i in range(num_letters):
            seqs = string_to_one_hot(text, num_chars)[:, np.newaxis, :]
            logprobs = gru_predict(trained_params, seqs)[-1].ravel()
            text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
        print(text)
