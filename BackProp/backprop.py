from random import seed
from random import random
from math import exp
import sys
sys.path.append('../DataLoad')
sys.path.append('../EvalMetrics')
sys.path.append('../TestHarness')
from testharness import evaluate_algorithm
from load_data import load_csv, str_column_to_float, str_column_to_int
from split import train_test_split, cross_validation_split
from evalmetrics import accuracy_metric, mae_metric, rmse_metric, cross_entropy_loss
from normalize import dataset_minmax, normalize_dataset



def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] *inputs[i]
    return activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def transfer_derivative(output):
    return output * (1.0 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j]-neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i!= 0:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate-%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = max(set([row[-1] for row in train]))+1
    print(n_outputs)
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


if __name__ == '__main__':
    seed(1)
    network = initialize_network(2, 1, 2)
    for layer in network:
        print(layer)
    row = [1, 0, None]
    output = forward_propagate(network, row)
    print(output)
    expected = [0, 1]
    backward_propagate_error(network, expected)
    for layer in network:
        print(layer)
    dataset = [[2.7810836,2.550537003,0],
[1.465489372,2.362125076,0],
[3.396561688,4.400293529,0],
[1.38807019,1.850220317,0],
[3.06407232,3.005305973,0],
[7.627531214,2.759262235,1],
[5.332441248,2.088626775,1],
[6.922596716,1.77106367,1],
[8.675418651,-0.242068655,1],
[7.673756466,3.508563011,1]]
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, dataset, 0.5, 20, n_outputs)
    for layer in network:
        print(layer)
    for row in dataset:
        prediction = predict(network, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))

    filename = 'seeds_dataset.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        print(dataset[i])
        str_column_to_float(dataset, i)

    str_column_to_int(dataset, len(dataset[0])-1)

    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    
    
    
    n_folds = 5
    l_rate = 0.3
    n_epoch = 50
    n_hidden = 5
    scores = evaluate_algorithm(dataset, back_propagation,(cross_validation_split, n_folds), accuracy_metric, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
