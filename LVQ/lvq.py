from math import sqrt
from random import randrange, seed
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def get_best_matching_unit(codebooks, test_row, bmus = 1):
    distances = list()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key = lambda tup: tup[1])
    return [distances[i][0] for i in range(bmus)]


def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(n_records)][i] for i in range(n_features)]
    return codebook

def train_codebooks(train, n_codebooks, lrate, epochs, codebooks = None, bmus=1):
    if not codebooks:
        codebooks = [random_codebook(train) for i in range(n_codebooks)]
    for epoch in range(epochs):
        rate = lrate * (1.0-(epoch/float(epochs)))
        sum_error = 0.0
        for row in train:
            for bmu in get_best_matching_unit(codebooks, row, bmus):
                for i in range(len(row)-1):
                    error = row[i] - bmu[i]
                    sum_error += error**2
                    if bmu[-1] == row[-1]:
                        bmu[i] += rate*error
                    else:
                        bmu[i] -= rate*error
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
    return codebooks

def predict(codebooks, test_row):
    bmus =  get_best_matching_unit(codebooks, test_row)
    return bmus[0][-1]

def learning_vector_quantitization(train, test, n_codebooks, lrate, epochs, n_bmus = 1):
    if not isinstance(lrate, list):
        codebooks = train_codebooks(train, n_codebooks, lrate, epochs, bmus = n_bmus)
    else:
        codebooks = None
        for lr, eps in zip(lrate, epochs):
            codebooks = train_codebooks(train, n_codebooks, lr, eps, codebooks, n_bmus)

    predictions = list()
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
    return predictions

if __name__ == '__main__':
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
    test_row = dataset[0]
    bmu = get_best_matching_unit(dataset, test_row)
    print(bmu)
    seed(1)
    learn_rate = 0.3
    n_epochs = 10
    n_codebooks = 2
    codebooks = train_codebooks(dataset, n_codebooks, learn_rate, n_epochs)
    print('Codebooks: %s' % codebooks)


