from random import seed
from random import randrange

def random_algorithm(train, test):
    unique = list(set([row[-1] for row in train]))
    return [unique[randrange(len(unique))] for _ in test]

def zero_rule_mode(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    return [prediction for _ in test]

def zero_rule_algorithm_classification(train, test):
    return zero_rule_mode(train, test)

def zero_rule_algorithm_regression(train, test, tendency='mean'):
    train_labels = [row[-1] for row in train]
    if tendency == 'mean':
        return [sum([row[-1] for row in train])/float(len(train)) for _ in test]
    if tendency == 'mode':
        return zero_rule_mode(train, test)
    if tendency == 'median':
        train.sort()
        if len(train)%2 == 0:
            return [(train_labels[int(len(train_labels)/2)]+train_labels[int(len(train_labels)/2-1)])/2 for _ in test]
        else:
            return [train_labels[int(len(train_labels)/2)] for _ in test]

def predict_series_mean(train, test):
    means = []
    n = len(test)
    for i in range(len(train)-n+1):
        means.append(sum([row[-1] for row in train][i:i+n])/n)
    return sum(means)/len(means)

if __name__ == '__main__':
    seed(1)
    train = [[0], [1], [0], [1], [0], [1]]
    test = [[None], [None], [None], [None]]
    predictions = random_algorithm(train, test)
    print(predictions)
    predictions = zero_rule_algorithm_classification(train, test)
    print(predictions)
    train = [[10], [10], [12], [15], [18], [20]]
    predictions = zero_rule_algorithm_regression(train, test)
    print(predictions)
    predictions = zero_rule_algorithm_regression(train, test, 'mode')
    print(predictions)
    predictions = zero_rule_algorithm_regression(train, test, 'median')
    print(predictions)
    predictions = predict_series_mean(train, [[None], [None], [None]])
    print(predictions)

