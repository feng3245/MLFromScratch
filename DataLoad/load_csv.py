from csv import reader
import sys
sys.path.append('../DataScaling')
sys.path.append('../Baseline')
sys.path.append('../TestHarness')
sys.path.append('../AlgEvaluation')
sys.path.append('../EvalMetrics')
sys.path.append('../Regression')
from normalize import dataset_minmax, normalize_dataset
from standardize import column_means, column_stdevs, standardize_dataset
from baseline import zero_rule_algorithm_classification
from testharness import evaluate_algorithm
from split import train_test_split, cross_validation_split
from evalmetrics import accuracy_metric, mae_metric, rmse_metric
from regression import linear_regression_sgd
def load_csv(filename, delimiter = ','):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file, delimiter=delimiter)
        for row in csv_reader:
            if row:
                dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


filename = 'iris.data.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))

print(dataset[0])

for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
print(dataset[0])

lookup = str_column_to_int(dataset, len(dataset[0])-1)
print(dataset[0])
print(lookup)

filename = 'pima-indians-diabetes.data.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
print(dataset[0])
split = 0.6
accuracy = evaluate_algorithm(dataset, zero_rule_algorithm_classification, (train_test_split,split), accuracy_metric)
print('Accuracy: %.3f%%' % (accuracy))

n_folds = 5
scores = evaluate_algorithm(dataset, zero_rule_algorithm_classification, (cross_validation_split, n_folds), accuracy_metric)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/len(scores)))

score_dev = column_stdevs([[s] for s in scores], [(sum(scores)/len(scores))])

print('Standard deviation ', score_dev)


scores = evaluate_algorithm(dataset, zero_rule_algorithm_classification, (cross_validation_split, n_folds), mae_metric)


print('Mae scores: %s' % scores)
print('Mean absolute error: %.3f' % (sum(scores)/len(scores)))


means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
standardize_dataset(dataset, means, stdevs)
print(dataset[0])

filename = 'winequality-white.csv'
dataset = load_csv(filename, ';')[1:]
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

minmax = dataset_minmax(dataset)
#normalize_dataset(dataset, minmax)
means = column_means(dataset)
standardize_dataset(dataset, means, column_stdevs(dataset, means))
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

n_folds = 5
l_rate = 0.000001
n_epoch = 5000

scores = evaluate_algorithm(dataset, linear_regression_sgd, (cross_validation_split, n_folds), rmse_metric, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))
