from csv import reader
import sys
from random import seed
sys.path.append('../DataScaling')
sys.path.append('../Baseline')
sys.path.append('../TestHarness')
sys.path.append('../AlgEvaluation')
sys.path.append('../EvalMetrics')
sys.path.append('../Regression')
sys.path.append('../Perceptron')
sys.path.append('../CART')
sys.path.append('../NaiveBayes')
sys.path.append('../K-Nearest')
sys.path.append('../LVQ')
from normalize import dataset_minmax, normalize_dataset
from standardize import column_means, column_stdevs, standardize_dataset
from baseline import zero_rule_algorithm_classification
from testharness import evaluate_algorithm
from split import train_test_split, cross_validation_split
from evalmetrics import accuracy_metric, mae_metric, rmse_metric, cross_entropy_loss
from regression import linear_regression_sgd
from perceptron import perceptron
from cart import decision_tree, get_leaves, prune_tree, build_tree
from naiveBayes import naive_bayes
from knearest import k_nearest_neighbors, manhattan_distance, minkowski_distance
from lvq import learning_vector_quantitization


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
        try:
            row[column] = float(row[column].strip())
        except:
            print(row[column], " can't be converted to float")

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

if __name__ == '__main__':
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
    n_epoch = 100
    
    scores = evaluate_algorithm(dataset, linear_regression_sgd, (cross_validation_split, n_folds), rmse_metric, l_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))
    
    
    from logisticregression import logistic_regression
    
    filename = 'pima-indians-diabetes.data.csv'
    dataset = load_csv(filename)
    print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    means = column_means(dataset)
    standardize_dataset(dataset, means, column_stdevs(dataset, means))
    
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    
    n_folds = 5
    l_rate = 0.05
    n_epoch = 100
    scores = evaluate_algorithm(dataset, logistic_regression, (cross_validation_split, n_folds), accuracy_metric, l_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    filename = 'sonar.all-data.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    
    str_column_to_int(dataset, len(dataset[0])-1)
    n_folds = 3
    l_rate = 0.01
    n_epoch = 500
    scores = evaluate_algorithm(dataset, perceptron, (cross_validation_split, n_folds), accuracy_metric,l_rate, n_epoch)
    
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    
    seed(1)
    
    filename = 'data_banknote_authentication.csv'
    
    dataset = load_csv(filename)
    
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    
    n_folds = 5
    max_depth = 20
    min_size = 1
    scores = evaluate_algorithm(dataset, decision_tree, (cross_validation_split, n_folds), accuracy_metric, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    scores = evaluate_algorithm(dataset, decision_tree, (cross_validation_split, n_folds), cross_entropy_loss, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Log loss: %s' % (sum(scores)/float(len(scores))))
    
    
    train, test = train_test_split(dataset, 0.2)
    tree = build_tree(train, max_depth, min_size)
    scores = evaluate_algorithm(dataset, decision_tree, (cross_validation_split, n_folds), accuracy_metric, max_depth, min_size, tree)
    
    bestavg = sum(scores)/float(len(scores))
    
    print('Existing average accuracy: %.3f%%' % bestavg)
    current_best = bestavg
    for i in range(1, max_depth):
        for j in range(len(get_leaves(tree, max_depth - i+1))):
            trimedtree = prune_tree(tree, max_depth - i + 1, j)
            scores = evaluate_algorithm(dataset, decision_tree, (cross_validation_split, n_folds), accuracy_metric, max_depth, min_size, trimedtree)
            if sum(scores)/float(len(scores)) > current_best:
                current_best = sum(scores)/float(len(scores))
                tree = trimedtree
            
    
    print('Pruned Mean Accuracy: %.3f%%' % current_best)
    
    filename = 'sonar.all-data.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    
    str_column_to_int(dataset, len(dataset[0])-1)
    
    train, test = train_test_split(dataset, 0.75)
    tree = build_tree(train, max_depth, min_size)
    scores = evaluate_algorithm(dataset, decision_tree, (cross_validation_split, n_folds), accuracy_metric, max_depth, min_size, tree)
    
    bestavg = sum(scores)/float(len(scores))
    
    print('Existing average accuracy: %.3f%%' % bestavg)
    current_best = bestavg
    for i in range(1, max_depth):
        for j in range(len(get_leaves(tree, max_depth - i+1))):
            trimedtree = prune_tree(tree, max_depth - i + 1, j)
            scores = evaluate_algorithm(dataset, decision_tree, (cross_validation_split, n_folds), accuracy_metric, max_depth, min_size, trimedtree)
            if sum(scores)/float(len(scores)) > current_best:
                current_best = sum(scores)/float(len(scores))
                tree = trimedtree
    
    print('Pruned Mean Accuracy: %.3f%%' % current_best)
    seed(1)
    
    filename = 'iris.data.csv'
    dataset = load_csv(filename)
    
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    str_column_to_int(dataset, len(dataset[0])-1)
    
    n_folds = 5
    scores = evaluate_algorithm(dataset, naive_bayes, (cross_validation_split, n_folds), accuracy_metric)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    
    filename = 'abalone.data.csv'
    dataset = load_csv(filename)
    for i in range(1, len(dataset[0])):
        str_column_to_float(dataset, i)
    
    str_column_to_int(dataset, 0)
    means = column_means(dataset)
    standardize_dataset(dataset, means, column_stdevs(dataset, means))
    
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    
    n_folds = 5
    num_neighbors = 10
    scores = evaluate_algorithm(dataset, k_nearest_neighbors, (cross_validation_split, n_folds), accuracy_metric, num_neighbors)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    scores = evaluate_algorithm(dataset, k_nearest_neighbors, (cross_validation_split, n_folds), rmse_metric, num_neighbors, False)
    print('Scores: %s' % scores)
    print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))
    
    scores = evaluate_algorithm(dataset, k_nearest_neighbors, (cross_validation_split, n_folds), accuracy_metric, num_neighbors, True, manhattan_distance)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    scores = evaluate_algorithm(dataset, k_nearest_neighbors, (cross_validation_split, n_folds), accuracy_metric, num_neighbors, True, lambda x, y: minkowski_distance(x, y, 4))
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    
    seed(1)
    # load and prepare data
    filename = 'ionosphere.data.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # evaluate algorithm
    n_folds = 5
    learn_rate = 0.3
    n_epochs = 50
    n_codebooks = 20
    scores = evaluate_algorithm(dataset, learning_vector_quantitization, (cross_validation_split, n_folds), accuracy_metric, n_codebooks, learn_rate, n_epochs)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    scores = evaluate_algorithm(dataset, learning_vector_quantitization, (cross_validation_split, n_folds), accuracy_metric, n_codebooks, [0.4, 0.3], [5, 45], 3)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
