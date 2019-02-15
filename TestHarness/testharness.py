import sys
sys.path.append('../AlgEvaluation')
sys.path.append('../EvalMetrics')
from split import train_test_split
from evalmetrics import accuracy_metric
def evaluate_algorithm(dataset, algorithm, splitmethod, metric, *args):
    splitres = splitmethod[0](dataset, splitmethod[1])
    if type(splitres) == type(()):
        train, test = splitres
        test_set = [list(row[:-1]+[None]) for row in test]
        predicted = algorithm(train, test_set, *args)
        actual = [row[-1] for row in test]
        return metric(actual, predicted)
    else:
        folds = splitres
        return [metric([row[-1] for row in fold], algorithm(sum(folds[:i]+folds[i+1:], []), [row[:-1]+[None] for row in fold])) for i, fold in enumerate(folds)]




