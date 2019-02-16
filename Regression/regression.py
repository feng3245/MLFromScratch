import sys
sys.path.append('../TestHarness')
sys.path.append('../EvalMetrics')
sys.path.append('../AlgEvaluation')
from testharness import evaluate_algorithm
from evalmetrics import rmse_metric
from split import train_test_split
from math import sqrt
def mean(values):
    return sum(values)/len(values)

def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

def covariance(x, mean_x, y, mean_y):
    return sum([(xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)])

def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean)/variance(x, x_mean)
    b0 = y_mean - b1* x_mean
    return [b0, b1]

def simple_linear_regression(train, test):
    b0, b1 = coefficients(train)
    return [ b0+b1*row[0] for row in test]

def predict(row, coefficients):
    return coefficients[0] + sum([b*x for b, x in zip(coefficients[1:],row)])

def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        coefupdates = [0.0 for i in range(len(train[0]))]
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error**2
            coefupdates[0] += (- l_rate * error)
            for i in range(len(row)-1):
                coefupdates[i+1] += (-l_rate * error * row[i])
        for i in range(len(coef)):
            coef[i] += coefupdates[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef

def linear_regression_sgd(train, test, l_rate, n_epoch):
    coef = coefficients_sgd(train, l_rate, n_epoch)
    return [predict(row, coef) for row in test]

if __name__ == '__main__':
    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    mean_x, mean_y = mean(x), mean(y)
    var_x, var_y = variance(x, mean_x), variance(y, mean_y)
    print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
    print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))

    covar = covariance(x, mean_x, y, mean_y)
    print('Covariance: %.3f' % (covar))

    b0, b1 = coefficients(dataset)
    for row in dataset:
        yhat = predict(row, [b0, b1])
        print('Expected=%.3f, Predicted=%.3f' % (row[-1], yhat))


    l_rate = 0.001
    n_epoch = 3000
    coef = coefficients_sgd(dataset, l_rate, n_epoch)
    print(coef)

    print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))
    rmse = evaluate_algorithm(dataset, simple_linear_regression, (lambda x, y: (x, x), 0), rmse_metric)
    print('RMSE: %.3f' % (rmse))

    sweden_auto_ins = [(108	,3925),(19	,462     ),(13	,157     ),(124	,4222),(40	,1194    ),(57	,1709    ),(23	,569     ),(14	,775     ),(45	,214     ),(10	,653     ),(5	,209     ),(48	,2481    ),(11	,235     ),(23	,396     ),(7	,488     ),(2	,66      ),(24	,1349    ),(6	,509     ),(3	,44      ),(23	,113     ),(6	,148     ),(9	,487     ),(9	,521     ),(3	,132     ),(29	,1039    ),(7	,775     ),(4	,118     ),(20	,981     ),(7	,279     ),(4	,381     ),(0	,0       ),(25	,692     ),(6	,146     ),(5	,403     ),(22	,1615    ),(11	,572     ),(61	,2176    ),(12	,581     ),(4	,126     ),(16	,596     ),(13	,899     ),(60	,2024    ),(41	,1813    ),(37	,1528    ),(55	,1628    ),(41	,734     ),(11	,213     ),(27	,926     ),(8	,761     ),(3	,399     ),(17	,1421    ),(13	,93      ),(13	,319     ),(15	,321     ),(8	,556     ),(29	,1333    ),(30	,1945    ),(24	,1379    ),(9	,874     ),(31	,2098    ),(14	,955     ),(53	,2446    ),(26	,1875    )]

    split = 0.6
    dataset = [[sai[0], sai[1]/10] for sai in sweden_auto_ins]
    rmse = evaluate_algorithm(dataset, simple_linear_regression, (train_test_split, split), rmse_metric)
    print('RMSE: %.3f' % (rmse))




