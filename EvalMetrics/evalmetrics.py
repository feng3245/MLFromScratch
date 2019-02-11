from math import sqrt
def accuracy_metric(actual, predicted):
    return sum([a == p for a, p in zip(actual, predicted)])/float(len(actual)) * 100.0

def confusion_matrix(actual, predicted):
    unique = set(actual)
    matrix = [[0 for y in range(len(unique))] for x in range(len(unique))]

    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for a, p in zip(actual, predicted):
        x = lookup[a]
        y = lookup[p]
        matrix[y][x] += 1
    return lookup, matrix

def precision(xi, lookup, matrix):
    index = lookup[xi]
    return matrix[index][index]/float(sum(matrix[index]))

def recall(xi, lookup, matrix):
    index = lookup[xi]
    return matrix[index][index]/float(sum([row[index] for row in matrix]))

def f1(xi, lookup, matrix):
    return 2*(precision(xi, lookup, matrix)*recall(xi, lookup, matrix)/(precision(xi, lookup, matrix)+recall(xi, lookup, matrix)))

def print_confusion_matrix(unique, matrix):
    print('(A)' + ' '.join(str(x) for x in unique))
    print('(P)---')
    for i, x in enumerate(unique):
        print('%s| %s' % (x, ' '.join(str(x) for x in matrix[i])))

def mae_metric(actual, predicted):
    return sum([abs(p-a)for a, p in zip(actual, predicted)]) / float(len(actual))

def auc(fps, tps):
    auc = 0
    for i in range(len(fps)-1):
        auc += tps[i]*(fps[i+1]-fps[i])+(tps[i+1]-tps[i])*(fps[i+1]-fps[i])
    return auc

def chi_sqr(expected, predicted):
    return sum([(p-e)**2/e for e, p in zip(expected, predicted)])

def rmse_metric(actual, predicted):
    return sqrt(sum([abs(p-a)**2 for a, p in zip(actual, predicted)]) / float(len(actual)))
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,0,0,0,1,0,0,1,1]
accuracy = accuracy_metric(actual, predicted)
print(accuracy)

lookup, matrix = confusion_matrix(actual, predicted)
print_confusion_matrix(set(lookup.values()), matrix)

print(precision(1, lookup, matrix))
print(recall(1, lookup, matrix))
print(f1(1, lookup, matrix))

print(precision(0, lookup, matrix))
print(recall(0, lookup, matrix))
print(f1(0, lookup, matrix))

actual = [i/10.0 for i in range(1,6)]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
print(mae)

rmse = rmse_metric(actual, predicted)
print(rmse)

fps = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
tps = [0.2, 0.5, 0.6, 0.8, 0.85, 0.9, 1]

auc = auc(fps, tps)
print(auc)

chi_sqr = chi_sqr([20, 20, 30, 40, 60, 30], [30, 14, 34, 45, 57, 20])
print(chi_sqr)
