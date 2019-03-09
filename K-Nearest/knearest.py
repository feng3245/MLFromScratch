from math import sqrt


def euclidean_distance(row1, row2):
    return sqrt(sum([ (x1 - x2)**2 for x1, x2 in zip(row1[:-1], row2[:-1]) ]))

def manhattan_distance(row1, row2):
    return sum([abs(x1-x2) for x1, x2 in  zip(row1[:-1], row2[:-1])])

def minkowski_distance(row1, row2, q):
    return pow(sum([ abs(x1 - x2)**q for x1, x2 in zip(row1[:-1], row2[:-1]) ]), 1/9)

def get_neighbors(train, test_row, num_neighbors, distfunc):
    distances = [(row, distfunc(test_row, row)) for row in train]
    distances.sort(key=lambda tup: tup[1])
    return [dist[0] for dist in distances[:num_neighbors]]

def predict_classification(train, test_row, num_neighbors, distfunc):
    neighbors = get_neighbors(train, test_row, num_neighbors, distfunc)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def predict_regression(train, test_row, num_neighbors, distfunc):
    neighbors = get_neighbors(train, test_row, num_neighbors, distfunc)
    output_values = [row[-1] for row in neighbors]
    return sum(output_values)/float(len(output_values))

def k_nearest_neighbors(train, test, num_neighbors, classificiation = True, distfunc = euclidean_distance):
    if classificiation:
        return [predict_classification(train, row, num_neighbors, distfunc) for row in test]
    return [predict_regression(train, row, num_neighbors, distfunc) for row in test]

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
    row0 = dataset[0]
    for row in dataset:
        distance = euclidean_distance(row0, row)
        print(distance)
    neighbors = get_neighbors(dataset, dataset[0], 3, euclidean_distance)
    for neighbor in neighbors:
        print(neighbor)
    prediction = predict_classification(dataset, dataset[0], 3, euclidean_distance)
    print('Expected %d, Got %d.' % (dataset[0][-1], prediction))

