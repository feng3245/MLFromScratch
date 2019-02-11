from math import sqrt
def column_means(dataset):
    return [sum([row[i] for row in dataset])/float(len(dataset)) for i in range(len(dataset[0]))]

def column_stdevs(dataset, means):
    return [sqrt(sum([pow(row[i]-means[i], 2) for row in dataset])/float(len(dataset)-1)) for i in range(len(dataset[0]))]

def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i, r in enumerate(row):
            row[i] = (r - means[i])/stdevs[i]
    



