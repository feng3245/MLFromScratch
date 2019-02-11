def dataset_minmax(dataset):
    return [[min([row[i] for row in dataset]), max([row[i] for row in dataset])] for i, row in enumerate(dataset[0])]

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][1]-minmax[i][0])

