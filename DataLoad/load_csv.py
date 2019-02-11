from csv import reader
import sys
sys.path.append('../DataScaling')
from normalize import dataset_minmax, normalize_dataset
from standardize import column_means, column_stdevs, standardize_dataset

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
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

#minmax = dataset_minmax(dataset)

#normalize_dataset(dataset, minmax)
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
standardize_dataset(dataset, means, stdevs)
print(dataset[0])

