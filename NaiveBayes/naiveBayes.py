import sys
from math import exp, sqrt, pi, log
sys.path.append('../DataScaling')
from standardize import column_means, column_stdevs
def separate_by_class(dataset):
    separated = dict()
    for c in set([d[-1] for d in dataset]):
        separated[c] = [d for d in dataset if d[-1] == c]
    return separated

def summarize_dataset(dataset):
    ds = [d[:-1] for d in dataset]
    return [(mean, std, len(ds), col) for mean, std, col in zip(column_means(ds), column_stdevs(ds, column_means(ds)), [d for d in zip(*dataset)])]

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries
def calculate_probability(x, mean, stdev, col):
    if isinstance(x, int):
        return 1-log(2-(col.count(x)/float(len(col))))
    exponent = exp(-((x-mean)**2/(2*stdev**2)))
    prob = ((1/ (sqrt(2*pi)*stdev))*exponent)
    if prob >= 1:
        return 1 
    return 1-log(2-((1/ (sqrt(2*pi)*stdev))*exponent))

def calculate_class_probabilities(summaries, row):
    total_rows = sum([summary[0][2] for label, summary in summaries.items()])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = class_summaries[0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count, col = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev, col)
    return probabilities

def predict(summaries, row):
    class_probabilities = [(class_value, probability) for class_value, probability in calculate_class_probabilities(summaries, row).items()]
    return max(class_probabilities, key=lambda cprob: cprob[1])[0]

def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    return [predict(summarize, row) for row in test]

if __name__ == '__main__':
    dataset = [[3.393533211,2.331273381,0],
[3.110073483,1.781539638,0],
[1.343808831,3.368360954,0],
[3.582294042,4.67917911,0],
[2.280362439,2.866990263,0],
[7.423436942,4.696522875,1],
[5.745051997,3.533989803,1],
[9.172168622,2.511101045,1],
[7.792783481,3.424088941,1],
[7.939820817,0.791637231,1]]
    separated = separate_by_class(dataset)
    for label in separated:
        print(label)
        for row in separated[label]:
            print(row)
    summary = summarize_dataset(dataset)
    print(summary)
    summary = summarize_by_class(dataset)
    for label in summary:
        print(label)
        for row in summary[label]:
            print(row)
    print(calculate_probability(1.0, 1.0, 1.0, []))
    print(calculate_probability(2.0, 1.0, 1.0, []))
    print(calculate_probability(0.0, 1.0, 1.0, []))
    probabilities = calculate_class_probabilities(summary, dataset[0])
    print(probabilities)


