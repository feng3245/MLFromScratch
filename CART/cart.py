def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))

    gini = 0.0

    for group in groups:
        size = float(len(group))

        if size == 0:
            continue
        score = 0.0

        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/size
            score += p*p
        gini += (1.0-score)*(size/n_instances)
    return gini

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini<b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%a[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train, test, max_depth, min_size, tree= None):
    if not tree:
        tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

def prune_tree(tree, depth, index):
    leaves = get_leaves(tree, depth)
    outcomes = get_values(leaves[index])
    leaves[index]['left'] =  leaves[index]['right']  = max(set(outcomes), key=outcomes.count)
    return tree

def get_values(node):
    return get_values(node['left']) if isinstance(node['left'], dict) else list([node['left']]) + get_values(node['right']) if isinstance(node['right'], dict) else list([node['right']])

def get_leaves(tree, depth):
    if not isinstance(tree, dict):
        return []
    if depth == 1:
        return list([tree['left']] if isinstance(tree['left'], dict) else []) +list([tree['right']] if isinstance(tree['right'], dict) else []) 
    else:
        return get_leaves(tree['left'], depth-1)+get_leaves(tree['right'], depth-1)

if __name__ == '__main__':
    print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
    print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
    dataset = [[2.771244718,1.784783929,0],
[1.728571309,1.169761413,0],
[3.678319846,2.81281357,0],
[3.961043357,2.61995032,0],
[2.999208922,2.209014212,0],
[7.497545867,3.162953546,1],
[9.00220326,3.339047188,1],
[7.444542326,0.476683375,1],
[10.12493903,3.234550982,1],
[6.642287351,3.319983761,1]]
    #split = get_split(dataset)
    #print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))

    tree = build_tree(dataset, 1, 1)
    print_tree(tree)

    tree = build_tree(dataset, 2, 1)
    print_tree(tree)

    tree = build_tree(dataset, 3, 1)
    print_tree(tree)

    leaves = get_leaves(tree, 1)
    print(leaves)

    tree = prune_tree(tree, 2, 1)
    print_tree(tree)
    stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}

    for row in dataset:
        prediction = predict(stump, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))
