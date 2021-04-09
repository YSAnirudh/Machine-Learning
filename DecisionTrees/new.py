import pandas as pd
import numpy as np
import math
from time import time
import copy
import graphviz as gv

discretization_fineness = 50
total_data_length = 0
percent_data_left = 0.01
max_depth = 0

class Node:
    def __init__(self, entropy, f_index, value, data, depth, classes):
        self.left = None
        self.right = None
        self.entropy = entropy
        self.f_index = f_index
        self.value = value
        self.data = data
        self.depth = depth
        self.classes = copy.deepcopy(classes)
        self.the_class = ''
    
    def calc_class_count(self):
        # length = len(self.data[0]) - 1
        # for j in range(len(self.data)):
        #     for i in self.data[j]:
        #         if (not i[-1] in self.classes):
        #             self.classes[i[-1]] = 1
        #         else:
        #             self.classes[i[-1]] = self.classes[i[-1]] + 1
        
        max = -math.inf
        class_key = None
        for key in self.classes:
            if (self.classes[key] > max):
                max = self.classes[key]
                class_key = key
        return class_key
    
    def dfs(self, text, dot):
        self.the_class = self.calc_class_count()
        dot.node(text, text + " " + str(self.the_class))
        if (not self):
            return
        print(self.classes)
        # print("Class: ", self.the_class)
        # print("Entr:", self.entropy, "F:", self.f_index, "Val:", self.value)
        # print("No of samples:", self.classes)
        # print("Depth:", self.depth)
        # print("Class:", self.the_class)
        # print(text)
        # print()
        if (self.left):
            self.left.dfs(text + "0", dot)
        if (self.right):
            self.right.dfs(text + "1", dot)
        dot.edge(text, text + "0")
        dot.edge(text, text + "1")
        return

def read(filename:str):
    try :
        data = pd.read_csv(filename, header = None, sep=',| ', engine = 'python')
    except Exception:
        print("Error Opening File or File is Empty.")
        exit()
    return data

def test_split(dataset, f_index, value):
    left = list()
    right = list()
    for row in range(len(dataset)):
        if (dataset[row][f_index] > value):
            right.append(dataset[row])
        else:
            left.append(dataset[row])

    return [np.asarray(left), np.asarray(right)]

def get_entropy_of_split(groups, classes:dict):
    entropy = 0.0
    for group in groups:
        #print(group.shape)
        if (group.size == 0):
            continue
        m_classes, m_classes_count = np.unique(group[:,-1], return_counts = True)
        class_count = dict(zip(m_classes, m_classes_count))
        #print(class_count)
        norm_group_size = len(group)/np.sum([len(groups[i]) for i in range(len(groups))])
        group_sum = 0.0
        for m_class in classes:
            x = class_count[m_class]/len(group) if (m_class in class_count) else 0.0
            if (x != 0):
                group_sum = group_sum - x*math.log2(x)
        entropy = entropy + norm_group_size*group_sum            
    return entropy

def get_best_split(dataset, classes:dict, range_for_features):
    entropy_data = 0.0
    for m_class in classes:
        x = classes[m_class]/len(dataset) if (m_class in classes) else 0.0
        if (x != 0):
            entropy_data = entropy_data - x*math.log2(x)
    #print (dataset[0])
    max_f_indices = len(dataset[0])-1
    best_min_entropy = math.inf
    best_f_index = -1
    best_value = -math.inf
    best_groups = None
    for f_index in range(max_f_indices):
        size_range = abs(range_for_features[f_index][1] - range_for_features[f_index][0])
        size_range = discretization_fineness if size_range == 0 else size_range
        numpy_help_range = np.arange(range_for_features[f_index][0],\
            range_for_features[f_index][1], size_range/discretization_fineness)
        for value in numpy_help_range:
            groups = test_split(dataset, f_index, value)
            entropy_of_split = get_entropy_of_split(groups, classes)
            if (entropy_of_split < best_min_entropy):
                best_min_entropy = entropy_of_split
                best_f_index = f_index
                best_value = value
                best_groups = groups
            continue
    print(len(best_groups[0]), len(best_groups[1]), best_value, best_f_index)
    return best_min_entropy, best_f_index, best_value, best_groups

def get_range_for_features(dataset):
    ranges_of_features = []
    for f_index in range(len(dataset[0])-1):
        min_val = math.inf
        max_val = -math.inf
        for i in range(len(dataset)):
            val = dataset[i][f_index]
            if (val < min_val):
                min_val = val
            if (val > max_val):
                max_val = val
        ranges_of_features.append([min_val, max_val])
    return ranges_of_features

def calc_classes(dataset):
    classes, count = np.unique(dataset[:,-1], return_counts = True)
    return dict(zip(classes, count))

def make_tree(entropy, f_index, value, data, depth, classes):
    if (depth > max_depth):
        return None
    tree = Node(entropy, f_index, value, data, depth, classes)
    if (entropy == 0):
        return tree
    if (len(data[0]) + len(data[1]) < total_data_length * percent_data_left):
        return tree

    classes_left = calc_classes(data[0])
    classes_right = calc_classes(data[1])
    # if (type(data[0]) == numpy.float64):
    #     print(data[0], data[1])
    range_for_features_left = get_range_for_features(data[0])
    range_for_features_right = get_range_for_features(data[1])

    left_entr, left_f_ind, left_value, left_groups = get_best_split(data[0], classes_left, range_for_features_left)
    tree.left = make_tree(left_entr, left_f_ind, left_value, left_groups, depth + 1, classes_left)

    right_entr, right_f_ind, right_value, right_groups = get_best_split(data[0], classes_right, range_for_features_right)
    tree.right = make_tree(right_entr, right_f_ind, right_value, right_groups, depth + 1, classes_left)
    return tree

def predict(tree, test_data):
    x = copy.deepcopy(tree)
    prediction_array = [0] * len(test_data)
    for i in range(len(test_data)):
        prediction_array[i] = dfs_for_class(tree, test_data[i])
        #print(prediction_array[i])
    
    return prediction_array

def dfs_for_class(node, value):
    if (not node):
        return None
    #print(value)
    if (value[node.f_index] <= node.value):
        if (node.left):
            return dfs_for_class(node.left, value)
    else:
        if (node.right):
            return dfs_for_class(node.right, value)
    return node.the_class

def build_decision_tree(dataset):
    total_data_length = len(dataset)
    #print(total_data_length)
    class_count = calc_classes(dataset)
    #print(class_count)
    range_for_features = get_range_for_features(dataset)
    #print(range_for_features)
    entr, f_ind, value, groups = get_best_split(dataset, class_count, range_for_features)
    tree = make_tree(entr, f_ind, value, groups, 0, class_count)
    dot = gv.Digraph(format='png')
    #print(tree.classes)
    tree.dfs("[Root]", dot)
    return tree, dot

def test_accuracy(pred, test_data):
    count = 0
    length = len(test_data[0]) - 1 
    for i in range(len(test_data)):
        if (test_data[i][length] == pred[i]):
            count = count + 1
    accuracy = count / len(test_data)
    return accuracy
        
def train_test_split(data, ratio_of_split):
    size_of_data = len(data)

    train_size = int(size_of_data * ratio_of_split)
    test_size = int(size_of_data * (1 - ratio_of_split))

    train_data = list()
    test_data = list()

    for i in range(train_size):
        random_int = np.random.randint(0, len(data))
        train_data.append(data[random_int])
        data = np.delete(data, random_int, 0)
    test_data = data
    return train_data, test_data

def k_fold_split(data, no_of_parts):
    k_data = [list() for i in range(no_of_parts)]
    total_data_size = len(data)
    for i in range(no_of_parts):
        if (total_data_size%no_of_parts == 0):
            number = int(total_data_size/no_of_parts)
        else:
            number = int(total_data_size/no_of_parts) + 1
        for j in range(number):
            if (len(data) == 0):
                return k_data
            random_int = np.random.randint(0, len(data))
            k_data[i].append(data[random_int])
            data = np.delete(data, random_int, 0)
    return k_data

def DecisionTreeAlgo(filename, is_bagging = False, train_split = 0.9):
    data = read(filename)
    dataset = data.to_numpy()
    #print(dataset)
    train, test = train_test_split(dataset, train_split)
    #print(len(train), len(test))
    #print(test)
    if (not is_bagging):
        tree, dot = build_decision_tree(dataset)
        dot.render('decision', format='png')
        prediction = predict(tree, test)
        #print(prediction)
        accuracy = test_accuracy(prediction, test)
        print("Accuracy of the Single Decision Tree Model:", accuracy)
    else:
        # Bagged Decision Tree
        return

t1 = time()
DecisionTreeAlgo('data_banknote_authentication.txt')
t2 = time()

print("Time", t2-t1)