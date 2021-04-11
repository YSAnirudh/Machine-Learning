import pandas as pd
import math
import numpy as np
import graphviz as gv
import copy
from time import time

discretization_fineness = 10
total_data_length = 0
percent_data_left = 0.001
max_depth = 0

#Class for the main Decision Tree Algorithm
class DecisionAlgo:
    #Constructor
    def __init__(self, data = None, filename = None, depth = 4, data_length = 0):
        #store for data (2D List)
        self._data = data
        #store for the class and its count
        self.class_count = dict()
        #store for the individual classes present
        self._classes = dict()
        #Assignment of no of samples in this node and features if data is present
        if (data):
            self._no_of_samples = len(data)
            self._features = len(data[0]) - 1
        else :
            self._no_of_samples = 0
            self._features = 0
        #if data is not provided check for a filename and pull data from it
        if (not data):
            if (filename):
                self.read(filename)
        #initialize tree
        self.tree = None
        #initialize max depth of the tree with the given argument, default = 4
        self.max_depth = depth
        #initialize data_length with the length of data at the root node 
        #for calculation of min no of samples to stop splitting
        self.data_length = data_length

    #Function to help read input from a file with (filename)
    #exit if error
    def read(self, filename:str):
        try :
            data = pd.read_csv(filename, header = None, sep=',|\s+|\t+|\s+\t+|\t+\s+', engine = 'python').values.tolist()
        except Exception:
            print("Error Opening File or File is Empty.")
            self._data = pd.DataFrame().values.tolist()
            exit()
        #assign the no of samples at this node and features
        self._no_of_samples = len(self._data)
        self._features = len(self._data[0]) - 1

    #Gets the unique classes at this node and assignes to predeclared dictionary _classes
    def find_classes(self):
        self._classes = dict()
        for i in range(len(self._data)):
            new_class = self._data[i][self._features]
            if (not new_class in self._classes):
                self._classes[new_class] = True

    #splits the data according to the given f_index and value (into 2 groups [left, right])
    def test_split(self, f_index, value):
        left = list()
        right = list()
        for row in range(len(self._data)):
            if (self._data[row][f_index] > value):
                right.append(self._data[row])
            else:
                left.append(self._data[row])
        return [left, right]

    #for finding the entropy of the groups and classes of the particular split
    def entropy_of_split(self, groups_and_classes):
        entropy = 0.0
        prob_of_class_in_group = [None for i in range(len(groups_and_classes))]
        for group_num in range(len(groups_and_classes)):
            #Normalized group size for a group (left or right)
            normalized_grp_size = len(groups_and_classes[group_num]) / self._no_of_samples
            
            #if it is 0 continue to the next group
            if (normalized_grp_size == 0.0):
                continue
            #get the probability of the classes in each group
            prob_of_class_in_group[group_num] = self.get_prob_of_class_in_group(groups_and_classes[group_num])
            group_sum = 0.0
            #for each class calculate group sum and add it to entropy
            for class_num in self._classes:
                entr = 0.0
                x = prob_of_class_in_group[group_num][class_num]
                if (x != 0.0):
                    entr = x * math.log2(x)
                group_sum = group_sum - entr
            entropy = entropy + normalized_grp_size * group_sum
        return entropy

    #given a group, get the probability of each class in the group
    def get_prob_of_class_in_group(self, group):
        prob_of_samples_of_class = dict().fromkeys(self._classes.keys(), 0.0)
        if (len(group) == 0):
            return prob_of_samples_of_class
        for i in range(len(group)):
            for j in prob_of_samples_of_class:
                if (group[i][self._features] == j):
                    prob_of_samples_of_class[j] = prob_of_samples_of_class[j] + 1

        for key in prob_of_samples_of_class:
            prob_of_samples_of_class[key] = prob_of_samples_of_class[key]/len(group)
        return prob_of_samples_of_class

    #gets the best split by taking the dataset and calculate the best entropy, split, and f_index of the split
    def get_best_split(self):
        self.find_classes()
        for key in self._classes:
            self.class_count[key] = 0
        range_for_features = self.get_range_for_features()
        best_min_entropy = math.inf
        best_f_index = -1
        best_value = -math.inf
        best_groups = [[],[]]
        
        for f_index in range(self._features):
            #10 steps as discretization fineness
            size_range = abs(range_for_features[f_index][1] - range_for_features[f_index][0])
            size_range = discretization_fineness if size_range == 0 else size_range
            numpy_help_range = np.arange(range_for_features[f_index][0],\
                range_for_features[f_index][1], size_range/discretization_fineness)
            
            for value in numpy_help_range:
                groups = self.test_split(f_index, value)
                entropy_of_split = self.entropy_of_split(groups)
            
                if (entropy_of_split < best_min_entropy):
                    best_min_entropy = entropy_of_split
                    best_f_index = f_index
                    best_value = value
                    best_groups = groups
        
        return [best_min_entropy, best_f_index, best_value, best_groups]

    #gets the range of each values for each feature
    def get_range_for_features(self):
        ranges_of_features = []
        for f_index in range(self._features):
            min_val = math.inf
            max_val = -math.inf
            for i in range(self._no_of_samples):
                val = self._data[i][f_index]
                if (val < min_val):
                    min_val = val
                if (val > max_val):
                    max_val = val
            ranges_of_features.append([min_val, max_val])
        return ranges_of_features

    #calls the make tree function on the root node
    def build_tree(self, entropy, f_index, value, data, depth):
        self.tree = self.make_tree(entropy, f_index, value, data, depth)

    #for making the tree for each node   
    def make_tree(self, entropy, f_index, value, data, depth):
        #if depth is greater than max_depth, end tree growth
        if (depth > self.max_depth):
            return None
        #initializes the particular node with the values
        tree = Node(entropy, f_index, value, data, depth)
        power = int(math.log10(self.data_length) - 1)
        #if length ar the node is less than a fracation of the dataset, end the tree growth
        # if (len(data[0]) + len(data[1]) < self.data_length * math.pow(self.data_length, -power)):
        if (len(data[0]) + len(data[1]) < self.data_length * percent_data_left):
            return tree
        #if entropy == 0, end the tree growth
        if (entropy == 0):
            return tree

        #expand on the left side of the tree by calling the same class
        left_node = DecisionAlgo(data[0], depth = self.max_depth, data_length = self.data_length)
        left_best = left_node.get_best_split()
        tree.left = self.make_tree(left_best[0], left_best[1], left_best[2], left_best[3], depth + 1)

        #expand on the right side of the tree by calling the same class
        right_node = DecisionAlgo(data[1], depth = self.max_depth, data_length = self.data_length)
        right_best = right_node.get_best_split()
        tree.right = self.make_tree(right_best[0], right_best[1], right_best[2], right_best[3], depth + 1)
        return tree

    #applies the DT algorithm and returns the dot function for drawing a graph 
    def algorithm(self):
        best_set = self.get_best_split()
        self.build_tree(best_set[0], best_set[1], best_set[2], best_set[3], 0)
        # dot = gv.Digraph(format='png')
        # dot.edge("[Root]", "[Root]" + "0" , "F" + str(self.tree.f_index) + " <= " + str(round(self.tree.value, 4)), arrowhead="open")
        # dot.edge("[Root]", "[Root]" + "1" , "F" + str(self.tree.f_index) + " > " + str(round(self.tree.value, 4)), arrowhead="open")
        self.tree.dfs()#"[Root]", dot)

    #returns the array containing a predicted class for each data point in test data
    def predict(self, test_data):
        x = copy.deepcopy(self.tree)
        prediction_array = [0 for i in range(len(test_data))]
        for i in range(len(test_data)):
            prediction_array[i] = self.dfs_to_get_class(x, test_data[i])
        return prediction_array
    
    #a dfs function for getting the class at the leaves
    def dfs_to_get_class(self, node, value):
        if (not node):
            return None
        if (value[node.f_index] <= node.value):
            if (node.left):
                return self.dfs_to_get_class(node.left, value)
        else:
            if (node.right):
                return self.dfs_to_get_class(node.right, value)
        return node.the_class
            
#### CLASS END ####

#calculate the accuracy given the prdiction array and test_data
def test_accuracy(pred, test_data):
    count = 0
    county = dict()
    count_class = dict()
    length = len(test_data[0]) - 1 
    for i in range(len(test_data)):
        if (pred[i] not in count_class):
            count_class[pred[i]] = 1
        else:
            count_class[pred[i]] = count_class[pred[i]] + 1
    for i in range(len(test_data)):
        if (test_data[i][length] == pred[i]):
            count = count + 1
        else:
            if (pred[i] not in county):
                county[pred[i]] = 1
            else:
                county[pred[i]] = county[pred[i]] + 1
    for key1 in county:
        county[key1] = county[key1]/count_class[key1]
    for key in county:
        print("Misclassification of class " + str(key) + ": " + str(round(county[key] * 100,4)))
        

    accuracy = count / len(test_data)
    return accuracy, [count, len(test_data) - count]
        
#splits the dataset in train and test data given dataset and ratio of the split (training ratio)
def train_test_split(dataset, ratio_of_split):
    data = copy.deepcopy(dataset)
    size_of_data = len(data)

    train_size = int(size_of_data * ratio_of_split)
    test_size = int(size_of_data * (1 - ratio_of_split))

    train_data = list()
    test_data = list()

    for i in range(train_size):
        random_int = np.random.randint(0, len(data))
        train_data.append(data[random_int])
        del(data[random_int])
    test_data = data
    return train_data, test_data

#splits the dataset into 'no_of_parts'
def k_fold_split(dataset, no_of_parts):
    data = copy.deepcopy(dataset)
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
            del(data[random_int])
    return k_data

#function to read the data from a file given filename
def read(filename:str):
    try :
        data = pd.read_csv(filename, header = None).values.tolist()
        if (data and type(data[0][0]) == str):
            try:
                data = pd.read_csv(filename, header = None, sep=',|\s+|\t+|\s+\t+|\t+\s+', engine = 'python').values.tolist()
            except:
                print("Error Opening File or File is Empty.")
                exit()
    except Exception:
        print("Error Opening File or File is Empty.")
        exit()
    return data

#Node class for tree construction
class Node:
    def __init__(self, entropy, f_index, value, data, depth):
        self.left = None
        self.right = None
        self.entropy = entropy
        self.f_index = f_index
        self.value = value
        self.data = data
        self.depth = depth
        self.class_count = dict()
        self.the_class = ''
        self.samples = 0
    
    #calculating the count for each class in the node data
    def calc_class_count(self):
        length = len(self.data[0]) - 1
        for j in range(len(self.data)):
            for i in self.data[j]:
                if (not i[-1] in self.class_count):
                    self.class_count[i[-1]] = 1
                else:
                    self.class_count[i[-1]] = self.class_count[i[-1]] + 1
        
        max = -math.inf
        class_key = None
        for key in self.class_count:
            if (self.class_count[key] > max):
                max = self.class_count[key]
                
                class_key = key
        return class_key
    
    #dfs for generating dot file
    def dfs(self): #, text, dot):
        self.the_class = self.calc_class_count()
        for key in self.class_count:
            self.samples = self.samples + self.class_count[key]
        # if (text == "[Root]"):
        #     details = \
        #         "Samples=" + str(self.samples) + \
        #         "\nClasses=" + str(self.class_count) + \
        #         "\nEntropy=" + str(round(self.entropy, 4))
        #     dot.node("[Root]", details, fillcolor="#ff9900", style="filled,rounded", shape="box")
        if (self.left):
            self.left.dfs()#text + "0", dot)
            # details = \
            #     "Samples=" + str(self.left.samples) + \
            #     "\nClasses=" + str(self.left.class_count) + \
            #     "\nEntropy=" + str(round(self.left.entropy, 4))
            # if (not self.left.left and not self.left.right):
            #     details = "Class\nPredicted=" + str(self.the_class) + \
            #         "\nClasses=" + str(self.left.class_count)
            #     dot.node(text + "0", details, fillcolor="#ffff00", style="filled,rounded", shape="box")
            # else:
            #     dot.edge(text + "0", text + "00" , "F" + str(self.left.f_index) + " <= " + str(round(self.left.value, 4)), arrowhead="open")
            #     dot.edge(text + "0", text + "01" , "F" + str(self.left.f_index) + " > " + str(round(self.left.value, 4)), arrowhead="open")
            #     dot.node(text + "0", details, fillcolor="#ff9900", style="filled,rounded", shape="box")
        if (self.right):
            self.right.dfs()#text + "1", dot)
            # details = \
            #     "Samples=" + str(self.right.samples) + \
            #     "\nClasses=" + str(self.right.class_count) + \
            #     "\nEntropy=" + str(round(self.right.entropy, 4))
            # if (not self.right.left and not self.right.right):
            #     details = "Class\nPredicted=" + str(self.the_class) + \
            #         "\nClasses=" + str(self.right.class_count)
            #     dot.node(text + "1", details, fillcolor="#ffff00", style="filled,rounded", shape="box")
            # else:
            #     dot.edge(text + "1", text + "10" , "F" + str(self.right.f_index) + " <= " + str(round(self.right.value, 4)), arrowhead="open")
            #     dot.edge(text + "1", text + "11" , "F" + str(self.right.f_index) + " > " + str(round(self.right.value, 4)), arrowhead="open")
            #     dot.node(text + "1", details, fillcolor="#ff9900", style="filled,rounded", shape="box")
        return

#algorithm to decide whether to apply bagging or not and the no of splits
def decision_tree_algo(dataset_filename, is_bagging = False, traintest_ratio = 0.9, bagged_splits = 2, max_depth = 100000):
    dataset = read(dataset_filename)
    total_data_length = len(dataset)
    train, test = train_test_split(dataset, traintest_ratio)
    print(len(train), len(test))
    if (not is_bagging):
        print("Generating Single Decision Tree")
        single_tree = DecisionAlgo(train, depth=max_depth, data_length = total_data_length)
        dot = single_tree.algorithm()
        # dot.render('SingleDecTreeNoDep-'+dataset_filename[:-4], format='png')
        print("Predicting and testing accuracy")
        predicted = single_tree.predict(test)
        accuracy, counts = test_accuracy(predicted, test)
        print("Accuracy of the Single Decision Tree Model:", accuracy)
        print("Predicted correctly =", counts[0], ",Predicted Incorrectly=", counts[1])
    else:
        k_train = None
        train_ref = train[:]
        k_train = k_fold_split(train_ref, bagged_splits)

        bagged_trees = [None for i in range(len(k_train))]
        predicted = [None for i in range(len(k_train))]
        #accuracy = [None for i in range(len(k_train))]
        # dot = [None for i in range(len(k_train))]
        #constructing k trees and generating dot files for each tree
        print("Generating Bagged Decision Trees")
        for j in range(len(k_train)):
            bagged_trees[j] = DecisionAlgo(k_train[j], depth=max_depth, data_length = total_data_length)
            bagged_trees[j].algorithm()
            predicted[j] = bagged_trees[j].predict(test)
            #accuracy[j] = test_accuracy(predicted[j], test)
            #print("Accuracy of the Tree No=" + str(j + 1) +" Bagged Decision Tree Model:", accuracy[j])
        #print()
        # for i in range(len(dot)):
        #     dot[i].render('BaggedDecTree'+str(i+1)+'-'+dataset_filename[:-4], format='png')
        
        print("Predicting and testing accuracy")
        class_prediction = [0.0 for j in range(len(predicted[0]))]

        #aggregation of the accuracy predicted by each tree by using statistical mode
        for k in range(len(predicted[0])):
            temp = list()
            for j in range(len(predicted)):
                temp.append(predicted[j][k])
            count = dict()
            for j in range(len(temp)):
                if (temp[j] not in count):
                    count[temp[j]] = 0
                else:
                    count[temp[j]] = count[temp[j]] + 1
            keyVal = max(count, key=count.get)

            class_prediction[k] = keyVal
        acc = 0.0
        counts = [0.0 for j in range(2)]
        #accuracy for the aggregate of the tree
        acc, counts = test_accuracy(class_prediction, test)
        print("Accuracy of the Bagged Decision Tree Model(" + str(bagged_splits) + " Splits):", acc)
        print("Predicted correctly =", counts[0], ",Predicted Incorrectly=", counts[1])

filename = input ("Enter filename:")
print("Decision Tree Algorithm for:", filename)
t1 = time()
decision_tree_algo(filename, is_bagging = False, traintest_ratio = 0.9, bagged_splits=9, max_depth=100)
t2 = time()

elapsed = t2 - t1
print("Time Taken:", elapsed)
