import numpy as np
from time import time
import pandas as pd
import math

discretization_fineness = 10

class DecisionTree:
    def __init__(self, feature_data, features, classes_data, classes):
        self._feature_data = feature_data
        self._features = features
        self._classes_data = classes_data
        self._classes = classes
        self._n_samples = self._feature_data.shape[1]
        self._n_features = self._features.shape[0]
        self._n_classes = self._classes.shape[0]

    def print_data(self):
        print(self._feature_data)
        print(self._features)
        print(self._classes_data)
        print(self._classes)
        print(self._n_samples)
        print(self._n_features)
        print(self._n_classes)
    


def read(filename:str):
    try :
        data = pd.read_csv(filename, header = None, sep=',| ', engine = 'python')
    except Exception:
        print("Error Opening File or File is Empty.")
        exit()
    return data #pandas.DataFrame

def parse_data(dataset):
    actual_classes = dict()
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]-1
    n_classes = 0
    features = np.ndarray(shape = (n_features))
    feature_data = np.ndarray(shape = (n_features, n_samples))
    classes_data = np.ndarray(shape = (n_samples))
    
    feature = dataset[:,:n_features]
    feature_data = np.transpose(feature)
    classes_data = dataset[:,-1]
    
    classes = np.ndarray(shape = (n_classes))
    classes = np.unique(classes_data)

    for i in range(len(actual_classes)):
        classes[i] = i
    for i in range(n_features):
        features[i] = i
    return feature_data, features, classes_data, classes, n_samples, actual_classes

# 'Sensorless_drive_diagnosis.txt'
# 'data_banknote_authentication.txt'
dataset = read('data_banknote_authentication.txt')
dataset = dataset.to_numpy()
feature_data, features, classes_data, classes, n_samples, actual_classes = parse_data(dataset)
decision_tree = DecisionTree(feature_data, features, classes_data, classes)


