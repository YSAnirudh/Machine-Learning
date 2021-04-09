import pandas as pd
import numpy as np

def read(filename:str):
    try :
        data = pd.read_csv(filename, header = None, sep=',| ', engine = 'python')
    except Exception:
        print("Error Opening File or File is Empty.")
        exit()
    return data

def calculate_entropy(df_label):
    classes,class_counts = np.unique(df_label,return_counts = True)
    entropy_value = np.sum([(-class_counts[i]/np.sum(class_counts))*np.log2(class_counts[i]/np.sum(class_counts)) 
                        for i in range(len(classes))])
    return entropy_value

def calculate_information_gain(dataset,feature,label): 
    # Calculate the dataset entropy
    dataset_entropy = calculate_entropy(dataset[label])   
    values,feat_counts= np.unique(dataset[feature],return_counts=True)
    
    # Calculate the weighted feature entropy                                # Call the calculate_entropy function
    weighted_feature_entropy = np.sum([(feat_counts[i]/np.sum(feat_counts))*calculate_entropy(dataset.where(dataset[feature]
                              ==values[i]).dropna()[label]) for i in range(len(values))])    
    feature_info_gain = dataset_entropy - weighted_feature_entropy
    return feature_info_gain

def create_decision_tree(dataset, df, features, label, parent):
    datum = np.unique(df[label], return_counts = True)
    unique_data = np.unique(dataset[label])

    if (len(unique_data) <= 1):
        return unique_data[0]

    elif (len(dataset) == 0):
        return unique_data[np.argmax(datum[1])]

    elif (len(features) == 0):
        return parent

    else:
        parent = unique_data[np.argmax(datum[1])]

        item_values = [calculate_information_gain(dataset, feature, label) for feature in features]

        for value in np.unique(dataset[optimum_feature]):
            min_data = dataset.where(dataset[optimum_feature] == value).dropna()
            min_tree = create_decision_tree(min_data, df, features, label, parent)
            decision_tree[optimum_feature][value] = min_tree 
        return (decision_tree)

df = read('data_banknote_authentication.txt')
features = df.columns[:-1]
label = df.columns[-1]
parent=None
print(features, label)
decision_tree = create_decision_tree(df,df,features,label,parent)
