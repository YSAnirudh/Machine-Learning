import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from time import time

def read(filename:str):
    try :
        data = pd.read_csv(filename, header = None)
        if (data and type(data[0][0]) == str):
            try:
                return pd.read_csv(filename, header = None, sep=' ')
            except:
                print("Error Opening File or File is Empty.")
                exit()
    except Exception:
        print("Error Opening File or File is Empty.")
        exit()
    return data

t1 = time()
#data = read('data_banknote_authentication.txt')
#data = read('Sensorless_drive_diagnosis.txt')
data = pd.read_csv('Sensorless_drive_diagnosis.txt',  header = None, sep=' ')
#print(data)
col = [i for i in range(48)]

X = data[col]
y = data[48]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
clf = DecisionTreeClassifier(criterion='entropy')

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
t2 = time()
e = t2 - t1
print(e)