import pandas as pd
import numpy as np
import skfuzzy as fuzzy
def read() :
    try :
        return pd.read_csv('input.csv')
    except Exception:
        print("Error Opening File.")
        return pd.DataFrame()

df = read()
if (df.empty):
    print("Empty Dataset.")
    exit()
#getting the crisp inputs from the .csv file
x1 = df['x1 (°C/sec)'].values.tolist()
x2 = df['x2 (°C)'].values.tolist()

#upper bounds (ub) and lower bounds (lb) for x1 and x2
#given bounds
#x1 = -10 : 10
#x2 = -30 : 30
ub1 = 10 #max(x1)
lb1 = -10 #min(x1)
ub2 = 30 #max(x2)
lb2 = -30 #min(x2)

temp = np.arange(-10, 10, .1)
rate = np.arange(-30, 30, .1)

temp1 = fuzzy.trapmf(temp, [-10, -10, -6.6, -3.3])
temp2 = fuzzy.trimf(temp, [-6.6, -3.3, 0])
temp3 = fuzzy.trimf(temp, [-3.3, 0, 1.5])
temp4 = fuzzy.trimf(temp, [0, 1.5, 3.3])
temp5 = fuzzy.trimf(temp, [1.5, 3.3, 4.5])
temp6 = fuzzy.trimf(temp, [3.3, 4.5, 7.5])
temp7 = fuzzy.trapmf(temp, [4.5, 7.5, 10, 10])

rate1 = fuzzy.trapmf(rate, [-30, -30, -19.8, -9.9])
rate2 = fuzzy.trimf(rate, [-19.8, -9.9, 0])
rate3 = fuzzy.trimf(rate, [-9.9, 0, 4.5])
rate4 = fuzzy.trimf(rate, [0, 4.5, 9.9])
rate5 = fuzzy.trimf(rate, [4.5, 9.9, 13.5])
rate6 = fuzzy.trimf(rate, [9.9, 13.5, 22.5])
rate7 = fuzzy.trapmf(rate, [13.5, 22.5, 30, 30])

r = fuzzy.interp_membership(temp, temp1, -7)
print(temp3)
#x1 = normalize(x1, lb1, ub1)
#x2 = normalize(x2, lb2, ub2)

#using a total of 7 fuzzy sets for input
#input fuzzy dictionary.


print(x1)
print(x2)