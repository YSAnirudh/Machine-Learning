import pandas as pd
import numpy as np
STEP_SIZE = 100

#reads inputs from input.csv csv file
def read() :
    try :
        return pd.read_csv('input.csv')
    except Exception:
        print("Error Opening File.")
        return pd.DataFrame()

def write(dataFrame):
    dataFrame.to_csv("output.csv", index = False)

#NEEDS CHECK
#makes the dataframe into input array values
def retInputArrays(dataframe):
    inputArrs = []
    #getting the crisp inputs from the .csv file
    x1 = dataframe['x1 (°C/sec)'].values.tolist()
    x2 = dataframe['x2 (°C)'].values.tolist()
    #upper bounds (ub) and lower bounds (lb) for x1 and x2
    #given bounds
    #x1 = -10 : 10
    #x2 = -30 : 30
    ub1 = 10 #max(x1)
    lb1 = -10 #min(x1)
    ub2 = 30 #max(x2)
    lb2 = -30 #min(x2)

    x1 = normalize(x1, lb1, ub1)
    x2 = normalize(x2, lb2, ub2)

    inputArrs.append(x1)
    inputArrs.append(x2)
    return inputArrs

#normalizes the given values according to given lower and upper bound
def normalize(crispInp, lb, ub):
    for i in range (len(crispInp)):
        crispInp[i] = (2*crispInp[i] - (ub + lb))/(ub-lb)
    return crispInp

#gets the membership value for a given triangle
def triVal(x, a, b, c):
    if x < a:
        return 0.0
    elif x >= a and x < b:
        return (x-a)/(b-a)
    elif x >= b and x <= c:
        return (c-x)/(c-b)
    else :
        return 0.0

#gets the membership value for a given trapezoid
def trapVal(x, a, b, c, d):
    if x < a:
        return 0.0
    elif x >= b and x <= c:
        return 1.0
    elif x >= a and x <= b :
        return (x-a)/(b-a)
    elif x >= c and x <= d :
        return (d-x)/(d-c)
    else :
        return 0.0

#gets the degree of belonging of xVal to each fuzzy set given in xArr
#=>returns an array of degree of belongin to each of the fuzzy sets in xArr
def getDegOfBel(xVal):
    x = []
    x.append(trapVal(xVal, -1,-1,-0.66,-0.33))
    x.append(triVal(xVal,-0.66, -0.33, 0))
    x.append(triVal(xVal,-0.33, 0, 0.15))
    x.append(triVal(xVal,0, 0.15, 0.33))
    x.append(triVal(xVal,0.15, 0.33, 0.45))
    x.append(triVal(xVal,0.33, 0.45, 0.75))
    x.append(trapVal(xVal, 0.45, 0.75 , 1,1))
    return x

#returns 2D list for FAM
#7x7 2D List as we have 7 fuzzy sets for each input
def getFAM():
    FAM = [[0]*7 for _ in range(7)]
    for i in range(len(FAM)):
        for j in range(len(FAM[i])):
            if j - i <= len(FAM):
                FAM[i][j] = 1
            if i + j == len(FAM):
                FAM[i][j] = 2
            if i + j == len(FAM) + 1:
                FAM[i][j] = 3
            if i + j == len(FAM) + 2:
                FAM[i][j] = 4
            if i + j == len(FAM) + 2 and (i+4 == len(FAM) or j+4 == len(FAM)):
                FAM[i][j] = 5
            if i + j == len(FAM) + 3:
                FAM[i][j] = 6
            if i + j == len(FAM) + 4:
                FAM[i][j] = 8
    FAM[2][5] = 1
    FAM[4][4] = 2
    FAM[6][6] = 9
    return FAM

#calculates the 2D list of weights given the degree of belonging arr for both inputs
def getWeightsMat(rateDegOfBel, tempDegOfBel):
    weights = [[0]*7 for _ in range(7)]
    for i in range(len(rateDegOfBel)):
        for j in range(len(tempDegOfBel)):
            weights[i][j] = rateDegOfBel[i] * tempDegOfBel[j]
    return weights

#NEEDS CHECK
#gets centroid for a triangle
def centroid(arr):
    if len(arr) == 3:
        return (arr[0] + arr[1] + arr[2]) / 3
    return 0.0

#defuzzifies when given the FAM, weights and the centroids of the output sets
def defuzzify(FAM, weights, centroid):
    mat = [[0]*7 for _ in range(7)]
    defuzzOut = 0.0
    sumWeights = 0.0
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] = weights[i][j] * centroid[FAM[i][j]]
            defuzzOut += mat[i][j]
            sumWeights += weights[i][j]
    return defuzzOut/sumWeights

#for denormalizing an output from -1 to 1 -> 0 to 100
def denormalize(val, ub, lb):
    val = val * (ub - lb)
    val = val + (ub - lb)
    val = val / 2
    return val

#gets the index value from the given input to easily find degree of belonging from the array.
#for -0.1 returns 90 which is -0.1 from a 201 sized array of step size .01
def valToIndex(val):
    return (int)(val * STEP_SIZE) + STEP_SIZE

#given the inputs applies the fuzzy logic to get the output
def applyLogic(inputRate, inputTemp):
    degBelRate = getDegOfBel(inputRate)
    degBelTemp = getDegOfBel(inputTemp)
    weights = getWeightsMat(degBelRate, degBelTemp)
    FAM = getFAM()

    return denormalize(defuzzify(FAM, weights, centroids), 100, 0)
    
#given the arrays of both the inputs, returns array of output values
#between 0 and 100
def fuzzy(x1, x2):
    if len(x1) != len(x2):
        return None
    output = []
    for i in range(len(x1)):
        output.append(applyLogic(x1[i], x2[i]))
    return output

df = read()
if (df.empty):
    print("Empty Dataset.")
    exit()

inputs = retInputArrays(df)

#centroids of the 9 output fuzzy sets
centroids = {}
centroids[1] = -0.75
centroids[2] = centroid([-0.5, 0.0, 0.2]) 
centroids[3] = centroid([0, 0.2, 0.4])
centroids[4] = centroid([0.2, 0.4, 0.5])
centroids[5] = centroid([0.4, 0.5, 0.6])
centroids[6] = centroid([0.5, 0.6, 0.7])
centroids[7] = centroid([0.6, 0.7, 0.8])
centroids[8] = centroid([0.7, 0.8, 0.9])
centroids[9] = 0.95

for i in range(7):
    print(getFAM()[i])

output = fuzzy(inputs[0], inputs[1])
df['Breakoutability'] = output
write(pd.DataFrame(df))