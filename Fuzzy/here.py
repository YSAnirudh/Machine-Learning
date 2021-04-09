import numpy as np
import pandas as pd

def calcDOBTriangle(a,b,c,x):
    if x<a:
        return 0
    elif a<=x<=b:
        return (x-a)/(b-a)
    elif b<=x<=c:
        return (c-x)/(c-b)
    elif c<x:
        return 0
    
def calcDOBTrapez(a,b,c,d,x):
    if x<a:
        return 0
    elif b<=x<=c:
        return 1
    elif a<=x<=b:
        return (x-a)/(b-a)
    
    elif c<=x<=d:
        return (d-x)/(d-c)
    elif d<x:
        return 0
    
def normalize(x,u,l):
    Nvalues=[]
    i=0
    for x_value in x:
        Nvalues.append((2*x_value-(u+l))/(u-(l)))
        i+=1
    return Nvalues

def calcAllDOB(x):
    DOB=[]
    DOB.append(calcDOBTrapez(-1,-1,-0.66,-0.33,x))
    DOB.append(calcDOBTriangle(-0.66,-0.33,0,x))
    DOB.append(calcDOBTriangle(-0.33,0,0.15,x))
    DOB.append(calcDOBTriangle(0,0.15,0.33,x))
    DOB.append(calcDOBTriangle(0.15,0.33,0.45,x))
    DOB.append(calcDOBTriangle(0.33,0.45,0.75,x))
    DOB.append(calcDOBTrapez(0.45,0.75,1,1,x))
    return DOB

def getCentroid(x1,x2,x3):
    return (x1+x2+x3)/3

centroids=[]
centroids.append(-0.75)
centroids.append(getCentroid(-0.5,0,0.2))
centroids.append(getCentroid(0,0.2,0.4))
centroids.append(getCentroid(0.2,0.4,0.5))
centroids.append(getCentroid(0.4,0.5,0.6))
centroids.append(getCentroid(0.5,0.6,0.7))
centroids.append(getCentroid(0.6,0.7,0.8))
centroids.append(getCentroid(0.7,0.8,0.9))
centroids.append(0.95)

def sumOfWeights(wtMat,FAM):
    wtSum=0.0
    for i in range(7):
        for j in range(7):
            wtSum+=(wtMat[i][j]*centroids[FAM[i][j]-1])
            
    return wtSum

def getWtMat(x,y):
    wtMat = [[0 for i in range(len(x))] for j in range(len(y))] 
    for i in range(len(x)):
        for j in range(len(y)):
            wtMat[i][j]=x[i]*y[j]
            
    return wtMat    

def denormalize(output):
    for i in range(len(output)):
        output[i] = (output[i] + 1) * 50
    return output

def execFuzzy(inp1, inp2):
    tempdob = calcAllDOB(inp1)
    tempdeldob = calcAllDOB(inp2)
    wt = getWtMat(tempdeldob, tempdob)
    FAM = [[1,2,3,5,6,8,9],[1,1,2,3,4,6,8],[1,1,1,2,2,4,6],[1,1,1,1,2,3,5],[1,1,1,1,1,1,3],[1,1,1,1,1,1,2],[1,1,1,1,1,1,1]]
    FAM.reverse()
    sumCent = sumOfWeights(wt, FAM)
    sumWt = 0.0
    for i in range(7):
        for j in range(7):
            sumWt = sumWt + wt[i][j]
    return sumCent/sumWt

def fuzzy(ntemp, ndeltemp):
    output = [0]*len(ntemp)
    for i in range(len(ntemp)):
        output[i] = execFuzzy(ntemp[i], ndeltemp[i])
    output = denormalize(output)
    return output

initList=np.arange(-1,1.01,0.01)
data = pd.read_csv('input.csv')
delTemp = data['x1 (°C/sec)'].values.tolist()
temp = data['x2 (°C)'].values.tolist()

normalizedTemp=normalize(temp,30,-30)
normalizedDelTemp=normalize(delTemp,10,-10)

output = fuzzy(normalizedTemp, normalizedDelTemp)
data["Breakoutability"] = output

data.to_csv("output.csv", index=False)