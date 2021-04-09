import pandas as pd
import numpy as np

data = pd.read_csv("input.csv")

rate_temp = data["x1 (°C/sec)"].values.tolist()
temp = data["x2 (°C)"].values.tolist()

def normalise(u,l,arr):
    j = 0
    n_arr =[]
    for i in arr:
        x = (2*i-(u+l))/(u-l)
        n_arr.append(x)
        j += 1
    return n_arr    

n_rate_temp = normalise(10,-10,rate_temp)
n_temp = normalise(30,-30,temp)
#print(n_temp)

#print(n_rate_temp)
def triangle(x,a,b,c):
    if x<a:
        return 0
    elif a<=x<=b:
        return (x-a)/(b-a)
    elif b<=x<=c:
        return (c-x)/(c-b)
    elif c<x:
        return 0  

def trap(x,a,b,c,d):
    if x<a:
        return 0
    elif a<=x<=b:
        return (x-a)/(b-a)
    elif b<=x<=c:
        return 1
    elif c<=x<=d:
        return (d-x)/(d-c)
    elif d<x:
        return 0

def DOB(i):
    #for i in arr:
    a1 = [0,0,0,0,0,0,0]
    if i>=-1 and i<=-0.33:
        a = trap(i,-1,-1,-0.66,-0.33)
        a1[0] = a 
    if i>=-0.66 and i<=0:
        a = triangle(i,-0.66,-0.33,0)
        a1[1] = a
    if i>=-0.33 and i<=0.15:
        a = triangle(i,-0.33,0,0.15)
        a1[2] = a    
    if i>=-0 and i<=0.33:
        a = triangle(i,0,0.15,0.33)
        a1[3] = a
    if i>=0.15 and i<=0.45:
        a = triangle(i,0.15,0.33,0.45)
        a1[4] = a  
    if i>=0.33 and i<=0.75:
        a = triangle(i,0.33,0.45,0.75)
        a1[5] = a      
    if i>=0.45 and i<=1:
        a = trap(i,0.45,0.75,1,1)
        a1[6] = a    
    return a1

def fuzzyLogic(inp1, inp2):
    w = [[0]*7 for _ in range(7)]
    for i in range(7):
        for j in range(7):
            x = inp1[i]*inp2[j]
            w[i][j] = x

    center = [0]*9
    for i in range(9):
        if i == 0:
            center[i] = -0.75
        elif i == 8:
            center[i] = 0.95
        elif i == 1:
            center[i] = (-0.5 + 0 +0.2)/3
        elif i == 2:
            center[i] = (0 + 0.2 +0.4)/3
        elif i == 3:
            center[i] = (0.2 + 0.4 +0.5)/3
        elif i == 4:
            center[i] = (0.4 + 0.5 +0.6)/3
        elif i == 5:
            center[i] = (0.5 + 0.6 +0.7)/3
        elif i == 6:
            center[i] = (0.6 + 0.7 +0.8)/3
        elif i == 7:
            center[i] = (0.7 + 0.8 +0.9)/3 

    fam = [[1,1,1,1,1,1,1],[1,1,1,1,1,1,2],[1,1,1,1,1,2,3],[1,1,1,1,2,3,5],[1,1,1,2,2,4,6],[1,1,1,3,4,6,8],[1,2,3,5,6,8,9]]
    sum = 0.0
    sum_w = 0.0
    for i in range(7):
        for j in range(7):
            sum = sum + w[i][j]*center[fam[i][j]-1]
            sum_w = sum_w + w[i][j]
    return sum/sum_w

def denormalize(out):
    for i in range(len(out)):
        out[i] = (out[i] + 1)*50
    return out

def fuzzOutput(inp1, inp2):
    output = [0]*len(inp1)
    for i in range(len(inp1)):
        output[i] = fuzzyLogic(DOB(inp1[i]), DOB(inp2[i]))
    denormalize(output)
    return output

output = fuzzOutput(n_rate_temp, n_temp)

data["Breakoutability"] = output

data.to_csv("output.csv",index = False)