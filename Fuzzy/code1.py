####Implementation of fuzzy logic

#####Finding the range of the inputs 
import pandas as pd
import numpy as np

#####There are the fuzzy sets which we have used for this program
vector = [[-1,-0.66,-0.33],[-0.66,-0.33,0],[-0.33,0,0.15],[0,0.15,0.33],[0.15,0.33,0.45],[0.33,0.45,0.75],[0.45,0.75,1]]

FAM = [[-0.75,-0.75,-0.75,-0.75,-0.75,-0.75,-0.75],
    [-0.75,-0.75,-0.75,-0.75,-0.75,-0.75,(-0.5 + 0.2)/3],
    [-0.75,-0.75,-0.75,-0.75,-0.75,-0.75,0.2],
    [-0.75,-0.75,-0.75,-0.75,(-0.5 + 0.2)/3,0.2,0.5],
    [-0.75,-0.75,-0.75,(-0.5 + 0.2)/3,(-0.5 + 0.2)/3,(0.2 + 0.4 + 0.5)/3,0.6],
    [-0.75,-0.75,(-0.5 + 0.2)/3,0.2,(0.2 + 0.4 + 0.5)/3,0.6,0.8],
    [-0.75,(-0.5 + 0.2)/3,0.2,0.5,0.6,0.8,0.95]]

#####Finding the belonging of the normalized values

def fuzzification(x):
    vec = []
    belonging = []
    counter = 0 ##This is to find if the value is in the set
    count1 = 0 ##This is to find the value of the set
    for j in x:
        vec.append([])
        for i in vector:
            if(j> i[0] and j<i[2]):
                vec[counter].append(1)
            else:
                vec[counter].append(0)
        counter = counter + 1
        
    for k in x:
        belonging.append([])
        for i in vector:
            if(k<i[0]):
                belonging[count1].append(0)
            elif(k >= i[0] and k<= i[1]):
                if(k>0.75 and k<1 or k>-1 and k<-0.66):
                    belonging[count1].append(1.0)
                else:
                    belonging[count1].append((k-i[0])/(i[1]-i[0]))
            elif(k>= i[1] and k<= i[2]):
                if(k>0.75 and k<1 or k>-1 and k<-0.66):
                    belonging[count1].append(1.0)
                else:
                    belonging[count1].append((i[2]-k)/(i[2]-i[1]))
            else:
                belonging[count1].append(0)
        count1 = count1 + 1
    for i in range(len(belonging)):
        print(belonging[i])
    print()
    return vec,belonging

def Output(x1,x2):
    finalout = []
    for k in range(len(x1)):
        output = 0
        for i in range(7):
            if(x1[k][i] != 0):
                for j in range(7):
                    if(x2[k][j] != 0):
                        print(x1[k][i]*x2[k][j])
                        output = output + x1[k][i]*x2[k][j]*FAM[i][j]
                        #output1 = output1 + x1[k][i]*x2[k][j]
        finalout.append(output)
    return finalout


def finalvalues(out,u,l):
    finarr = []
    for i in out:
        finarr.append(((i*(u-l))+(u-l))/2)
    return finarr


###Finding the normalized inputs
def normalize(x,mi,ma):
    X = []
    for i in x:
        X.append((2*i-(ma+mi))/(ma-mi))
    return X
        


data = pd.read_csv('input.csv')## The input data given to us

x1 = data['x1 (°C/sec)'].values.tolist()
x2 = data['x2 (°C)'].values.tolist()


x1min,x1max = (-10,10)## Finding the min and max values of the input 1
x2min,x2max = (-30,30)## Finding the min and max values of the input 2


X1 = normalize(x1,x1min,x1max) ## Normalizing the input 1
X2 = normalize(x2,x2min,x2max) ## Normalizing the input 2


X1vec,X1belonging = fuzzification(X1) ## Finding the belonging of the values in the normalized input1
X2vec,X2belonging = fuzzification(X2) ## Finding the belonging of the values in the normalized input2


Outarr = Output(X1belonging,X2belonging) ## This is the output values without the crisp values



answer = finalvalues(Outarr,100,0)
finalres = pd.DataFrame(answer) ## Converting the list to a dataframe

finalres.to_csv('18XJ1A0511.csv',index = False,header = False) ## Final output file