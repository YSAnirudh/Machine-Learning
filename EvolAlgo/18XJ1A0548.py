import numpy as np
from matplotlib import pyplot

#objective funtion for egg holder
def eggHolder(x,y):
    t1 = -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + y + 47)))
    t2 = - x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return  t1 + t2

#objective funtion for holder table
def holderTable(x,y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1-(np.sqrt(x**2 + y**2) / np.pi))))

#sets values specified in a 2d list
# [[xMin, xMax], [yMin, yMax]] 
def setRange(xMin, xMax, yMin, yMax):
    return [[xMin, xMax], [yMin, yMax]]

# returns list with the constant F for all the generations
# [f1, f2, .....] 
# this ensures the F is random across all the generations and the same within a generation
def factorF(F, genSize):
    for i in range(genSize):
        F[i] = np.random.rand() * 4 - 2
        while (F[i] < -2.0 or F[i] > 2.0):
            F[i] = np.random.rand() * 4 - 2
    return F

# The initialization of random start population
# This return a random list of [x,y] with the size of population 
def initPopulation(xy_ranges, populSize):
    thePopulation = [[0 for i in range(2)] for j in range(populSize)]
    satisfied = False
    count = 0
    while(not satisfied):
        count = 0
        for i in range(len(thePopulation)):
            thePopulation[i][0] = np.random.rand() * (xy_ranges[0][1] - xy_ranges[0][0]) + xy_ranges[0][0]
            thePopulation[i][1] = np.random.rand() * (xy_ranges[1][1] - xy_ranges[1][0]) + xy_ranges[1][0]
        for i in range(len(thePopulation)):
            if (thePopulation[i][0] >= xy_ranges[0][0] and thePopulation[i][0] <= xy_ranges[0][1]):
                count = count + 1
            if (thePopulation[i][1] >= xy_ranges[1][0] and thePopulation[i][1] <= xy_ranges[1][1]):
                count = count + 1
        if count == 2 * populSize:
            satisfied = True
    return thePopulation

# performs mutation on XY, given the total population and the constants K and F
def mutate(XY, popul, K, F):
    r1 = popul[np.random.randint(len(popul))]
    while(r1 == XY):
        r1 = popul[np.random.randint(len(popul))]
    r2 = popul[np.random.randint(len(popul))]
    while(r2 == r1 or r2 == XY):
        r2 = popul[np.random.randint(len(popul))]
    r3 = popul[np.random.randint(len(popul))]
    while(r3 == r1 or r3 == XY or r3 == r2):
        r3 = popul[np.random.randint(len(popul))]
    res = [XY[i] + K * (r1[i] - XY[i]) + F * (r2[i] - r3[i]) for i in range(len(XY))]
    return res

# performs crossover given mutant vector, parent vector and crossover probability
def crossOver(mutant, parent, COProb):
    rand1 = np.random.rand()
    rand2 = np.random.rand()
    res = [0.0,0.0]
    if (rand1 <= COProb):
        res[0] = mutant[0]
    else:
        res[0] = parent[0]
    if (rand2 <= COProb):
        res[1] = mutant[1]
    else:
        res[1] = parent[1]
    return res

# performs the mutation and crossover given the ranges of values, 
# parent vector, population, and the constants with the current gen(i)
def computeTrialVector(xy_ranges, parent, population, K, F, i, crossOverProb):
    mutVec = mutate(parent, population, K, F[i])
    triVec = crossOver(mutVec, parent, crossOverProb)
    while (triVec[0] < xy_ranges[0][0] or triVec[0] > xy_ranges[0][1] or triVec[1] < xy_ranges[1][0] or triVec[1] > xy_ranges[1][1]):
        mutVec = mutate(parent, population, K, F[i])
        triVec = crossOver(mutVec, parent, crossOverProb)
    return triVec

#finds the vector with best fitness in the population and its value, given population and objective function
def findBestFit(population, objFunc):
    bestFitIndex = 0
    bestFit = 1.0e+307
    for i in range(len(population)):
        loss = objFunc(population[i][0], population[i][1])
        if (loss < bestFit):
            bestFit = loss
            bestFitIndex = i
    return bestFit, population[bestFitIndex]

# finds the average fitness of all the vectors in the population
def findAvgFit(population, objFunc):
    sum = 0.0
    for i in range(len(population)):
        sum = sum + objFunc(population[i][0], population[i][1])
    return sum / len(population)

# performs the differential evolution taking in all the parameters needed and uses the above mentioned functions
def differential_evolution(xy_ranges, objective, populationSize, genSize, K, F, crossOverProb):
    population = initPopulation(xy_ranges, populationSize)
    
    bestFitGen = [0.0 for j in range(genSize)]
    avgFitGen = [0.0 for j in range(genSize)]
    for i in range(genSize): # going over every generation
        for j in range(populationSize): # going over every vector in the population
            parent = population[j] # [x,y]
            triVec = computeTrialVector(xy_ranges, parent, population, K, F, i, crossOverProb)
            triVecLoss = objective(triVec[0], triVec[1])
            parentLoss = objective(parent[0], parent[1])
            if (triVecLoss < parentLoss): # if better fitness replace
                population[j] = triVec
        # avg and best fitness values for each generation for plots
        bestFitGen[i], notUsed = findBestFit(population, objective)
        avgFitGen[i] = findAvgFit(population, objective)
    # best fitness vector and its fitness value
    bestFit, fitVec = findBestFit(population, objective)
    return bestFitGen, avgFitGen, bestFit, fitVec

#function for platting the graph and making an image from it
def plotGraph(genBestFit, genAvgFit, populationSize, genSize, best, option):
    pyplot.cla()
    pyplot.plot(genBestFit, color='green', linestyle='-', label="Best Fitness through Gens")
    pyplot.plot(genAvgFit, color='blue', linestyle='-', label="Average Fitness through Gens")
    pyplot.title("Population Size: " + str(populationSize) + " | No Of Generations : " + str(genSize) + " | Global Minimum: " + str(genBestFit[len(genBestFit)-1]) + "\n Point at Minimum =" + str(best[0]) + "," + str(best[1]),fontdict={'fontsize' : 8})
    pyplot.legend()
    pyplot.locator_params(axis='y', nbins=10)
    pyplot.xlabel("Generation")
    pyplot.ylabel("Fitness")
    # check whether it is eggholder or holder table for the naming convention and saves the image
    if option == True:
        pyplot.savefig("EggHolder_Population_"+str(populationSize)+"_Generations_"+str(genSize)+".png", dpi=500)   
    else:
        pyplot.savefig("HolderTable_Population_"+str(populationSize)+"_Generations_"+str(genSize)+".png", dpi=500)

#function to get all 8 plots by performing differential evolution
def plotsForBothFuncs():
    #initialization of parameters
    populationSize = [20,50,100,200]
    genSize = 200
    K = 0.5
    F = [0.0]*genSize
    crossOverProb = 0.8

    #Egg holder function
    print("Egg Holder Function")
    # parameters for Egg Holder Function
    ranges = setRange(-512.0, 512.0, -512.0, 512.0)
    # for all population sizes
    for i in range(len(populationSize)):
        # F inside for loop to get different set of F values for each population size
        F = factorF(F,genSize)
        # perform diff evolution
        bestFit, avgFit, best, bestVec = differential_evolution(ranges, eggHolder, populationSize[i], genSize, K, F, crossOverProb)
        # plot graph
        plotGraph(bestFit, avgFit, populationSize[i], genSize, bestVec, True)

        #prints the required values
        print("Population Size :", populationSize[i], ", Generation Size :", genSize)
        print("Global Minimum =", best, ", Best value at Point, x =", bestVec[0], ",y =", bestVec[1])
    #Holder table function
    print("Holder Table Function")
    # parameters for Egg Holder Function
    ranges = setRange(-10.0, 10.0, -10.0, 10.0)
    # for all population sizes
    for i in range(len(populationSize)):
        # F inside for loop to get different set of F values for each population size
        F = factorF(F,genSize)
        # perform diff evolution
        bestFit, avgFit, best, bestVec = differential_evolution(ranges, holderTable, populationSize[i], genSize, K, F, crossOverProb)
        # plot graph
        plotGraph(bestFit, avgFit, populationSize[i], genSize,bestVec, False)
        #prints the required values
        print("Population Size :", populationSize[i], ", Generation Size :", genSize)
        print("Global Minimum =", best, ", Best value at Point, x =", bestVec[0], ",y =", bestVec[1])

plotsForBothFuncs()