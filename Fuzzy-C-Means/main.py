import pandas as pd
import numpy as np
import numpy.random as npRand
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Reads the data from the 'filename' excel file, 'sheet_name' sheet
# Also takes in the number of rows to skip if there are any non data points at the top 
def read_data(filename, sheet_name, rows_to_skip):
    df = pd.read_excel(filename, sheet_name, skiprows=rows_to_skip, header=None)
    if df is not None:
        return df.values.tolist()
    else:
        return pd.DataFrame().values.tolist()

# Given the dataset and the NUMBER of training examples
# Returns the training set and writes the data points into the classification_data_18548.txt
# Does not return test set as it is not needed in this training program
def train_test_split(dataset, no_of_training_examples):
    train = [[0.0, 0.0] for i in range(no_of_training_examples)]
    train = np.asarray(train)
    # Shuffle for interspersed approach of splitting
    npRand.shuffle(dataset)
    for i in range(no_of_training_examples):
        ran = npRand.randint(0, dataset.shape[0]-1)
        train[i,0:2] = dataset[ran,0:2]
        dataset = np.delete(dataset, ran, 0)
    i = 0
    f = open("classification_data_18548.txt", 'w')
    while len(dataset) > 1:
        ran = npRand.randint(0, dataset.shape[0]-1)
        f.write(str(dataset[ran,0])+','+str(dataset[ran,1])+'\n')
        i = i + 1
        dataset = np.delete(dataset, ran, 0)
    f.write(str(dataset[0,0])+','+str(dataset[0,1])+'\n')
    print("Training Samples:", len(train), ",Testing Samples:", i+1)
    f.close()
    return train

# Class FuzzyCMeans to handle the Algorithm
class FuzzyCMeans():
    # Constructor to initialize everything required for the algorithm
    def __init__(self, dataset, min_clusters, max_clusters):
        self.A = np.zeros((dataset.shape[0], dataset.shape[0])) # norm inducing matrix
        self.init_a_norm_inducing_matrix()
        self.dataset = dataset # dataset
        self.N = dataset.shape[0] # length of dataset
        self.M = 2
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        # To store clustered results
        self.clustered_mult_clusters = np.zeros((self.max_clusters - (min_clusters), self.N))
        # To store objective function (J) Value for each cluster
        self.objective_mult_clusters = np.zeros(self.max_clusters - (min_clusters))
        # To store the number of iteration taken per cluster
        self.iterations_mult_clusters = np.zeros(self.max_clusters - (min_clusters))
        # To store the best cluster means
        self.best_cluster_means = None
        # Epsilon value
        self.threshold = 0.001
        # The cluster size for best split
        self.best_cluster = -1

    # init Mpc not Mfc
    def init_u_part_matrix(self, no_of_clusters):
        return npRand.random((no_of_clusters, self.N))

    # Checking if U is valid (Mpc)
    def is_valid_u_part_matrix(self, U):
        if (U is None):
            return False
        sum_mew = 0
        count_i = 0
        # print(U)
        for i in range(U.shape[0]):
            count = 0
            for j in range(U.shape[1]):
                if (U[i, j] < 0 or U[i, j] > 1):
                    return False
                if (U[i, j] != 0):
                    count = count + 1
            if (count == self.N):
                break
            else:
                count_i = count_i + 1

        if (count_i == U.shape[0]):
            return False

        for i in range(U.shape[0]):
            sum_mew = 0.0
            for j in range(U.shape[1]):
                sum_mew = sum_mew + U[i, j]
                # print(U[i,j])
        if (sum_mew >= self.N or sum_mew <= 0):
            return False
        return True

    # Initializing A to Identity Matrix
    def init_a_norm_inducing_matrix(self):
        for i in range(self.A.shape[0]):
            self.A[i][i] = 1

    # Calculating the cluster mean for cluster i given U 
    def calc_cluster_mean(self, U, i):
        sum_mew_zK = np.zeros(2)
        sum_mew = 0.0
        for k in range(U.shape[1]):
            power = math.pow(U[i, k], self.M)
            mewZk = power * self.dataset[k]
            sum_mew_zK = sum_mew_zK + mewZk
            sum_mew = sum_mew + power
        return sum_mew_zK / sum_mew

    # calculating distance information given cluster means
    def DikA_squared(self, cluster_means):
        dikA = np.zeros((cluster_means.shape[0], self.dataset.shape[0]))

        for i in range(cluster_means.shape[0]):
            tempzK_vI = self.dataset - cluster_means[i]
            # To get the distance it is x**2 + y**2
            # This can be obtained only when we transpose the matrix back before
            # performing element wise multiplication 
            tempzK_vI = tempzK_vI.transpose().dot(self.A).transpose() * tempzK_vI
            #print(temp[:,0])#, temp[:,1])
            dikA[i] = np.add(tempzK_vI[:,0], tempzK_vI[:,1])
        return dikA

    # Calculating the objective function value given U and distances
    def objective_function_Jc(self, U, dikA):
        return np.sum(U * dikA);

    # clusttering algorithm given the number of clusters
    def c_means_clustering(self, no_of_clusters):
        clus_num = no_of_clusters - self.min_clusters
        cluster_means = np.zeros((no_of_clusters, 2))
        dikA = np.zeros((cluster_means.shape[0], self.dataset.shape[0]))
        U = None
        U_temp = None
        iter_no = 0
        # Initializing a valid U Matrix
        while (not self.is_valid_u_part_matrix(U)):
            U = self.init_u_part_matrix(no_of_clusters)
        while (True):
            # get cluster means for each cluster
            for j in range(no_of_clusters):
                cluster_means[j] = self.calc_cluster_mean(U, j)
            # calculate distances
            dikA = self.DikA_squared(cluster_means)
            count_k = 0
            # deepcopy to not copy U by reference as it would change both of the matrices when one is changed
            U_temp = copy.deepcopy(U)
            # change U according to the distances
            for k in range(dikA.shape[1]):
                count_i = 0
                for i in range(dikA.shape[0]):
                    if (dikA[i][k] < 0.1 and dikA[i][k] >= 0):#HOT
                        for sub_i in range(dikA.shape[0]):
                            U_temp[sub_i][k] = 0.0
                        U_temp[i][k] = 1.0
                        #print(U[:,k])
                        break;
                    else:
                        count_i = count_i + 1
                if (count_i == dikA.shape[0]):
                    for i in range(dikA.shape[0]):
                        sum_dika_djka = 0.0
                        for j in range(dikA.shape[0]):
                            sum_dika_djka = sum_dika_djka + math.pow((dikA[i][k]/dikA[j][k]),2/(self.M-1))
                        U_temp[i][k] = 1 / sum_dika_djka
                    count_k = count_k + 1
            
            # If the error id less than epsilon (self.threshold)
            # break the loop
            if (np.amax(np.abs(U_temp - U)) < self.threshold):
                break;
            # If not update U and continue
            U = copy.deepcopy(U_temp)
            iter_no = iter_no + 1
        # initializing the required arrays
        self.iterations_mult_clusters[clus_num] = iter_no
        self.objective_mult_clusters[clus_num] = self.objective_function_Jc(U, dikA)
        self.clustered_mult_clusters[clus_num] = self.classify_into_clusters(U)

        # this gets initialized for every iteration from 2-9 clusters
        # but as we run the algorithm again after we find the best cluster
        # it will set the cluster means to the best one
        self.best_cluster_means = cluster_means

    # iterates from min_clusters(included) to max_clusters(excluded)
    def iterate_c_means_clustering(self):
        for clus_num in range(self.max_clusters - self.min_clusters):
            no_of_clusters = clus_num + self.min_clusters
            self.c_means_clustering(no_of_clusters=no_of_clusters)
        return

    # classifies the data into clusters
    # i.e. tag each data points with the respective cluster it is assigned to
    def classify_into_clusters(self, U):
        clustered = np.zeros(U.shape[1])
        for k in range(U.shape[1]):
            cluster_index = -1.0
            cluster_max = -1.0
            for i in range((U.shape[0])):
                if (U[i][k] > cluster_max):
                    cluster_max = U[i][k]
                    cluster_index = i
            clustered[k] = cluster_index
        return clustered

    # Helper function to calculate Rc
    def ratio_calc(self, clus_num):
        return self.objective_mult_clusters[clus_num] - self.objective_mult_clusters[clus_num + 1]

    # calculates the ratios Rc and decides on which is the best cluster
    def find_best_cluster(self):
        min_ratio = np.inf
        min_clus = 0
        for clus_num_plus_1 in range(self.iterations_mult_clusters.shape[0] - self.min_clusters):
            clus_num = clus_num_plus_1 + (self.min_clusters - 1)
            numer_ratio = self.ratio_calc(clus_num=clus_num)
            denom_ratio = self.ratio_calc(clus_num=clus_num-1)
            ratio = np.abs(numer_ratio/denom_ratio)
            if (ratio < min_ratio):
                min_ratio = ratio
                min_clus = clus_num
            self.best_cluster = min_clus
        return [min_ratio, min_clus + self.min_clusters]

    # writes the cluster means/centroids of the cluster to a text file
    def write_centriods_to_txt(self):
        f = open("centroids_data_18548.txt", 'w');
        for i in range(self.best_cluster_means.shape[0]):
            f.write(str(self.best_cluster_means[i,0])+','+str(self.best_cluster_means[i,1])+'\n')
        f.close()

    # plots the no of the number of clusters on the x-axis, 
    # and the value of Objective function and the number of iterations for convergence on the two y-axes
    # also commented lines writes all this information into an excel file for post preprocessing
    def plot_iter_and_obj(self, sheet_name):
        fig, ax = plt.subplots()
        ax.plot([i+self.min_clusters for i in range(self.max_clusters - self.min_clusters)], 
            self.objective_mult_clusters, 
            color='blue', marker='o')
        ax.set_xlabel("No Of Clusters", fontsize=12)
        ax.set_ylabel("Objective Function(J) Value", color='blue', fontsize=12)
        
        ax2 = ax.twinx()
        ax2.plot([i+self.min_clusters for i in range(self.max_clusters - self.min_clusters)], 
            self.iterations_mult_clusters, 
            color='orange', marker='o')
        ax2.set_ylabel("No of Iterations", color='orange', fontsize=12)

        # df = pd.DataFrame(zip([i+self.min_clusters for i in range(self.max_clusters - self.min_clusters)], 
        #     self.iterations_mult_clusters, self.objective_mult_clusters))
        # df.to_excel("Hello.xlsx", index = False)
        ax.set_title(sheet_name)
        plt.subplots_adjust(right=0.88, left=0.15)
        plt.show()
    
    # plots the clustered points for the training data
    def plot_training_clusters(self, results, min_clus):
        colors = cm.rainbow(np.linspace(0, 1, min_clus))
        plt.scatter(results[:,0], results[:,1], c=colors[results[:,2].astype(int)], s=4)
        plt.scatter(self.best_cluster_means[:,0], self.best_cluster_means[:,1], c='black', s=50)
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.title("FCM Clustering (Training Result)")
        plt.show()

    # The Main Fuzzy C Means Logic
    def main_fuzzy_c_means_algo(self, sheet_name):
        self.iterate_c_means_clustering()
        [min_ratio, min_clus] = self.find_best_cluster()
        print("Number of Clusters for min Rc:", min_clus)
        print("Ratio Rc:", min_ratio)
        self.c_means_clustering(min_clus)
        

        results = np.zeros((self.dataset.shape[0], 3))
        for i in range(self.dataset.shape[0]):
            results[i,0] = self.dataset[i,0]
            results[i,1] = self.dataset[i,1]
            results[i,2] = self.clustered_mult_clusters[min_clus-self.min_clusters][i].astype(int)

        self.plot_training_clusters(results=results, min_clus=min_clus)
        self.write_centriods_to_txt()
        self.plot_iter_and_obj(sheet_name)

### FOR CLASSIFICATION TASK BELOW ###

# Class FuzzyCMeansClassify to handle the Classification Task
class FuzzyCMeansClassify():
    def __init__(self):
        self.testing_data = None
        self.cluster_means = None
    
    # Functions to find the distances given the testing data points and the cluster means
    def DikA_squared_trained(self):
        dikA = np.zeros((self.cluster_means.shape[0], self.testing_data.shape[0]))

        A = np.zeros((self.testing_data.shape[0], self.testing_data.shape[0]))
        for i in range(A.shape[0]):
            A[i][i] = 1

        for i in range(self.cluster_means.shape[0]):
            tempzK_vI = self.testing_data[:] - self.cluster_means[i]
            tempzK_vI = tempzK_vI.transpose().dot(A).transpose() * tempzK_vI
            dikA[i] = np.add(tempzK_vI[:,0], tempzK_vI[:,1])
        return dikA

    # reads the data which is written by the training code
    # and parses it into arrays, the cluster means(centroids) and the testing data points
    # also deletes the file so that there aren't any extra files made and it is only a temporary storage
    def read_classsification_centroid_data(self):
        f = open("centroids_data_18548.txt", 'r')
        centroids = f.readlines()
        good_centroids = [centroid.split()[0].split(',') for centroid in centroids]
        f.close()
        f = open("classification_data_18548.txt", 'r')
        data_points = f.readlines()
        good_data_points = [data_point.split()[0].split(',') for data_point in data_points]
        f.close()
        for i in range(len(good_centroids)):
            for j in range(2):
                good_centroids[i][j] = float(good_centroids[i][j])
        for i in range(len(good_data_points)):
            for j in range(2):
                good_data_points[i][j] = float(good_data_points[i][j])
        self.cluster_means = np.asarray(good_centroids)
        self.testing_data = np.asarray(good_data_points)

    # finds the U matrix given distances and cluster means
    def find_U_matrix(self):
        M = 2
        dikA = self.DikA_squared_trained()
        count_k = 0
        U = np.zeros((self.cluster_means.shape[0], self.testing_data.shape[0]))
        for k in range(dikA.shape[1]):
            count_i = 0
            for i in range(dikA.shape[0]):
                if (dikA[i][k] < 0.1 and dikA[i][k] >= 0):
                    for sub_i in range(dikA.shape[0]):
                        U[sub_i][k] = 0.0
                    U[i][k] = 1.0
                    break;
                else:
                    count_i = count_i + 1
            if (count_i == dikA.shape[0]):
                for i in range(dikA.shape[0]):
                    sum_dika_djka = 0.0
                    for j in range(dikA.shape[0]):
                        sum_dika_djka = sum_dika_djka + math.pow((dikA[i][k]/dikA[j][k]),2/(M-1))
                    U[i][k] = 1 / sum_dika_djka
                count_k = count_k + 1
        return U
    
    # Classification task i.e. tagging the data points with their appropriate clusters
    def classify(self, U):
        clustered = np.zeros(U.shape[1])
        for k in range(U.shape[1]):
            cluster_index = -1.0
            cluster_max = -1.0
            for i in range((U.shape[0])):
                if (U[i][k] > cluster_max):
                    cluster_max = U[i][k]
                    cluster_index = i
            clustered[k] = cluster_index
        return clustered

    # plots the clustered points for the testing data
    def plot_testing_clusters(self, results):
        colors = cm.rainbow(np.linspace(0, 1, self.cluster_means.shape[0]))
        plt.scatter(results[:,0], results[:,1], c=colors[results[:,2].astype(int)], s=4)
        plt.scatter(self.cluster_means[:,0], self.cluster_means[:,1], c='black', s=50)
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.title("FCM Clustering (Classification Result)")
        plt.show()

### Function for Classification
def FuzzyCMeansClassification():
    fuzzyClassify = FuzzyCMeansClassify()
    fuzzyClassify.read_classsification_centroid_data()
    
    U = fuzzyClassify.find_U_matrix()
    clustered = fuzzyClassify.classify(U)

    results = np.zeros((fuzzyClassify.testing_data.shape[0], 3))
    for i in range(fuzzyClassify.testing_data.shape[0]):
        results[i,0] = fuzzyClassify.testing_data[i,0]
        results[i,1] = fuzzyClassify.testing_data[i,1]
        results[i,2] = clustered[i].astype(int)
    
    fuzzyClassify.plot_testing_clusters(results=results)
    return

### Function for Training
def FuzzyCMeansClustering():
    sheet_name = "Data Set 2"
    rows_to_skip = 0
    df = read_data(filename="Data Sets.xlsx",sheet_name=sheet_name, rows_to_skip=rows_to_skip)
    df = np.asarray(df)

    # THE CODE TAKES IN THE NO OF TRAINING SAMPLES, 
    # NOT THE RATIO OF TRAINING SAMPLES FOR SPLITTING
    no_of_training_examples = 480

    train = train_test_split(df, no_of_training_examples)
    min_clusters = 2
    max_clusters = 11
    # Min and Max clusters can be adjusted, defaulted to 
    fuzzy = FuzzyCMeans(dataset=train,min_clusters=min_clusters, max_clusters=max_clusters)
    fuzzy.main_fuzzy_c_means_algo(sheet_name=sheet_name)
    
if __name__ == "__main__":
    FuzzyCMeansClustering()
    FuzzyCMeansClassification()