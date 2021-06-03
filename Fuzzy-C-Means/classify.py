import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Functions to find the distances given the testing data points and the cluster means
def DikA_squared(dataset, cluster_means):
    dikA = np.zeros((cluster_means.shape[0], dataset.shape[0]))

    A = np.zeros((dataset.shape[0], dataset.shape[0]))
    for i in range(A.shape[0]):
        A[i][i] = 1

    for i in range(cluster_means.shape[0]):
        tempzK_vI = dataset[:] - cluster_means[i]
        tempzK_vI = tempzK_vI.transpose().dot(A).transpose() * tempzK_vI
        #print(temp[:,0])#, temp[:,1])
        dikA[i] = np.add(tempzK_vI[:,0], tempzK_vI[:,1])
    return dikA

# reads the data which is written by the training code
# and parses it into arrays, the cluster means(centroids) and the testing data points
# also deletes the file so that there aren't any extra files made and it is only a temporary storage
def read_classsification_centroid_data():
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
    return np.asarray(good_centroids), np.asarray(good_data_points)

# finds the U matrix given distances and cluster means
def find_U_matrix(dataset, cluster_means):
    M = 2
    dikA = DikA_squared(dataset, cluster_means)
    count_k = 0
    U = np.zeros((cluster_means.shape[0], dataset.shape[0]))
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
def classify(U):
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
def plot_testing_clusters(results, cluster_means):
    colors = cm.rainbow(np.linspace(0, 1, cluster_means.shape[0]))
    plt.scatter(results[:,0], results[:,1], c=colors[results[:,2].astype(int)], s=4)
    plt.scatter(cluster_means[:,0], cluster_means[:,1], c='black', s=50)
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("FCM Clustering (Classification Result)")
    plt.show()

def FuzzyCMeansClassification():
    cluster_means, test = read_classsification_centroid_data()
    
    U = find_U_matrix(test, cluster_means)
    clustered = classify(U)

    results = np.zeros((test.shape[0], 3))
    for i in range(test.shape[0]):
        results[i,0] = test[i,0]
        results[i,1] = test[i,1]
        results[i,2] = clustered[i].astype(int)
    
    plot_testing_clusters(results=results, cluster_means=cluster_means)
    return

if __name__ == "__main__":
    FuzzyCMeansClassification()

