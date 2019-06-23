import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class kClusters():
    '''Clustering class used to preprocess datasets
    Attributes:
        k: number of clusters
        totPoints: total number of points
        distance: sum of all distances from point to respective centroid
        KMean: Sci-kit Kmeans class instance
        assignments: dictionary mapping clusters to list of points
        x: minimum number of points in all of the clusters
    Methods:
        numberClusters: returns number of clusters
        labels: returns the centroids
        cluster: takes a dataset and clusters each datapoint into k clusters
        sample: uniformly samples from each of the clusters and returns array with x points from each cluster
        plot: takes a dataset and performs clustering with k within given range inclusive. Displays graph with inertia wrt #clusters'''

    def __init__(self, numClusters = 8):
        self.k = numClusters
        self.totPoints = 0
        self.distance = None
        self.KMean = KMeans(n_clusters = numClusters)
        self.assignments = {}
        self.x = 0 #minimum number of points in a cluster

    def numberClusters(self):
        return k

    def labels(self):
        return self.KMean.labels_

    def cluster(self, data):
        #cluster data on k clusters.
        concatenated = np.hstack(data)
        print("Number of data points before sampling: ", concatenated.shape[0])
        self.KMean.fit(concatenated)
        self.totPoints = concatenated.shape[0]
        self.distance = self.KMean.inertia_
        labels = self.KMean.labels_
        for index in list(range(self.totPoints)):
            cluster = labels[index]
            if cluster in self.assignments:
                self.assignments[cluster] = np.vstack((self.assignments[cluster], concatenated[index, :]))
            else:
                self.assignments[cluster] = concatenated[index, :]
        sizes = []
        for lst in self.assignments.values():
            sizes += [lst.shape[0]]
        self.x = min(sizes)
        print("Data has been clustered into ", self.k, " clusters")

    def sample(self): #automatically samples equal to number of smallest cluster
        #make sure assignments are not none
        if (len(self.assignments) == 0):
            print("No clustering yet. Sampling unsuccessful")
            return None
        result = None
        empty = True
        for i in range(self.x):
            for cluster, points in self.assignments.items():
                idx = np.random.randint(0, points.shape[0])
                if empty:
                    result = points[idx, :]
                    empty = False
                else:
                    result = np.vstack((result, points[idx,:]))
        #otherwise sample x points from each cluster. Make sure to take ONE data point from each cluster and iterrate that many timesteps
        print("Number of data points after sampling: ", self.x * self.k)
        return (result[:, :27], result[:,27:39], result[:,39:])

    def plot(self, clusters, data):
        curr = clusters[0]
        end = clusters[1]
        c = []
        inertias = []
        data = np.hstack(data)
        while curr <= end:
            km = KMeans(n_clusters = curr)
            km.fit(data)
            c += [curr]
            inertias += [km.inertia_]
            print("Clustered with k == ", curr)
            curr += 1
        plt.plot(c, inertias, 'b--')
        plt.show()
        return 0
