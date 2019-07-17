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

################################################################################
    '''General methods. Init and methods to return certain parameters'''

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

################################################################################
    '''Clusters the given data into k clusters. Also finds x, the number of points in the smallest cluster.'''

    def cluster(self, data):
        '''General attributes and variables'''

        concatenated = np.hstack(data)
        print("Number of data points before sampling: ", concatenated.shape[0])
        self.KMean.fit(concatenated)
        self.totPoints = concatenated.shape[0]
        self.distance = self.KMean.inertia_
        labels = self.KMean.labels_

        '''Mapping each of the datapoints to the cluster they are attached to'''

        for index in list(range(self.totPoints)):
            cluster = labels[index]
            if cluster in self.assignments:
                self.assignments[cluster] = np.vstack((self.assignments[cluster], concatenated[index: index + 1, :]))
            else:
                self.assignments[cluster] = concatenated[index: index + 1, :]

        '''Finding the number of points in the smallest cluster'''

        sizes = []
        for lst in self.assignments.values():
            sizes += [lst.shape[0]]
        self.x = min(sizes)
        print("Data has been clustered into ", self.k, " clusters")

################################################################################
    '''Samples self.x points from each of the k clusters for training. Also, samples more data points from clusters that have:
        - Extreme values in linear acceleration
        - Mid-range values in euler angles          '''

    def sample(self):
        '''General attributes and varibles'''

        BUFF = .2 #designates how many more of these points we want in the training set (1 + BUFF) = percentage more than other clusters
        PADDING = 0 #add this many sampling points to each of the clusters to ideally get closer to the .8 split ratio

        if (len(self.assignments) == 0):
            print("No clustering yet. Sampling unsuccessful")
            return None
        result = None
        empty = True
        '''
        Finding centroids that correspond to requested extra requested data points

        centroids = list(self.KMean.cluster_centers_)
        indices = list(range(self.k))
        map = list(zip(indices, centroids))
        map.sort(key = lambda centroid: centroid[1][-7] + centroid[1][-8] + centroid[1][-9]) #not squared since negative values need to be taken into account
        highOmega = map[:int(self.k / 6)] + map[5* int(self.k / 6):] #take the far sixths of extremes
        map.sort(key = lambda points: (points[1][-4] + points[1][-5] + points[1][-6]))
        midEuler = map[int(self.k / 3): 2 * int(self.k / 3)]
        assert self.KMean.predict(centroids[0].reshape(1,-1)) == 0 #Checks that assumption is correct...that the index of our centroids corresponds to the labels

        Sampling from the requested extra centroids
        for i in range(int(self.x * BUFF)):
            for centroid in highOmega:
                label = centroid[0]
                points = self.assignments[label]
                if points.shape[0] == 0:
                    print("Out of points to sample from cluster: ", cluster)
                    break;
                idx = np.random.randint(0, points.shape[0])
                if empty:
                    result = points[idx, :]
                    empty = False
                else:
                    result = np.vstack((result, points[idx, :]))
                self.assignments[label]=  np.delete(points, idx, 0)
            for centroid in midEuler:
                label = centroid[0]
                points = self.assignments[label]
                if points.shape[0] == 0:
                    print("Out of points to sample from cluster: ", cluster)
                    break;
                idx = np.random.randint(0, points.shape[0])
                result = np.vstack((result, points[idx, :]))
                self.assignments[label]=  np.delete(points, idx, 0)
        print("Sampled an extra ", len(result), " from the requested centroids.")
        print("")'''

        '''Normal sampling from each of the clusters'''
        for i in range(self.x + PADDING):
            for cluster, points in self.assignments.items():
                if points.shape[0] == 0:
                    print("Out of points to sample from cluster: ", cluster)
                    break
                idx = np.random.randint(0, points.shape[0])
                if empty:
                    result = points[idx, :]
                    empty = False
                else:
                    result = np.vstack((result, points[idx,:]))
                self.assignments[cluster] = np.delete(points, idx, 0)

        '''Returns the sampled points and the points that were not sampled in two different variables'''

        numPoints = len(result)
        print("Number of data points after sampling: ", numPoints)
        empty = True
        leftover = None
        for cluster, points in self.assignments.items():
            if empty:
                leftover = points
                empty = False
            else:
                leftover = np.vstack((leftover, points))
        return result, leftover

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
