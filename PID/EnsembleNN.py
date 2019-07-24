import pickle
import numpy as np

# neural nets
from GenNN import GeneralNN

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal

# More NN such
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import pickle
from sklearn.model_selection import KFold   # for dataset
from kMeansData import kClusters


from PNNLoss import PNNLoss_Gaussian
import matplotlib.pyplot as plt

class EnsembleNN(nn.Module):
    '''Ensemble neural network class
        Attributes:
            E: number of networks/ensembles
            prob: flag signaling PNN
            networks: list of networks/ensembles
        Methods:
            train_cust: takes data and train parameters to train each network
            init_weights_orth: initializes all weights/matrices in nn to be orthonormal
            predict: makes averaged prediction of next state given a current state/action pair
            plot_ensembles: makes a plot of training loss and testing loss with respect to epochs
            save_model: saves model to a certain path location'''

    def __init__(self, nn_params, E = 5):
        super(EnsembleNN, self).__init__()
        '''Lacks generality since we assume that each of the networks are trained'''
        self.E = E
        self.prob = nn_params[0]['bayesian_flag']
        self.networks = []
        self.n_in_input = nn_params[0]['du']
        self.n_in_state = nn_params[0]['dx']
        self.n_in = self.n_in_input + self.n_in_state
        self.n_out = nn_params[0]['dt']
        if self.prob:
            self.n_out = self.n_out * 2
        self.stack = nn_params[0]['stack']
        for i in range(self.E):
            self.networks.append(GeneralNN(nn_params[i]))
        self.epsilon = nn_params[0]['epsilon']


    def train_cust(self, dataset, train_params, cluster = False, gradoff = False):
        #print("")
        if cluster:
            print("")
            print("Clustering...")
            km = kClusters(min(np.shape(dataset[0])[0],9)) #dummy number of clusters
            #km.plot([2, 20], (X,U,dX))
            km.cluster(dataset)
            dataset, leftover = km.sample() #shuffling is handled in GeneralNN training loader...only shuffles the training data so the split is safe.
            lenTrain = len(dataset)
            lenTest = len(leftover)
            for dict in train_params:
                dict["split"] =lenTrain / (lenTrain + lenTest)
                print("Used a split ratio of: ", lenTrain / (lenTrain + lenTest))
            print("")
            dataset = np.vstack((dataset, leftover))
            dataset = ((dataset[:, :36], dataset[:, 36:48], dataset[:, 48:]))
        TrTePairs = []
        mintrain = []
        mintest = []
        lastEpoch = None

        '''Training each of the neural networks on the same dataset with different parameters'''

        for (i, net) in enumerate(self.networks):
            #print("Training network number: ", i + 1)
            acctest, acctrain = net.train_cust(dataset, train_params[i], gradoff = False)
            if acctrain[-1] == float('nan'): #get rid of the last number if it is Nan
                TrTePairs += [[acctrain[:-1], acctest[:-1]]]
                mintrain += [min(acctrain[:-1])]
                mintest += [min(acctest[:-1])]
            else:
                TrTePairs += [[acctrain, acctest]]
                mintrain += [min(acctrain)]
                mintest += [min(acctest)]
            #print("Training stopped after: ", len(acctrain), " with min test loss of ", mintest[i])
            #print("")

        '''Displaying the results'''

        #self.plot_ensembles(TrTePairs)
        #print("")
        #print("")
        #print("RESULTS:")
        #for i in range(len(self.networks)):
            #print("Network number", i + 1, ":", " Minimum Testing Loss: ", mintest[i], " Minimum Training Loss: ", mintrain[i], " Epochs trained: ", len(TrTePairs[i][0]))
        #print("")
        mintest = sum(mintest) / (len(mintest))
        mintrain = sum(mintrain) / (len(mintrain))
        print("Overall: Average testing loss: ", mintest, " Average training loss: ", mintrain)

        return mintest, mintrain


    def init_weights_orth(self):
        for nn in self.networks:
            nn.init_weights_orth()

    def predict(self, X, U):
        prediction = np.zeros((len(X), self.networks[0].n_out))

        for net in self.networks:
            for i in range(len(X)):
                mean, var = (net.predict(X[i, :], U[i, :]))
                prediction [i, :] += ((1/self.E) * np.hstack((mean,var)))

        return prediction #+ self.epsilon #adding it because our training included this epsilon

    def plot_ensembles(self, pairs):
        for pair in pairs:
            #training is 0, testing is 1
            eps = list(range(1, len(pair[0]) + 1))
            plt.plot(eps, pair[0], 'r--', eps, pair[1], 'b--')
            plt.show()

    def save_model(self, filepath):
        torch.save(self, filepath)
        print("")
        print("EnsembleModel has been saved to " + filepath)
