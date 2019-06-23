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
        self.E = E
        self.prob = nn_params['bayesian_flag']
        self.networks = []
        for i in range(self.E):
            self.networks.append(GeneralNN(nn_params))

    def train_cust(self, dataset, train_params, gradoff = False):
        #try splitting each nn to be trained on different segments of the dataset. This prevents any one of the nns to get overfitted and can better represents our data
        trainLoss = 0
        testLoss = 0
        length = dataset[0].shape[0]
        trainingLoss = np.zeros((self.E, train_params["epochs"]))
        testingLoss = np.zeros((self.E, train_params["epochs"]))
        lastEpoch = None
        for (i, net) in enumerate(self.networks):
            print("Training network number: ", i + 1)
            acctest, acctrain = net.train_cust(dataset, train_params, gradoff = True)
            trainingLoss[i,:len(acctrain)-1] = (np.asarray(acctrain[:-1]))
            testingLoss[i,:len(acctrain)-1] = (np.asarray(acctest[:-1])) #averaging is done two lines down...no need to divide by self.E
            if lastEpoch == None or len(acctrain) - 1 < lastEpoch:
                lastEpoch = len(acctrain) - 1
        trainingLoss = np.average(trainingLoss[:, :lastEpoch], axis = 0)
        testingLoss = np.average(testingLoss[:, :lastEpoch], axis = 0)
        #plot_ensembles(trainingLoss, testingLoss, train_params["epochs"])

        minitrain = min(trainingLoss)
        minitest = min(testingLoss)
        minepoch = np.where(testingLoss == np.amin(testingLoss))[0][0]

        print("")
        print("Minimum training loss: ", min(trainingLoss))
        print("Minimum testing loss: ", min(testingLoss))
        print("Minimum epoch found: ", minepoch)
        return minitest, minitrain, minepoch

    def init_weights_orth(self):
        for nn in self.networks:
            nn.init_weights_orth()

    def predict(self, X, U):
        prediction = np.zeros([9])

        for net in self.networks:
            prediction += (1/self.E)*net.predict(X,U)

        return prediction

    def plot_ensembles(self, trainingLoss, testingLoss, epochs):
        eps = list(range(1, epochs + 1))
        plt.plot(eps, trainingLoss, 'r--', eps, testingLoss, 'b--')
        plt.show()
        print("")

    def save_model(self, filepath):
        torch.save(self, filepath)
        print("EnsembleModel has been saved to" + filepath)
