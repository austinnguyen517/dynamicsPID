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
    def __init__(self, nn_params, E = 5):
        super(EnsembleNN, self).__init__()
        self.E = E
        self.prob = nn_params['bayesian_flag']
        self.networks = []
        for i in range(self.E):
            self.networks.append(GeneralNN(nn_params))
        self.scalarX = StandardScaler()# MinMaxScaler(feature_range=(-1,1))#StandardScaler()# RobustScaler()
        self.scalarU = MinMaxScaler(feature_range=(-1,1))
        self.scalardX = MinMaxScaler(feature_range=(-1,1)) #StandardScaler() #MinMaxScaler(feature_range=(-1,1))#StandardScaler() # RobustScaler(quantile_range=(25.0, 90.0))


    def train_cust(self, dataset, train_params, gradoff = False):
        #try splitting each nn to be trained on different segments of the dataset. This prevents any one of the nns to get overfitted and can better represents our data
        trainLoss = 0
        testLoss = 0
        length = dataset[0].shape[0]
        setSize = length // self.E
        trainingLoss = []
        testingLoss = []
        epochs = []
        totalEpochs = train_params['epochs']
        currEpochs = 5
        train_params['epochs'] = 5
        while (currEpochs <= totalEpochs):
            for (i, net) in enumerate(self.networks):
                #firstIndex = i*setSize
                #endIndex = (i+1)*setSize
                #input = (dataset[0][firstIndex:endIndex], dataset[1][firstIndex:endIndex], dataset[2][firstIndex:endIndex])
                input = (dataset[0], dataset[1], dataset[2])
                acctest, acctrain = net.train_cust(input, train_params, gradoff = True)
                trainLoss += min(acctrain)/self.E
                testLoss += min(acctest)/self.E
            trainingLoss += [trainLoss]
            testingLoss += [testLoss]
            epochs += [currEpochs]
            currEpochs += 5
            trainLoss = 0
            testLoss = 0
        plt.plot(epochs, trainingLoss, 'r--', epochs, testingLoss, 'b--')
        plt.show()
        return testLoss, trainLoss

    def init_weights_orth(self):
        for nn in self.networks:
            nn.init_weights_orth()

    def predict(self, X, U):
        prediction = np.zeros([9])

        for net in self.networks:
            prediction += (1/self.E)*net.predict(X,U)

        return prediction

    def save_model(self, filepath):
        torch.save(self, filepath)
        print("EnsembleModel has been saved to" + filepath)
