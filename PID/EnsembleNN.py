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
        self.prob = nn_params[0]['bayesian_flag']
        self.networks = []
        for i in range(self.E):
            self.networks.append(GeneralNN(nn_params[i]))

    def train_cust(self, dataset, train_params, gradoff = False):
        TrTePairs = []
        mintrain = []
        mintest = []
        lastEpoch = None

        for (i, net) in enumerate(self.networks):
            print("Training network number: ", i + 1)
            acctest, acctrain = net.train_cust(dataset, train_params[i], gradoff = True)
            if acctrain[-1] == float('nan'): #get rid of the last number if it is Nan
                TrTePairs += [[acctrain[:-1], acctest[:-1]]]
                mintrain += [min(acctrain[:-1])]
                mintest += [min(acctest[:-1])]
            else:
                TrTePairs += [[acctrain, acctest]]
                mintrain += [min(acctrain)]
                mintest += [min(acctest)]
            print("Training stopped after: ", len(acctrain), " with min test loss of ", mintest[i])
            print("")

        #displaying the results
        self.plot_ensembles(TrTePairs)
        print("")
        print("")
        print("RESULTS:")
        for i in range(len(self.networks)):
            print("Network number", i + 1, ":", " Minimum Testing Loss: ", mintest[i], " Minimum Training Loss: ", mintrain[i], " Epochs trained: ", len(TrTePairs[i][0]))
        print("")
        mintest = sum(mintest) / (len(mintest))
        mintrain = sum(mintrain) / (len(mintrain))
        print("Overall: Average testing loss: ", mintest, " Average training loss: ", mintrain)

        return mintest, mintrain


    def init_weights_orth(self):
        for nn in self.networks:
            nn.init_weights_orth()

    def predict(self, X, U):
        prediction = np.zeros([9])

        for net in self.networks:
            prediction += (1/self.E)*net.predict(X,U)

        return prediction

    def plot_ensembles(self, pairs):
        for pair in pairs:
            #training is 0, testing is 1
            eps = list(range(1, len(pair[0]) + 1))
            plt.plot(eps, pair[0], 'r--', eps, pair[1], 'b--')
            plt.show()

    def save_model(self, filepath):
        torch.save(self, filepath)
        print("EnsembleModel has been saved to" + filepath)
