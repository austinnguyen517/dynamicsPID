'''Dynamics model to test PID optimization scheme. Constructs a neural network that takes in a state action pair
and returns the resulting state.

State has 9 parameters while action has 4 parameters for total of 13

The model is optimized using MSELoss function and Adam optimize with learning rate of .01 and discount factor of .9

Uses data provided by past runs of the crazyflie

'''

import torch
import torch.nn as nn
import math
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F

import copy


#somehow import all of our training data....

BATCH_SIZE = 64
LR = .01
DF = .9

class Net(nn.Module):
    def __init__(self, nn_params):
        super(Net, self).__init__()
        self.in_n_action = nn_params['du']
        self.in_n_state = nn_params['dx']
        self.in_n = self.in_n_action + self.in_n_state
        self.out_n = nn_params['dt']
        self.width_hid = nn_params['hid_width']
        self.activation = nn_params['activation']
        self.dropout = nn_params['dropout']
        self.features = nn.Sequential(
                nn.Linear(self.in_n, self.width_hid),
                copy.deepcopy(self.activation),
                nn.Dropout(p = self.dropout),
                nn.Linear(self.width_hid, self.width_hid),
                copy.deepcopy(self.activation),
                nn.Dropout(p = self.dropout),
                nn.Linear(self.width_hid, self.width_hid),
                copy.deepcopy(self.activation),
                nn.Dropout(p = self.dropout),
                nn.Linear(self.width_hid, self.out_n / 3) #only looking at euler angles
        )
        #maybe include drop out to prevent overfitting

    def forward(self, x):
        x = self.features(x)
        return x

    def predict(self, X, u): #complete this
        self.features.eval()
        normX = self.scalarX.transform(X.reshape(1, -1))
        normU = self.scalarU.transform(U.reshape(1, -1))


        input = torch.Tensor(np.concatenate((normX, normU), axis=1))

        NNout = self.forward(input).data[0]

        return NNout

    def train_cust(self, dataset, train_params):
        epochs = train_params['epochs']
        batch_size = train_params['batch_size']
        optim = train_params['optim']
        split = train_params['split']
        lr = train_params['lr']
        lr_step_eps = train_params['lr_schedule'][0]
        lr_step_ratio = train_params['lr_schedule'][1]
        self.loss_fnc = nn.MSELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr = lr) #this might be wrong...Nathan uses the super class and passes self for first argument
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = lr_step_eps, gamma = lr_step_ratio)

        testLoss, trainLoss = self.optimize(self.loss_fnc, optimizer, split, lr_scheduler, epochs, batch_size, dataset)

        return testLoss, trainLoss

    def optimize(self, optimizer, split, lr_scheduler, epochs, batch_size, dataset):
        errors = []
        trainErrors = []

        trainLoad = DataLoader(dataset[:int(split*len(dataset))], batch_size = batch_size, shuffle = True)
        testLoad = DataLoader(dataset[int(split*len(dataset)):], batch_size = batch_size)

        for epoch in range(epochs):
            scheduler.step()

            for i, (input, target) in enumerate(trainLoad):
                optim.zero_grad()
                output = self.forward(input)
                loss = self.loss_fnc(output, target)
                loss.backward()
                optim.step()

            self.features.eval()
            test_error = torch.zeros(1)
            for i, (input, target) in enumerate(testLoad):
                output = self.forward(input)
                loss = self.loss_fnc(output, target)
                test_error += loss.item()/(len(testLoad) * batch_size)
            self.features.train()
            error_train.append(avg_loss.data[0].numpy())
            errors.append(test_error.data[0].numpy())

        return errors, error_train

    def save(self, path):
        torch.save(self, path)
