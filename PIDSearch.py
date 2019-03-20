import torch
import torch.nn as nn
import math
import gpytorch
import numpy as np
from matplotlib import pyplot as plt

#general class for PID
class PID():
    def __init__(self, desired,
                    kp, ki, kd,
                    ilimit, dt, outlimit = np.inf,
                    samplingRate = 0, cutoffFreq = -1,
                    enableDFilter = False):

        self.error = 0  
        self.error_prev = 0
        self.integral = 0
        self.deriv = 0
        self.out = 0

        self.desired = desired
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.ilimit = ilimit
        self.outlimit = outlimit

        # timee steps for changing step size of PID response
        self.dt = dt
        self.samplingRate = samplingRate    # sample rate is for filtering

        self.cutoffFreq = cutoffFreq

    def update(self, measured):
        self.out = 0.

        self.error_prev = self.error

        self.error = self.desired - measured
        self.out += self.kp*self.error

        self.deriv = (self.error-self.error_prev) / self.dt
        self.out += self.deriv*self.kd

        self.integral = self.error*self.dt

        # limitt the integral term
        if self.ilimit !=0:
            self.integral = np.clip(self.integral,-self.ilimit, self.ilimit)

        self.out += self.ki*self.integral

        # limitt the total output
        if self.outlimit !=0:
            self.out = np.clip(self.out, -self.outlimit, self.outlimit)

        return self.out

class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, gp_model, best_y):
        super(ExpectedImprovement, self).__init__(gp_model)
        self.best_y = best_y

    def forward(self, candidate_set):
        self.grid_size = 10000

        self.gp_model.eval()
        self.gp_model.likelihood.eval()

        pred = self.gp_model.likelihood(self.gp_model(candidate_set))

        mu = pred.mean().detach()
        sigma = pred.std().detach()

        u = (self.best_y - mu) / sigma
        m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        ucdf = m.cdf(u)
        updf = torch.exp(m.log_prob(u))
        ei = sigma * (updf + u * ucdf)

        return ei

class ExactGPModel(gpytorch.models.ExactGP):
    #the class for our GP model used in our loss function
    def __init__():
        super(ExactGPModel, self).__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood
        self.mean_module = gpytorch.mean.ConstantMean()
        self.cov_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.cov_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

class PIDBO():
    def __init__(self, itg, drv, mgn, model):
        #parameters should include target state, and respective weights for each of the calculations for loss
        #loss parameters:
            #itg: the weight of the summation of deviations for a given period
            #drv: the weight of the change in deviations for a given period
            #mgn: the weight of the magnitude of the PID inputs

        #loss parameters
        self.itg = itg
        self.drv = drv
        self.mgn = mgn
        self.loss = 0

        #models
        self.dynamicsModel = model
        self.lossModel = ExactGPModel()
        self.EI = ExpectedImprovement(lossModel)
        self.state = torch.zeros(9)

        #updating the loss model
        self.optimizer = torch.optim.Adam([{'params': lossModel.parameters()}], lr = 0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, lossModel)

        #current PIDs that give the lowest loss
        self.min = None
        self.minLoss = None
        self.dt = .1 #the timesteps in which we update the PID output

        #currentPIDS we are regulating
        self.PIDRoll = PID(0, 1, 1, 1, 1, dt)
        self.PIDPitch = PID(0, 1, 1, 1, 1, dt)

        #Limits for thrust
        self.equil = [30000, 30000, 30000, 30000]
        self.min_pwm = 0
        self.max_pwm = 65535
        self.output = torch.zeros(4)

    def main():
        #wrapper that ties all the processes together
        train() #put the model in training mode
        for (j in range(1000)):
            action = new_action()# given the current state, calculate an action to pursue
            N = 100
            states = torch.tensor(2, N)

        # input the state and action pair into the model and get a new state. Repeat for N iterations and concatenate together
            for (i in range(N)):
                input = torch.cat((state, action), 0)
                self.state = self.dynamicsModel(input)
                states[0][i] = self.state[3]
                states[1][i]= self.state[4]

            pid_in = torch.tensor([[self.PIDRoll.kp, self.PIDRoll.ki, self.PIDRoll.kd], [self.PIDPitch.kp, self.PIDPitch.ki, self.PIDPitch.kd]])
            loss = self.forward(states, pid_in)

            print("New Iteration:")
            print("Loss:", loss)
            print("Current State:", self.state)

            self.predict(self.chooseCandidates())

    def train(): #puts the model in training mode
        self.model.train()
        self.likelihood.train()

    def forward (self, output, inputs):
        #Looks at rolls and pitches and deviations from target
        #output: torch tensor (2,N) denoting the ROLLS and PITCHES at N timestamps
        #target: goal state (equilibrium in this case)
        #t: the increments of time between sampling states
        #inputs: 2 by 3 tensor representing PID inputs kp, ki, kd of ROLL and PITCH
        N = output.size()[1]
        target = torch.zeros(2, N)
        prev = torch.zeros(2, N).put_(output[1:])

        itgLoss = itg * torch.sum((torch.abs(output - target)) * self.dt) #hopefully (2,1) tensor
        mgnLoss = mgn * torch.sum(inputs) #hopefully (2,1) tensor
        drvLoss = drv * -torch.log(torch.sum((output[1:] - prev) / dt)) #negative log. Smaller changes yield larger loss. Want to correct deviations quickly

        loss = itgLoss + mgnLoss + drvLoss
        if self.EI.best_y == None or -loss > self.best_y: #negative since EI maximizes values
            self.EI.best_y = -loss
        if self.min == None or loss < self.minLoss:
            self.min = inputs
            self.minLoss = loss
        updateLossModel(inputs, loss)

        return loss

    def limit_thrust(PWM): #Limits the thrust
        return np.clip(PWM, self.min_pwm, self.max_pwm)

    def new_state(self, input):
        return self.dynamicsModel(input)


    def new_action(self):
        self.output[0] = limit_thrust(self.equil[0] + self.PID_att_pitch.out)
        self.output[1] = limit_thrust(self.equil[1] - self.PID_att_roll.out)
        self.output[2] = limit_thrust(self.equil[2] - self.PID_att_pitch.out)
        self.output[3] = limit_thrust(self.equil[3] + self.PID_att_roll.out)
        return self.output

    def updateLossModel(inputs, ActualLoss):
    #updates the loss GP model by adding in the point with inputs and loss
        inputs = inputs.view(1,-1)
        prevLoss = lossModel(Variable(inputs)).mean() #since the lossModel probably returns a gaussian distribution
        error = self.mll(prevLoss, ActualLoss.detach())
        optimizer.zerograd()
        error.backward()
        optimizer.step()

        print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (i + 1, training_iter, loss.item(),
                self.dynamicsModel.covar_module.base_kernel.log_lengthscale.item(),
                self.dynamicsModel.likelihood.log_noise.item()))

    def chooseCandidates():
    #try implementing something that smartly picks which candidates to choose
        N = 20
        return torch.rand(N,6)

    def predict(candidates):
    #changes pid inputs that will give the most expected improvement
        ei = self.EI(candidates)
        value, index = tensor.max(ei)
        best = candidates(index)
        PIDRoll.kp = best[0]
        PIDRoll.ki = best[1]
        PIDRoll.kd = best[2]
        PIDPitch.kp = best[3]
        PIDPitch.ki = best[4]
        PIDRoll.kd = best[5]
