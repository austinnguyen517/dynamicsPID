import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression


import numpy as np
import matplotlib.colors as color
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from dotmap import DotMap

import json
import datetime
import glob
import pandas
import os
import random as rand
import torch
import math

from GenNN import GeneralNN
from PID import PID
from ExecuteTrain import getinput
from sklearn.preprocessing import StandardScaler, MinMaxScaler

############################################################################
'''Main function for executing PID experiments in opto BO. General information'''
def PID_Objective(mode='Custom'):
    """
    Objective function of state data for a PID parameter tuning BO algorithm.
    Max flight time 10 seconds during rollouts. Operating at 25 Hz -> 250 Iterations.
    """
    '''states' : ['omega_x0', 'omega_y0', 'omega_z0',
                'pitch0',   'roll0',    'yaw0',
                'lina_x0',  'lina_y0',  'lina_z0',
                'omega_x1', 'omega_y1', 'omega_z1',
                'pitch1',   'roll1',    'yaw1',
                'lina_x1',  'lina_y1',  'lina_z1',
                'omega_x2', 'omega_y2', 'omega_z2',
                'pitch2',   'roll2',    'yaw2',
                'lina_x2',  'lina_y2',  'lina_z2']'''
    '''    'inputs' : ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
                    'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                    'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2']'''

################################################################################
'''Setting up Bayesian Optimization: model, initial conditions, parameters'''

    assert mode in ['IAE', 'Hurst'], 'Objective Function Not Found'
    model = torch.load("TrainedEnsembleModel2.txt")
    model.eval()
    inputs = getinput(model)
    length = len(inputs)

    equil = [34687.1, 37954.7, 38384.8, 36220.11]
    min_pwm = 20000
    max_pwm = 65500
    dt = .04 #data is at 25Hz

###############################################################################
'''General methods'''
    def new_action(min_pwm, max_pwm, PIDpitch, PIDroll, PIDyawrate, equil):
        '''TODO: Play with this policy. You can try to calculate multiple rollouts for multiple timesteps (horizon) and select the one with minimal loss
                    Then, choose only the top action'''
        def limit_thrust(PWM): #Limits the thrust
            return np.clip(PWM, min_pwm, max_pwm)
        output = torch.zeros(1,4).float()
        output[0][0] = limit_thrust(equil[0] + PIDpitch.out + PIDyawrate.out) #- pitchratePID.out)
        output[0][1] = limit_thrust(equil[1] - PIDroll.out - PIDyawrate.out) #+ rollratePID.ou)
        output[0][2] = limit_thrust(equil[2] - PIDpitch.out + PIDyawrate.out) #+ pitchratePID.out)
        output[0][3] = limit_thrust(equil[3] + PIDroll.out - PIDyawrate.out) #- rollratePID.out)
        return output[0]

    def get_initial_condition():
        validInput = False
        while (not validInput):
            randidx = np.random.randint(0, length)
            input = inputs[randidx][0].float()
            input = input.clone().detach()

            scalarX = model.scalarX
            scalarU = model.scalarU
            state = torch.from_numpy(scalarX.inverse_transform(input.narrow(0, 0, 27).expand(1, -1))[0]).float() #the current state is the first 9 elements of the input
            action = torch.from_numpy(scalarU.inverse_transform(input.narrow(0, 27, 12).expand(1,-1))[0]).float() #the most recent action is the first 4 elements of the action part of input
            if (state[3] < 5 and state[4] < 5):
                validInput = True
        return state, action

    def record_results(x, pLoss, rLoss, yLoss):
        Ppitch.append(x[0,0])
        Ipitch.append(x[0,1])
        Dpitch.append(x[0,2])
        pitchLoss.append((pLoss * itg).tolist())
        Proll.append(x[0,3])
        Iroll.append(x[0,4])
        Droll.append(x[0,5])
        rollLoss.append((rLoss * itg).tolist())
        Pyaw.append(x[0,6]) #rate
        Iyaw.append(x[0,7])
        Dyaw.append(x[0,8])
        yawLoss.append((yLoss * itg).tolist())

    def devFromLstSqr(errors):
        #Perform least square
        x = np.array(list(range(len(errors))))
        A = np.vstack([x, np.ones(len(errors))]).T
        m, c = np.linalg.lstsq(A, errors, rcond = None)[0]
        #Detrend the errors
        x = (m * x) + c
        resid = errors - x
        return resid
###############################################################################
'''Objective functions '''
    def IAE(x):
        pitchPID = PID(0, x[0,0], x[0,1], x[0,2], 20, dt) #why are we limiting the i term????
        rollPID = PID(0, x[0,3], x[0,4], x[0,5], 20, dt)
        yawratePID = PID(0, x[0,6], x[0,7], x[0,8], 360, dt)

        iLoss = 0
        rLoss = 0
        pLoss = 0
        yLoss = 0

        itg = 1 #tuneable weights for each loss
        state, action = get_initial_condition():

        for i in range(250): #tuneable number of runs for each PID input. Remember: 4 of these are used for just getting a full input
            if state[3] >= 30 or state[4] >= 30: #in case of system failure
                print("Roll or pitch has exceeded 30 degrees. Ending run after ", i, " iterations!")
                iLoss += 60 * (250 - (i+ 1))
                break

            '''Pass into the model'''
            output = model.predict(state, action) #pre and post processing already handled
            assert not torch.isnan(torch.FloatTensor(output)).byte().any() #this technically returns a tensor with 0 or 1s. 1 if there is Nan in that element...check if this statement works
            newState = torch.from_numpy(output).float()
            pLoss = abs(newState[3].detach()) * dt
            rLoss = abs(newState[4].detach()) * dt
            yLoss = abs(newState[2].detach()) * dt
            iLoss += abs(newState[3].detach() + newState[4].detach() + newState[2].detach()) * dt

            pitchPID.update(newState[3])
            rollPID.update(newState[4])
            yawratePID.update(newState[2])

            new_act = new_action(min_pwm, max_pwm, pitchPID, rollPID, yawratePID, equil).float()

            state = torch.cat((state.narrow(0,9,18), newState.detach()), 0)
            action = torch.cat((action.narrow(0,4,8), new_act), 0)

        totLoss = iLoss * itg

        print("Loss:", totLoss, " after ", i, "iterations")
        print("")

        record_results(x, pLoss, rLoss, yLoss)

        return totLoss

    def Hurst(x):
        '''The Hurst exponent is an attribute that, when computed, represents
        the presence of long-term correlation among error signals. Generally an exponent
        value of:
            - alpha = .5  -> white noise signal
            - alpha > .5 -> correlation in time series signal
            - alpha < .5 -> anti-correlation in time series signal
        As a result, alpha of .5 indicates good PID fitness

        Steps to calculate Hurst exponent:
         1. Find E(k) (the sum of mean centered signals up to k)
         2. Divide E(k) into boxes of equal length n (min 10 max N/4)
         3. For each box, perform linear least squares
         4. Subtract E(k) by the local trend
         5. Calculate F(n) (sqrt mean squared deviation from local trend of all points of all boxes)
         6. Repeat over many different size boxes n
         7. Find log-log plot of F(n) versus n.
         8. Calculate gradient of the line. This is the value of the Hurst exponent'''

        pitchPID = PID(0, x[0,0], x[0,1], x[0,2], 20, dt) #why are we limiting the i term????
        rollPID = PID(0, x[0,3], x[0,4], x[0,5], 20, dt)
        yawratePID = PID(0, x[0,6], x[0,7], x[0,8], 360, dt)

        rErrors = []
        pErrors = []
        yErrors = []
        boxes = [10, 12, 20, 30, 40, 60]
        state, action = get_initial_condition()

        for i in range(240):
            if state[3] >= 30 or state[4] >= 30: #in case of system failure
                print("Roll or pitch has exceeded 30 degrees. Ending run after ", i, " iterations!")
                iLoss += 60 * (250 - (i+ 1))
                break

            '''Pass into the model'''
            output = model.predict(state, action) #pre and post processing already handled
            assert not torch.isnan(torch.FloatTensor(output)).byte().any()
            newState = torch.from_numpy(output).float()

            if rErrors == []:
                pErrors = np.array([abs(newState[3].detach())])
                rErrors = np.array([abs(newState[4].detach())])
                yErrors = np.array([abs(newState[2].detach())])
            else:
                pErrors = pErrors.hstack(np.array([pErrors[i - 1] + abs(newState[3].detach())]))
                rErrors = rErrors.hstack(np.array([rErrors[i - 1] + abs(newState[4].detach())]))
                yErrors = yErrors.hstack(np.array([yErrors[i - 1] + abs(newState[2].detach())]))

            pitchPID.update(newState[3])
            rollPID.update(newState[4])
            yawratePID.update(newState[2])
            new_act = new_action(min_pwm, max_pwm, pitchPID, rollPID, yawratePID, equil).float()
            state = torch.cat((state.narrow(0,9,18), newState.detach()), 0)
            action = torch.cat((action.narrow(0,4,8), new_act), 0)

        pErrors = pErrors - np.mean(pErrors)
        rErrors = rErrors - np.mean(rErrors)
        yErrors = yErrors - np.mean(yErrors)
        n = []
        pF = []
        rF = []
        yF = []
        N = len(pErrors)

        for size in boxes:
            i = 0
            pbox = []
            rbox = []
            ybox = []
            while (i < len(pErrors)):
                pboxResults = devFromLstSqr(pErrors[i:i + size])
                rboxResults = devFromLstSqr(rErrors[i: i + size])
                yboxResults = devFromLstSqr(yErrors[i: i + size])
                if pbox == []:
                    pbox = pboxResults
                    rbox = rboxResults
                    ybox = yboxResults
                else:
                    pbox = np.hstack(pbox, pboxResults)
                    rbox = np.hstack(robx, rboxResults)
                    ybox = np.hstack(ybox, yboxResults)
                i += size
            pFvalue =  (np.sum(np.square(pbox)) / N) ** (1/2)
            rFvalue = (np.sum(np.square(rbox)) / N) ** (1/2)
            yFvalue = (np.sum(np.square(ybox)) / N) ** (1/2)

            n += [size]
            pF += [pFvalue]
            rF += [rFvalue]
            yF += [yFvalue]

        logn = np.log(np.array(n))
        logpF = np.log(np.array(pF))
        logrF = np.log(np.array(rF))
        logyF = np.log(np.array(yF))

        A = np.vstack([logn, np.ones(len(errors))]).T
        alphap, c = np.linalg.lstsq(A, logpF, rcond = None)[0]
        alphar, c = np.linalg.lstsq(A, logrF, rcond = None)[0]
        alphay, c = np.linalg.lstsq(A, logyF, rcond = None)[0]

        return ((alphap - .5)**2 + (alphar - .5) ** 2 + (alphay - .5)**2) ** (1/2)

###############################################################################
'''Wrapper for objective functions'''

    def objective(x):
        """
        Assess the objective value of a trajectory X.
        """

        # various modes of the objective function.
        if mode == 'IAE':
            return IAE(x)
        else:
            return Hurst(x)

    return objective

###############################################################################
'''Opto library setup for Bayesian Optimization'''

PID_Object = PID_Objective(mode='IAE')
task = OptTask(f=PID_Object, n_parameters=9, n_objectives=1, \
            bounds=bounds(min=[0,0,0,0,0,0,0,0,0],max=[250 ,250,5, 250,250,5, 250,250, 5]), task = {'minimize'}, vectorized=False, \
            labels_param = ['KP_pitch','KI_pitch','KD_pitch', 'KP_roll' 'KI_roll', 'KD_roll',
                            "KP_yaw", "KI_yaw", "KD_yaw"])
Stop = StopCriteria(maxEvals=2)

################################################################################
'''Recording results for plotting and saving'''

Proll = []
Iroll = []
Droll = []
rollLoss = []
Ppitch = []
Ipitch = []
Dpitch = []
pitchLoss = []
Pyaw = [] #rate
Iyaw = []
Dyaw = []
yawLoss = []

################################################################################
'''Execution of Bayesian Optimization'''

p = DotMap()
p.verbosity = 1
p.acq_func = EI(model = None, logs = None) #EI(model = None, logs = logs)
p.model = regression.GP
opt = opto.BO(parameters=p, task=task, stopCriteria=Stop)
opt.optimize()

################################################################################
'''Retrieval of parameters and objectives post optimization'''

log = opt.get_logs()
parameters = log.get_parameters()
losses = log.get_objectives()
best = log.get_best_parameters()
bestLoss = log.get_best_objectives()
nEvals = log.get_n_evals()

################################################################################
'''Plotting and visualizing results'''

def scatter3d(x,y,z, cs, name, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = color.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    plt.title(name)
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()

print("Best PID parameters found: ", best, " with loss of ", bestLoss, " in ", nEvals, " evaluations.")
plt.title("Evals vs Losses")
plt.plot(list(range(nEvals)), losses[0])
plt.show()
scatter3d(Proll,Iroll,Droll, rollLoss, "Losses W.R.T. Roll PIDs")
scatter3d(Ppitch,Ipitch,Dpitch, pitchLoss, "Losses W.R.T. Pitch PIDs")
scatter3d(Pyaw,Iyaw,Dyaw, yawLoss, "Losses W.R.T. Yaw Rate PIDs")
