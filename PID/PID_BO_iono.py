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
# Objective functions for PID BO
def InvHuber(x):
    """
    - Huber loss is linear outside of a quadratic inner region
    - ours is quadratic outside of a linear region in pitch and roll
    cost = -(c*p^2)  if p < a, for pitch and roll
    TODO: We will have to add loss when the mean PWM is below a certain value
    """
    pitch = x[:,0]
    roll = x[:,1]
    yaw = x[:,2]

    # tunable parameters
    a1 = 1
    a2 = 1
    lin_pitch = 5
    lin_roll = 5

    # sum up loss (Not vectorized because lazy)
    loss_pitch_total = 0
    loss_roll_total = 0

    for p,r in zip(pitch,roll):
        if p > lin_pitch:
            loss_pitch = a1*p**2
        else:
            loss_pitch = a1*abs(p)

        if r > lin_roll:
            loss_roll = a2*r**2
        else:
            loss_roll = a2*abs(r)

        loss_pitch_total += loss_pitch
        loss_roll_total += loss_roll

    loss = loss_pitch_total+loss_roll_total
    return .001*loss

def Time(x):
    """
    Returns value proportional to flight length
    """
    l = np.shape(x)[0]
    return -l

def Dual(x):
    """
    Returns a weighted objective value with a balance of Euler angles and flight time
    """
    invhuber = InvHuber(x)
    time = Time(x)
    a1 = 1
    a2 = 1
    total = a1*invhuber + a2*time
    return total


############################################################################
# Main function for executing PID experiments in opto BO code
def PID_Objective(mode='Custom'):
    """
    Objective function of state data for a PID parameter tuning BO algorithm.
    Max flight time 5 seconds during rollouts.

    Modes:
    - Time (time until failure of PID control run)
    - InvHuber (InvHuber loss defined by linear near 0, quadratic away from that for pitch, roll)
    - Dual (A weighted combo of time and Euler angle InvHuber)
    """
    assert mode in ['Time', 'InvHuber', 'Dual', 'Custom'], 'Objective Function Not Found'

    model = torch.load("TrainedEnsembleModel2.txt")
    model.eval()
    inputs = getinput(model)
    length = len(inputs)

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


    def objective(x):
        """
        Assess the objective value of a trajectory X.
        """

        # various modes of the objective function.
        if mode == 'Time':
            obj_val = len(x)

        elif mode == 'InvHuber':
            obj_val = InvHuber(x)
        elif mode == 'Custom':
            #initialize a random state (only omega and euler angle values really matter
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

            #gets the proper length of 39 vector to pass as input
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
            equil = [34687.1, 37954.7, 38384.8, 36220.11]
            #print(state)
            #print("")
            #print(action)
            min_pwm = 20000
            max_pwm = 65500
            dt = .04 #data is at 25Hz

            pitchPID = PID(0, x[0,0], x[0,1], x[0,2], 20, dt) #why are we limiting the i term????
            rollPID = PID(0, x[0,3], x[0,4], x[0,5], 20, dt)
            yawratePID = PID(0, x[0,6], x[0,7], x[0,8], 360, dt)
            #pitchratePID = PID(0, x[0,6], x[0,7], x[0,8], 0, dt)
            #rollratePID = PID(0, x[0,9], x[0,10], x[0,11], 0, dt)

            iLoss = 0
            dLoss = 0
            thrustLoss = 0
            #rateLoss = 0

            itg = 1 #tuneable weights for each loss
            drv = .1
            mgn = .0000001 #this needs to be really low since magnitudes are order e6
            #rateitg = .3


            for i in range(250): #tuneable number of runs for each PID input. Remember: 4 of these are used for just getting a full input
                if state[3] >= 30 or state[4] >= 30: #in case of system failure
                    print("Roll or pitch has exceeded 30 degrees. Ending run after ", i, " iterations!")
                    iLoss += 60 * (250 - (i+ 1))
                    break

                #pass into the model
                output = model.predict(state, action) #pre and post processing already handled
                newState = torch.from_numpy(output).float()

                iLoss += abs(newState[3].detach() + newState[4].detach() + newState[2].detach()) * dt

                pitchPID.update(newState[3])
                rollPID.update(newState[4])
                yawratePID.update(newState[2])
                #pitchratePID.update(pitchPID.out)
                #rollratePID.update(rollPID.out)

                new_act = new_action(min_pwm, max_pwm, pitchPID, rollPID, yawratePID, equil).float()
                #thrustLoss += torch.norm(new_act.detach())
                #rateLoss += abs(newState[6] + newState[7] + newState[8]) * dt
                #dLoss += (-1) * math.log(((abs(state[3] - newState[3]) + abs(state[4] - newState[4]))/dt))

                state = torch.cat((state.narrow(0,9,18), newState.detach()), 0)
                action = torch.cat((action.narrow(0,4,8), new_act), 0)
            iLoss = iLoss * itg
            #thrustLoss = thrustLoss * mgn
            #dLoss = dLoss * drv
            #rateLoss = rateLoss * rateitg

            totLoss = iLoss #+ thrustLoss + dLoss  # + rateLoss.detach() # + dLoss.detach()
            #if finished:
                #print("SUCCESSFUL ITERATION! Loss:", totLoss, "Parameters:", x)
                #print("")
            print("Loss:", totLoss, " after ", i, "iterations")
            print("")
            return totLoss

        else:
            obj_val = len(x) - InvHuber(x)
        return obj_val

    return objective

PID_Object = PID_Objective(mode='Custom')
task = OptTask(f=PID_Object, n_parameters=9, n_objectives=1, \
            bounds=bounds(min=[0,0,0,0,0,0,0,0,0],max=[250 ,250,5, 250,250,5, 250,250, 5]), task = {'minimize'}, vectorized=False, \
            labels_param = ['KP_pitch','KI_pitch','KD_pitch', 'KP_roll' 'KI_roll', 'KD_roll',
                            "KP_yaw", "KI_yaw", "KD_yaw"])
Stop = StopCriteria(maxEvals=2)

p = DotMap()
p.verbosity = 1
p.acq_func = EI(model = None, logs = None) #EI(model = None, logs = logs)
p.model = regression.GP
opt = opto.BO(parameters=p, task=task, stopCriteria=Stop)
opt.optimize()

log = opt.get_logs()
parameters = log.get_parameters()
losses = log.get_objectives()
best = log.get_best_parameters()
bestLoss = log.get_best_objectives()
nEvals = log.get_n_evals()

print(losses)

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

#assuming that the parameters are represented as a list of lists
Proll = []
Iroll = []
Droll = []
Ppitch = []
Ipitch = []
Dpitch = []
Pyaw = [] #rate
Iyaw = []
Dyaw = []

'''for parameter in parameters:
    Proll += [parameter[0, 0]]
    Iroll += [parameter[0,1]]
    Droll += [parameter[0,2]]
    Ppitch += [parameter[0,3]]
    Ipitch += [parameter[0,4]]
    Dpitch += [parameter[0,5]]
    Pyaw += [parameter[0,6]]
    Iyaw += [parameter[0,7]]
    Dyaw += [parameter[0,8]]'''

print("Best PID parameters found: ", best, " with loss of ", bestLoss, " in ", nEvals, " evaluations.")
plt.title("Evals vs Losses")
plt.plot(list(range(nEvals)), losses[0])
plt.show()]
scatter3d(Proll,Iroll,Droll, losses, "Losses W.R.T. Roll PIDs")
scatter3d(Ppitch,Ipitch,Dpitch, losses, "Losses W.R.T. Pitch PIDs")
scatter3d(Pyaw,Iyaw,Dyaw, losses, "Losses W.R.T. Yaw Rate PIDs")
