import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression


import numpy as np
import matplotlib.pyplot as plt
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

################################################################################
# A couple files for loading and storing parameters with objective values
def save_params(iteration, params, objectives = [], date=[]):
    """
    Save the parameters at a given iteration to a JSON file for easy reference in the future
    """
    directory = 'data/ionoBO/'

    # Date string if not passed
    if date == []:
        date = datetime.datetime.now().strftime("%Y-%m-%d")

    fname = directory + date + '-iter'+'{:02d}'.format(iteration)+'-params.json'
    with open(fname, 'w') as fp:
        if objectives != []:
            json.dump({**params, **objectives}, fp, indent=4)
        else:
            json.dump(params, fp, indent=4)

def load_params(iteration, date):
    """
    Loads the parameters at a given iteration at a specific date
    Date should be of the form '2011-03-21'
    """
    directory = 'data/ionoBO/'
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    fname = directory + date + '-iter'+'{:02d}'.format(iteration)+'-params.json'
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data

def gen_objectives(data, params_dict):
    """
    Takes in a datafile and returns the three different costs I created for ionocraft BO
    """
    objectives_dict = []
    return objectives_dict, params_dict

def get_info_date(date):
    """
    Use this function to create a dataframe with all the info that the BO needs to pick it's next value
    """
    directory = 'data/ionoBO/'
    files = glob.glob(directory+date+"*")
    param_jsons = [fname for fname in files if 'params' in fname]
    first_j = json.load(open(param_jsons.pop()))
    df = pandas.DataFrame(first_j,index=[0])

    i = 1
    for j in param_jsons:
        df_temp = pandas.DataFrame(json.load(open(j)),index=[1])
        df = df.append(df_temp,ignore_index=True)
        i+=1
    # print(df)
    return df

################################################################################
# Data read function
def load_iono(fname, m_avg = 0):
    """
    Minimal file for loading the raw data from the ionocraft experiment:

    The raw file has lines from Arduino serial print of the form:
    pwm1, pwm2, pwm3, pwm4, ax, ay, az, wx, wy, wz, pitch, roll, yaw
    """
    file = "data/ionoBO/"+fname
    with open(file, "rb") as csvfile:
        # laod data
        # cols_use = np.linspace(0,13,14)
        cols_use = (0,1,2,3,4,5,6,7,8,9,10,11,12)
        new_data = np.genfromtxt(csvfile, delimiter=",", usecols=cols_use, autostrip=True)

    serial_error_flag = (
            ((new_data[:, -1] > -360) & (new_data[:, -1] < 360)) &  # yaw
            ((new_data[:, -2] > -360) & (new_data[:, -2] < 360)) &  # roll
            ((new_data[:, -3] > -360) & (new_data[:, -3] < 360)) &  # pitch
            ((new_data[:, 4] > -500) & (new_data[:, 4] < 500)) &
            ((new_data[:, 5] > -500) & (new_data[:, 5] < 500)) &
            ((new_data[:, 6] > -500) & (new_data[:, 6] < 500))
        )


    new_data = new_data[serial_error_flag,:]

    if False and m_avg > 1:
            # fitlers the euler angles by targeted value
            new_data[:, -1] = np.convolve(
                new_data[:, -1], np.ones((m_avg,))/m_avg, mode='same')
            new_data[:, -2] = np.convolve(
                new_data[:, -2], np.ones((m_avg,))/m_avg, mode='same')
            new_data[:, -3] = np.convolve(
                new_data[:, -3], np.ones((m_avg,))/m_avg, mode='same')

            # filters accelerations by 2
            new_data[:, 4] = np.convolve(
                new_data[:, 4], np.ones((2,))/2, mode='same')
            new_data[:, 5] = np.convolve(
                new_data[:, 5], np.ones((2,))/2, mode='same')
            new_data[:, 6] = np.convolve(
                new_data[:, 6], np.ones((2,))/2, mode='same')

    # Check the range for each val
    if False:
        for c in cols_use:
            print("State col: ", c)
            print("Max val: ", np.max(new_data[:,c]))
            print("Min val: ", np.min(new_data[:, c]))
            print("Mean val: ", np.mean(new_data[:, c]))
            print('------')

    return new_data


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
def PID_Objective(mode='Dual'):
    """
    Objective function of state data for a PID parameter tuning BO algorithm.
    Max flight time 5 seconds during rollouts.

    Modes:
    - Time (time until failure of PID control run)
    - InvHuber (InvHuber loss defined by linear near 0, quadratic away from that for pitch, roll)
    - Dual (A weighted combo of time and Euler angle InvHuber)
    """
    assert mode in ['Time', 'InvHuber', 'Dual', 'Custom'], 'Objective Function Not Found'

    model = torch.load("TrainedDModel.txt")
    model.eval()
    inputs = getinput(model)
    length = len(inputs)

    def new_action(min_pwm, max_pwm, PIDpitch, PIDroll, PIDyawrate, equil):
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

        # TODO: Make the "evaluate" function for this have you load a data file.
        #   Or... we can make a deterministic file structure that it loads all of them in order?

        # various modes of the objective function.
        # TODO: Tune these values
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
            randidx = np.random.randint(0, length)
            input = inputs[randidx][0].float()
            input = input.clone().detach()

            scalarX = model.scalarX
            scalarU = model.scalarU
            state = torch.from_numpy(scalarX.inverse_transform(input.narrow(0, 0, 27).expand(1, -1))[0]).float() #the current state is the first 9 elements of the input
            action = torch.from_numpy(scalarU.inverse_transform(input.narrow(0, 27, 12).expand(1,-1))[0]).float() #the most recent action is the first 4 elements of the action part of input
            equil = [34687.1, 37954.7, 38384.8, 36220.11]
            #print(state)
            #print("")
            #print(action)
            min_pwm = 0
            max_pwm = 65535
            dt = .02
            finished = False

            pitchPID = PID(0, x[0,0], x[0,1], x[0,2], 20, dt)
            rollPID = PID(0, x[0,3], x[0,4], x[0,5], 20, dt)
            yawratePID = PID(0, x[0,6], x[0,7], x[0,8], 360, dt)
            #pitchratePID = PID(0, x[0,6], x[0,7], x[0,8], 0, dt)
            #rollratePID = PID(0, x[0,9], x[0,10], x[0,11], 0, dt)

            iLoss = 0
            dLoss = 0
            mLoss = 0
            #rateLoss = 0

            itg = 5 #tuneable weights for each loss
            drv = .1
            mgn = .2
            #rateitg = .3


            for i in range(50): #tuneable number of runs for each PID input. Remember: 4 of these are used for just getting a full input
                if state[21] >= 30 or state[22] >= 30: #in case of system failure
                    print("Roll or pitch has exceeded 30 degrees. Ending run after ", i, " iterations!")
                    iLoss += 60 * (50 - (i+ 1))
                    print("Loss:", iLoss)
                    print("")
                    break

                #pass into the model
                output = model.predict(state, action) #pre and post processing already handled
                newState = torch.from_numpy(output).float()

                state = torch.cat((newState.detach(), state.narrow(0,0,18)), 0)

                iLoss += abs(newState[3] + newState[4]) * dt

                pitchPID.update(newState[3])
                rollPID.update(newState[4])
                yawratePID.update(newState[2])
                #pitchratePID.update(pitchPID.out)
                #rollratePID.update(rollPID.out)

                new_act = new_action(min_pwm, max_pwm, pitchPID, rollPID, yawratePID, equil).float()
                 #consider passing in rate PIDs for better stability
                action = torch.cat((new_act, action.narrow(0,0,8)), 0)
                #rateLoss += abs(newState[6] + newState[7] + newState[8]) * dt
                #dLoss += abs(output[0][4] + output[0][5]) * -dt
                #if i == 49:
                    #finished = True
            #mLoss = torch.sum(torch.tensor(x)) * mgn
            iLoss = iLoss * itg
            #rateLoss = rateLoss * rateitg
            #dLoss = dLoss * drv

            totLoss = iLoss.detach()# + rateLoss.detach() # + dLoss.detach() + mLoss.detach()
            #if finished:
                #print("SUCCESSFUL ITERATION! Loss:", totLoss, "Parameters:", x)
                #print("")

            return totLoss

        else:
            obj_val = len(x) - InvHuber(x)
        return obj_val

    return objective



    #######################################################
    #date = "2019-03-26" #datetime.datetime.now().strftime("%Y-%m-%d")

    # Parameters from the last run
    #iteration = 0

    # check if this iteration has already been run (to prevent overwriting files)
    #p_str = date+'-iter'+'{:02d}'.format(iteration)+'-params.json'
    #exists = os.path.isfile('data/ionoBO/'+p_str)
    # if exists:
    #     raise ValueError("Preventing File Overwrite, quitting.")
    #     quit()

#param_labels = ['KP_pitch','KP_roll', 'KD_pitch', 'KD_roll']
#param_values = [10, 10, .1, .1]
#param_dict = dict(zip(param_labels,param_values))

    # objectives of last run
    #data_str = date+'-iter'+'{:02d}'.format(iteration)+'-data.txt'
    #data = load_iono(data_str)
    #Eulers = data[:,-3:]
    #obj_labels = ['Dual','Time','InvHuber']
    #obj_time = Time(Eulers)
    #obj_invhuber = InvHuber(Eulers)
    #obj_dual = Dual(Eulers)
    #obj_values = [obj_dual, obj_time, obj_invhuber]
    #obj_dict = dict(zip(obj_labels, obj_values))

    # save these parameters to a json for the iteration
    #save_params(iteration, param_dict, obj_dict, date = date)

    # load all of the parameters for the given date
    #df = get_info_date("2019-03-26")

    # creates dataset object for our parameters and objective values # df[obj_labels]
    #print("Parameters:")
    #print(np.matrix(df[param_labels].values.T))
    #print("Objectives:")
    #print(np.matrix(df['InvHuber'].values.T))
    #data_in = np.matrix(df[param_labels].values.T)
    #data_out = np.matrix(df['InvHuber'].values.T)
    #dataset = rdata.dataset(data_input=data_in, data_output=data_out)
PID_Object = PID_Objective(mode='Custom')
task = OptTask(f=PID_Object, n_parameters=9, n_objectives=1, \
            bounds=bounds(min=[0,0,0,0,0,0,0,0,0],max=[100,30,5, 100,30,5, 100,30, 5]), task = {'minimize'}, vectorized=False, \
            labels_param = ['KP_pitch','KI_pitch','KD_pitch', 'KP_roll' 'KI_roll', 'KD_roll',
                            "KP_yaw", "KI_yaw", "KD_yaw"])
Stop = StopCriteria(maxEvals=50)

    # p_EI = DotMap()
    # p_EI.target = 999
    # print(p_EI.get('target', 0))
    # quit()

    # Create our own log object
#logs = Logs(store_x = True, store_fx = True, store_gx =False)
    #logs.add_evals(x= data_in,          # matrix will all parameters evaluated (N_param x N_Data)
    #    fx = data_out,                  # matrix with all corresponding obj function (N_obj x N_Data)
        # opt_x = ,               # Best parameters so far
    #    opt_fx = np.min(data_out),              # Best obj val so far
    #    nIter = iteration,      # number of iterations performed so far
    #    opt_criteria = 'minima'
    #    )

p = DotMap()
p.verbosity = 1
p.acq_func = EI(model = None, logs = None) #EI(model = None, logs = logs)
# p.acq_func.set_target(999)
# p.optimizer = opto.CMAES
p.model = regression.GP
opt = opto.BO(parameters=p, task=task, stopCriteria=Stop)
opt.optimize()

log = opt.get_logs()
parameters = log.get_parameters()
best = log.get_best_parameters()
bestLoss = log.get_best_objectives()
nEvals = log.get_n_evals()
#print("Optimal PID parameters:", parameters)
print("Best PID paramters found: ", best, " with loss of ", bestLoss, " in ", nEvals, " evaluations.")
    #opt._iter = iteration+1
    #opt._logs = logs

    # load our dataset
    #opt._set_dataset(dataset)

    # directly generate next parameter set
    #next_params = opt._select_parameters()
    #print(next_params)
    ##quit()
    #print("~~~Below is the next set of PID parameter~~~")
    #print("  KP_pitch: ", next_params[0])
    #print("  KP_roll: ",  next_params[1])
    #print("  KD_pitch: ", next_params[2])
    #print("  KD_roll: ",  next_params[3])
    #print(" Other BO Information...")

    # Save the next parameters for reference
    #param_labels = ['KP_pitch','KP_roll', 'KD_pitch', 'KD_roll']
    #param_values = next_params
    #param_dict = dict(zip(param_labels, param_values))

    # save these parameters to a json for the iteration
    #save_params(iteration+1, param_dict, date = date)

    # TODO:
    # in BO the dataset is loaded in a line:
    #   dataset = rdata.dataset(data_input=self._logs.get_parameters(), data_output=self._logs.get_objectives())
    # I can replace this with a new dataset object from loading the df of past trials
    #   df = get_info_date("2019-03-26")
    #   dataset = rdata.dataset(data_input=df[labels], data_output=df[obj_labels])
    # Then should just be able to call
    #   x = opt._select_parameters()
    # x is the best guess of PID parameters

    # opt.optimize()
    # logs = opt.get_logs()
    # print("Parameters: " + str(logs.get_parameters()))
    # print("Objectives: " + str(logs.get_objectives()))
