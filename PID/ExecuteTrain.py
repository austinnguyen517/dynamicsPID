import torch
from torch.autograd import Variable
from torch.nn import MSELoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from GenNN import GeneralNN
from Parse import load_dirs, df_to_training
from utils.nn import Swish
from EnsembleNN import EnsembleNN

dir_list = ["data/c25_rand/",
            "data/c25_roll1/",
            "data/c25_roll2/",
            "data/c25_roll3/",
            "data/c25_roll4/",
            "data/c25_roll5/",
            "data/c25_roll6/",
            "data/c25_roll7/",
            "data/c25_roll8/",
            "data/c25_roll9/",
            "data/c25_roll10/",
            "data/c25_roll11/",
            "data/c25_roll12/",
            "data/c25_roll13/",
            "data/c25_roll14/",
            "data/c25_roll15/",]
'''
load_params ={
    'delta_state': True,                # normally leave as True, prediction mode
    'include_tplus1': True,             # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
    'trim_high_vbat': 4050,             # trims high vbat because these points the quad is not moving
    'takeoff_points': 180,              # If not trimming data with fast log, need another way to get rid of repeated 0s
    'trim_0_dX': True,                  # if all the euler angles (floats) don't change, it is not realistic data
    'find_move': True,
    'trime_large_dX': True,             # if the states change by a large amount, not realistic
    'bound_inputs': [20000,65500],      # Anything out of here is erroneous anyways. Can be used to focus training
    'stack_states': 3,                  # IMPORTANT ONE: stacks the past states and inputs to pass into network
    'collision_flag': False,            # looks for sharp changes to tthrow out items post collision
    'shuffle_here': False,              # shuffle pre training, makes it hard to plot trajectories
    'timestep_flags': [],               # if you want to filter rostime stamps, do it here
    'battery' : True,                   # if battery voltage is in the state data
    'terminals': True,                 # adds a column to the dataframe tracking end of trajectories
    'fastLog' : True,                   # if using the software with the new fast log
    'contFreq' : 1,                      # Number of times the control freq you will be using is faster than that at data logging
    'iono_data': True,
    'zero_yaw': True,
    'moving_avg': 7
}

df = load_dirs(dir_list, load_params)

data_params = {
    # Note the order of these matters. that is the order your array will be in
    'states' : ['omega_x0', 'omega_y0', 'omega_z0',
                'pitch0',   'roll0',    'yaw0',
                'lina_x0',  'lina_y0',  'lina_z0',
                'omega_x1', 'omega_y1', 'omega_z1',
                'pitch1',   'roll1',    'yaw1',
                'lina_x1',  'lina_y1',  'lina_z1',
                'omega_x2', 'omega_y2', 'omega_z2',
                'pitch2',   'roll2',    'yaw2',
                'lina_x2',  'lina_y2',  'lina_z2'],
                # 'omega_x3', 'omega_y3', 'omega_z3',
                # 'pitch3',   'roll3',    'yaw3',
                # 'lina_x3',  'lina_y3',  'lina_z3'],

    'inputs' : ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
                'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2'],# 'vbat'],
                # 'm1_pwm_3', 'm2_pwm_3', 'm3_pwm_3', 'm4_pwm_3', 'vbat'],

    'targets' : ['t1_omega_x', 't1_omega_y', 't1_omega_z',
                        'd_pitch', 'd_roll', 'd_yaw',
                        't1_lina_x', 't1_lina_y', 't1_lina_z'],

    'battery' : False                    # Need to include battery here too
}

X, U, dX = df_to_training(df, data_params)

nn_params = {                           # all should be pretty self-explanatory
    'dx' : np.shape(X)[1],
    'du' : np.shape(U)[1],
    'dt' : np.shape(dX)[1],
    'hid_width' : 250,
    'hid_depth' : 2,
    'bayesian_flag' : True,
    'activation': Swish(),
    'dropout' : 0.2,
    'split_flag' : False,
    'pred_mode' : 'Delta State',
    'ensemble' : 5
}

train_params = {
    'epochs' : 35, #start at this # of epochs to find optimal epochs w/ respect to testerror
    'batch_size' : 18,
    'optim' : 'Adam',
    'split' : 0.95,
    'lr': .00175, # bayesian .00175, mse:  .0001
    'lr_schedule' : [6,.7],
    'test_loss_fnc' : [],
    'preprocess' : True,
    'noprint' : False
}

#newNN = GeneralNN(nn_params)
#newNN.init_weights_orth()
#testLoss, trainLoss = newNN.train_cust((X, U, dX), train_params)
#path = "TrainedDModel.txt"
#newNN.save_model(path)

ensembleNN = EnsembleNN(nn_params)
ensembleNN.init_weights_orth()
ensembleNN.train_cust((X, U, dX), train_params)
path = "TrainedEnsembleModel2.txt"
ensembleNN.save_model(path)'''

def getinput(model):
    load_params ={
        'delta_state': True,                # normally leave as True, prediction mode
        'include_tplus1': True,             # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
        'trim_high_vbat': 4050,             # trims high vbat because these points the quad is not moving
        'takeoff_points': 180,              # If not trimming data with fast log, need another way to get rid of repeated 0s
        'trim_0_dX': True,                  # if all the euler angles (floats) don't change, it is not realistic data
        'find_move': True,
        'trime_large_dX': True,             # if the states change by a large amount, not realistic
        'bound_inputs': [20000,65500],      # Anything out of here is erroneous anyways. Can be used to focus training
        'stack_states': 3,                  # IMPORTANT ONE: stacks the past states and inputs to pass into network
        'collision_flag': False,            # looks for sharp changes to tthrow out items post collision
        'shuffle_here': False,              # shuffle pre training, makes it hard to plot trajectories
        'timestep_flags': [],               # if you want to filter rostime stamps, do it here
        'battery' : True,                   # if battery voltage is in the state data
        'terminals': True,                 # adds a column to the dataframe tracking end of trajectories
        'fastLog' : True,                   # if using the software with the new fast log
        'contFreq' : 1,                      # Number of times the control freq you will be using is faster than that at data logging
        'iono_data': True,
        'zero_yaw': True,
        'moving_avg': 7
    }

    df = load_dirs(dir_list, load_params)

    data_params = {
        # Note the order of these matters. that is the order your array will be in
        'states' : ['omega_x0', 'omega_y0', 'omega_z0',
                    'pitch0',   'roll0',    'yaw0',
                    'lina_x0',  'lina_y0',  'lina_z0',
                    'omega_x1', 'omega_y1', 'omega_z1',
                    'pitch1',   'roll1',    'yaw1',
                    'lina_x1',  'lina_y1',  'lina_z1',
                    'omega_x2', 'omega_y2', 'omega_z2',
                    'pitch2',   'roll2',    'yaw2',
                    'lina_x2',  'lina_y2',  'lina_z2'],
                    # 'omega_x3', 'omega_y3', 'omega_z3',
                    # 'pitch3',   'roll3',    'yaw3',
                    # 'lina_x3',  'lina_y3',  'lina_z3'],

        'inputs' : ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
                    'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                    'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2'],# 'vbat'],
                    # 'm1_pwm_3', 'm2_pwm_3', 'm3_pwm_3', 'm4_pwm_3', 'vbat'],

        'targets' : ['t1_omega_x', 't1_omega_y', 't1_omega_z',
                            'd_pitch', 'd_roll', 'd_yaw',
                            't1_lina_x', 't1_lina_y', 't1_lina_z'],

        'battery' : False                    # Need to include battery here too
    }

    X, U, dX = df_to_training(df, data_params)
    input = GeneralNN.preprocess(model, (X,U,dX)) #list of tuples representing inputs and outputs
    return input
