'''Uses all of the classes and functions we have defined thus far to make a wrapper function.

Steps:
    - Gathers data from quadcopter simulation from initial state inputs from execute train
    - Uses data from sim to train ensemble neural network with optimal parameters found previously
    - Uses the ensemble neural network in bayesian optimization to find a new set of optimal PID parameters
    - Uses those PID parameters to form a policy to pass into the simulation from initial state
    - Repeat '''
##############################################################################
'''Imports'''
import numpy as np
import torch
import torch.nn as nn
import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI
from opto import regression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from ExecuteTrain import radToDeg
import sys

from ExecuteTrain import getOptimalNNTrainParams, getState
from PIDPolicy import policy
from CrazyFlieSim import CrazyFlie
from EnsembleNN import EnsembleNN
from PID_BO_iono import BOPID, scatter3d
from operator import add
from operator import sub
###############################################################################
'''Methods of Policy/Fitness and Cycle Parameters'''
POLICYMODE = 'EULER' #euler, rate, hybrid etc
BOMODE = 'IAE'#Only IAE and Hurst implemented (Hurst might be buggy)
BOOBJECTIVE = 'EULER' #euler, rate, hybrid, all etc
CYCLES = 20#number of times we will cycle through this process
EQUILIBRIUM = [30000, 30000, 30000, 30000] #experimented and confirmed...intuitive..just make them all equal

'''Bayesian Optimization Parameters'''
BOITERATIONS = 100 #number of bayesian optimization iterations

'''Simulation Data Parameters'''
SIMFLIGHTS = 15 #number of flights the simulator will run while gathering data
MAXROLLOUT = 1000 #max frames a rollout from the simulation should run
MINFRAMES = 950 #minimum number of frames needed for a rollout to be considered "good" 250 == 10 seconds
REWARDTIME = True #tooggles whether to add additional data points that are considered "good" flights
VISUALS = False #toggles whether or not to display quadcopter visualization 
dt = .01 #100 Hz frequency
ADDNOISE = True
MAXANGLE = 10

'''Model Training Parameters'''
EPOCHS = 5#10#number of epochs training the network in each cycle #5 if noisevector 10 if not
LR = .0001 #beginning learning rate. #not being used currently
STEP = 5 #number of epochs to train before decaying the LR #use 6 if adding noisevector in simulation...30 if not
DECAY = .7 #amount to decay the learning rate by
KCLUSTER = False #toggles whether or not to use Kclustering on the data gathered from simulations
DECAYLR = True #toggles whether to decay the learning rate over time
DATASIZE = 2000
path = "EnsembleBOModelCycle1.txt"

###############################################################################
'''Initial Conditions and Variable Declarations'''
if POLICYMODE == 'EULER':
    #PID = [173.56, 181.99, 4.96, 173.48, 180.566, 1.82, 171.881, 173.279, 1.744]
    PID = [12,0,1,2,0,1,4,10,1]
elif POLICYMODE == 'HYBRID':
    PID = [41.229, 32.96, 2.17, 33.926, 40.742, .3328, 25.174, 36.356, .2985, 24.969, 35.21, .005]
elif POLICYMODE == 'RATE' or POLICYMODE == 'ALL':
    #PID = [.008, 250, 4.9, .006, 250, 5, .002, 250, 4.9, .008, .006, .004, .001, .003, .00186, .012, 250, .005] #initial PID parameters (18)
    PID = [150, 150, 1, 150, 150, 1, 150, 150, 1, 25, 25, .1, 25, 25, .1, 25, 25, .1]
else:
    print("Error. Invalid policy mode selected.")
    sys.exit(0)

nn_params, train_params = getOptimalNNTrainParams(True, epochs = EPOCHS) #feel free to change the parameters returned by this method
ensemble = EnsembleNN(nn_params)
ensemble.init_weights_orth()
trainingLoss = [] #for the model...probably
testingLoss = []
objectiveLoss = [] #for bayesian optimization
simulationTime = [] #for simulation flight times
simstd = [] #for simulation flight time variance
simavg = []
offsets = []
PIDParameters = [[] for p in range(len(PID))]
allData = None
firstCycle = True
stepCounter = 0

################################################################################
'''Helper to graph 2D projections onto 3D plane'''
def twoOnThreeGraph(dataset, offsets, sizes, title):
    maxSize = max(sizes)
    resolution = 60
    dataset = np.degrees(dataset)
    x = np.linspace(-30, 30, resolution) #each "bin has a width of (30 + 30) / 60 = 1"
    y = []
    z = []
    #make a loop that basically makes a list of np arrays...each one corresponding to a different point in the cycle
    for i in range(len(offsets)):
        newy = np.ones(resolution) * (i + 1)
        newz = np.zeros(resolution)
        insert = dataset[offsets[i]:offsets[i] + sizes[i]]
        for j in list(range(insert.size)): #sort each data point into one of the bins by rounding it to the nearest integer.
            newz[30 - int(round(insert[j]))] += 1 #THIS ONLY WORKS BECAUSE WE ARE SORTING INTO INTEGER 1 SIZE BINS
        y += [newy]
        z += [newz]

    pl.figure()
    ax = pl.subplot(projection='3d')
    for i in range(len(y)):
        ax.plot(x, y[i], z[i], color = 'r')

    ax.set_title(title)
    ax.set_xlabel('Euler Angle Value')
    ax.set_zlabel('Number of Occurrences')
    ax.set_ylabel('Cycle Number')
    plt.show()
################################################################################
'''Main Loop'''

for i in range(CYCLES):
    print("")
    print("")
    print("")
    print("####################### STARTING CYCLE NUMBER ", i + 1, "###################################")
    print("")
    print("Gathering data from simulation...")
    pol = policy(PID, POLICYMODE, dt, equil = EQUILIBRIUM)
    simulator = CrazyFlie(dt)
    if firstCycle:
        initStates = getState()
    else:
        initStates = allData[:, :12]
    simTimeTotal, avgTime, simVar, avgError = simulator.test_policy(initStates, pol, SIMFLIGHTS, VISUALS, MAXROLLOUT, addNoise = ADDNOISE, maxCond = MAXANGLE) #simLoss is inversely proportional to the length of the flights
    print("")
    print("Gathered data from ", SIMFLIGHTS, " rollouts with total flight time of: ", simTimeTotal, " average time ticks: ", avgTime, " average error: ", avgError)
    newData = simulator.get_recent_data() #structure: s0(12), a(4), s1(12), r(1)
    assert not np.any(np.isnan(newData)) #check for nans in dataset
    dataset, extra = simulator.stackDataset(newData, 12, 4, 12, ensemble.stack, REWARDTIME, MINFRAMES)
    if firstCycle:
        allData = np.hstack(dataset)
        firstCycle = False
    else:
        allData = np.vstack((allData, np.hstack(dataset)))
    if REWARDTIME:
        dataset = (np.vstack((dataset[0], extra[0])), np.vstack((dataset[1], extra[1])), np.vstack((dataset[2], extra[2])))
    print("")
    print("Training model...")
    print("")
    if DECAYLR: #condition to check if we should change the LR. We don't know whether the mod will equal 0...train_cust trains it for multiple epochs
        if (stepCounter >= STEP) and stepCounter != 0:
            for params in train_params:
                params['lr'] = params['lr'] * DECAY
            stepCounter = stepCounter - STEP #modulus for some reason did not work
            STEP *= 2
            print("LR UPDATE: Network 1 learning rate has been changed to: ", train_params[0]['lr'])
            print("")
    dataidx = np.random.choice(allData.shape[0], DATASIZE)
    dataTrain = allData[dataidx, :]
    dataTrain = (dataTrain[:, :12], dataTrain[:, 12:16], dataTrain[:, 16:28])
    testError, trainError = ensemble.train_cust(dataTrain, train_params, KCLUSTER) #the value returned is the average testing/training loss over ALL epochs across ALL networks in ensemble
    for netnum, network in enumerate(ensemble.networks):
        if any(torch.isnan(val).byte().any() for val in network.state_dict().values()):
            print("Error: Nan value in state dict of network number: ", netnum + 1, " (not zero indexed).")
            sys.exit(0)
    print("")
    print("Running BO on PID Parameters and Model...")
    print("")
    BO = BOPID(ensemble, BOMODE, BOOBJECTIVE, POLICYMODE, BOITERATIONS,  EQUILIBRIUM, dt, True, dataset, path)
    BO.optimize()
    newParams, BOloss = BO.getParameters(False, False)
    print("New PID parameters found after ", BOITERATIONS, " iterations with objective loss of: ", BOloss)
    PID = newParams
    stepCounter += EPOCHS

    '''Store values to display later'''
    for pidIndex in range(len(PID)):
        PIDParameters[pidIndex] += [PID[pidIndex]]
    simulationTime += [simTimeTotal]
    simavg += [[avgTime, avgError]]
    simstd += [simVar**(1/2)]
    testingLoss += [testError]
    trainingLoss += [trainError]
    objectiveLoss += [BOloss]
    if offsets == []:
        offsets += [0]
    else:
        offsets += [offsets[i-1] + simulationTime[i-1]]

################################################################################
'''Epilogue: displaying results'''
#Plotting average simulation time and variance over cycles
simavgarray = np.array(simavg)
avgError = (simavgarray[:, 1].T).tolist()
avgTime = (simavgarray[:,0].T).tolist()
print("")
plt.plot(list(range(CYCLES-1)), avgTime[1:], 'r--')
plt.plot(list(range(CYCLES-1)), list(map(add, avgTime[1:], simstd[1:])), 'b--')
plt.plot(list(range(CYCLES-1)), list(map(sub, avgTime[1:], simstd[1:])), 'b--')
plt.plot(list(range(CYCLES-1)), avgError[1:], 'g--')
plt.title("Simulation rollout time (y) over cycles (x) with standard deviation. Simulation error in green.")
plt.show()

#Plotting relationship between BO loss and respective simulation time
plt.plot(objectiveLoss[:-1], avgError[1:], 'ro')
plt.title("Average simulation error (y) with respect to BO objective loss (x)")

#Distributions of euler angles over time
twoOnThreeGraph(allData[:, 7].T, offsets, simulationTime, 'Pitch Distributions over Cycles (Not Including Failure Points)')
twoOnThreeGraph(allData[:, 8].T, offsets, simulationTime, 'Roll Distributions over Cycles (Not Including Failure Points)')
twoOnThreeGraph(allData[:, 6].T, offsets, simulationTime, 'Yaw Distributions over Cycles (Not including Failure Poionts)')

#Training and testing loss
plt.plot(list(range(CYCLES)), trainingLoss, 'r--', list(range(CYCLES)), testingLoss, 'b--')
plt.title("Training and Testing Loss of Model(y) over Cycles(x)")
red = mpatches.Patch(color = 'red', label = 'Training')
blue = mpatches.Patch(color = 'blue', label = 'Testing')
plt.legend(handles = [red, blue])
plt.show()

#3D scatter plots of PID parameters w.r.t. simulation time
scatter3d(PIDParameters[0], PIDParameters[1], PIDParameters[2], avgError, "Pitch Parameters w.r.t. SIMULATION ERROR")
scatter3d(PIDParameters[3], PIDParameters[4], PIDParameters[5], avgError, "Roll Parameters w.r.t. SIMULATION ERROR")
scatter3d(PIDParameters[6], PIDParameters[7], PIDParameters[8], avgError, "Yaw Parameters w.r.t. SIMULATION ERROR")
if POLICYMODE == 'HYBRID':
    scatter3d(PIDParameters[9], PIDParameters[10], PIDParameters[11], avgError, "Yaw Rate Parameters w.r.t. SIMULATION ERROR")
if POLICYMODE == 'RATE':
    scatter3d(PIDParameters[9], PIDParameters[10], PIDParameters[11], avgError, "Pitch Rate Parameters w.r.t. SIMULATION ERROR")
    scatter3d(PIDParameters[12], PIDParameters[13], PIDParameters[14], avgError, "Roll Rate Parameters w.r.t. SIMULATION ERROR")
    scatter3d(PIDParameters[15], PIDParameters[16], PIDParameters[17], avgError, "Yaw Rate Parameters w.r.t. SIMULATION ERROR")

###############################################################################
'''PRINT SUMMARY TO SCREEN'''
print("########################     Summary:    #################################")
print("")
print("Total Cycles: ", CYCLES, "   BOIterations: ", BOITERATIONS, "    Epochs: ", EPOCHS,"      SimFlights: ", SIMFLIGHTS)
print("Policy mode: ", POLICYMODE, " BO Objective: ", BOOBJECTIVE)
highestFrames = max(map(lambda x: x[0], simavg))
lowestLoss = min([lst[1] for lst in simavg if lst[0] == highestFrames])
c = simavg.index([highestFrames, lowestLoss])
print("Max average simulation flight time: ", highestFrames, " found at cycle number: ", c + 1, " with loss of ", lowestLoss)
print("Minimum model test loss: ", min(testingLoss), " found at cycle number: ", testingLoss.index(min(testingLoss)) + 1)
print("Minimum model train loss: ", min(trainingLoss), "found at cycle number: ", trainingLoss.index(min(trainingLoss)) + 1)
print("Minimum BO objective loss: ", min(objectiveLoss), "found at cycle number ", objectiveLoss.index(min(objectiveLoss)) + 1)
print("")
slope, intercept, r_value, p_value, std_error = stats.linregress(objectiveLoss[:-1], simulationTime[1:])
print("Linear Regression on BO objective loss vs Flight Simulation Time had slope: ", slope, " intercept: ", intercept, " and r value of ", r_value)
print("")
print("Best performing PID values in terms of simulator fitness from cycle number ", c + 1 , ":")
print("")
print("Pitch:   Prop: ", PIDParameters[0][c], " Int: ", PIDParameters[1][c], " Deriv: ", PIDParameters[2][c])
print("Roll:    Prop: ", PIDParameters[3][c], " Int: ", PIDParameters[4][c], " Deriv: ", PIDParameters[5][c])
print("Yaw:     Prop: ", PIDParameters[6][c], " Int: ", PIDParameters[7][c], " Deriv: ", PIDParameters[8][c])
if POLICYMODE == 'HYBRID':
    print("YawRate: Prop: ", PIDParameters[9][c], " Int: ", PIDParameters[10][c], "Deriv: ", PIDParameters[11][c])
if POLICYMODE == 'RATE':
    print("PitchRt: Prop: ", PIDParameters[9][c], " Int: ", PIDParameters[10][c], " Deriv: ", PIDParameters[11][c])
    print("RollRate:Prop: ", PIDParameters[12][c], " Int: ", PIDParameters[13][c], " Deriv: ", PIDParameters[14][c])
    print("YawRate: Prop: ", PIDParameters[15][c], " Int: ", PIDParameters[16][c], "Deriv: ", PIDParameters[17][c])
