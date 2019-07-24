'''Wrapper function that takes inputs of PID parameters, creates a policy, and runs a simulation with visuals on the policy.
   All simulations are based on outputs from CrazyFlieSim.py and noise can be added if need be. '''

from CrazyFlieSim import CrazyFlie
from PIDPolicy import policy
from ExecuteTrain import getState
import numpy
import torch
import math

#insert PID parameters here...
    #order: pitch roll yaw prate rrate yrate in PID order
PID = [162.4553068437353,92.33729041600824, .89670033561082,
        168.51970708675145, 104.62663226974074, 2.316215840641,
        167.76854190354715, 99.82218564660144,4.3133254181221]
dt = .01
equil = [30000, 30000, 30000, 30000]
PIDMODE = 'EULER'
ITERATIONS = 10
VISUALIZE = True
MAXROLLOUT = 500
ADDNOISE = True
MAXANGLE = 30

simulator = CrazyFlie(dt)
PIDpolicy = policy(PID, PIDMODE, dt)
states = getState()

totTime, avgTime, varTime, avgError = simulator.test_policy(states, PIDpolicy, ITERATIONS, VISUALIZE, maxFrames = MAXROLLOUT, addNoise = ADDNOISE, maxCond = MAXANGLE)
print("Results:")
print("Total Time: ", totTime, " with maximum total time of ", ITERATIONS * MAXROLLOUT)
print("Average Time: ", avgTime, " with maximum average time of ", MAXROLLOUT)
print("Standard Deviation of time: ", varTime**(1/2), " over ", ITERATIONS, " iterations.")
print("Average error: ", avgError, " with maximum error of ", 3 * math.radians(MAXANGLE) * MAXROLLOUT)
