#define the policy we are using
#pass into the model that we are using
#put that into our array of rolls, pitches etc.
#plot the arrays of roll,pitches with respect to our time steps
#remember we are operating at 25Hz at the moment
import torch
import numpy as np
import matplotlib.pyplot as plt
from EnsembleNN import EnsembleNN
from PID import PID
from ExecuteTrain import getinput

def policy(min_pwm, max_pwm, PIDpitch, PIDroll, PIDyawrate, equil):
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

model = torch.load("TrainedEnsembleModel2.txt")
model.eval()
inputs = getinput(model)
length = len(inputs)

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
min_pwm = 0
max_pwm = 65500
dt = .04 #data is at 25Hz

pitchPID = PID(0, 108.83, 77.14, 3.62, 0, dt) #ilimit = 20? or 0
rollPID = PID(0, 242.33, 1.556, 3.23, 0, dt) #ilimit = 20? or 0
yawratePID = PID(0, 231.21, 172.95, 1.25, 0, dt) #ilimit = 360? or 0

time = []
rolls = []
pitch = []
yawrate = []

for i in range(250):
    if state[3] >= 30 or state[4] >= 30: #in case of system failure
        print("Roll or pitch has exceeded 30 degrees. Ending run after ", i, " iterations!")
        break

    output = model.predict(state, action)
    newState = torch.from_numpy(output).float()
    time += [i * dt]
    rolls += [newState[4]]
    pitch += [newState[3]]
    yawrate += [newState[2]]
    pitchPID.update(newState[3])
    rollPID.update(newState[4])
    yawratePID.update(newState[2])
    new_act = policy(min_pwm, max_pwm, pitchPID, rollPID, yawratePID, equil).float()
    state = torch.cat((state.narrow(0,9,18), newState.detach()), 0)
    action = torch.cat((action.narrow(0,4,8), new_act), 0)

plt.plot(time, rolls, 'b--', time, pitch, 'r--', time, yawrate, 'y--')
plt.legend(loc = 'upper left')
plt.show()
