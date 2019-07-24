'''Code from: Nathan Lambert'''

import numpy as np
import math
from PIDPolicy import policy
from ExecuteTrain import getState, degToRad, radToDeg
from Visualizer import GUI

class CrazyFlie():
    def __init__(self, dt, m=.035, L=.065, Ixx=2.3951e-5, Iyy=2.3951e-5, Izz=3.2347e-5, x_noise=.0000001, u_noise=0):
        self._state_dict = {
            'X': [0, 'pos'],
            'Y': [1, 'pos'],
            'Z': [2, 'pos'],
            'vx': [3, 'vel'],
            'vy': [4, 'vel'],
            'vz': [5, 'vel'],
            'yaw': [6, 'angle'],
            'pitch': [7, 'angle'],
            'roll': [8, 'angle'],
            'w_x': [9, 'omega'],
            'w_y': [10, 'omega'],
            'w_z': [11, 'omega']
        }
        # user can pass a list of items they want to train on in the neural net, eg learn_list = ['vx', 'vy', 'vz', 'yaw'] and iterate through with this dictionary to easily stack data

        # input dictionary less likely to be used because one will not likely do control without a type of acutation. Could be interesting though
        self._input_dict = {
            'Thrust': [0, 'force'],
            'taux': [1, 'torque'],
            'tauy': [2, 'torque'],
            'tauz': [3, 'torque']
        }
        self.x_dim =12
        self.u_dim = 4
        self.dt = dt
        self.x_noise = x_noise

        # Setup the state indices
        self.idx_xyz = [0, 1, 2]
        self.idx_xyz_dot = [3, 4, 5]
        self.idx_ptp = [6, 7, 8]
        self.idx_ptp_dot = [9, 10, 11]

        # Setup the parameters
        self.m = m
        self.L = L
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = -9.81

        # Define equilibrium input for quadrotor around hover
        self.u_e = np.array([m*self.g, 0, 0, 0])               #This is not the case for PWM inputs
        # Four PWM inputs around hover, extracted from mean of clean_hover_data.csv
        # self.u_e = np.array([42646, 40844, 47351, 40116])

        # Hover control matrices
        self._hover_mats = [np.array([1, 0, 0, 0]),      # z
                            np.array([0, 1, 0, 0]),   # pitch
                            np.array([0, 0, 1, 0])]   # roll
        #variable to keep track of most recent policy test data
        self.recentDataset = None

    def pqr2rpy(self, x0, pqr):
        rotn_matrix = np.array([[1., math.sin(x0[0]) * math.tan(x0[1]), math.cos(x0[0]) * math.tan(x0[1])],
                                [0., math.cos(
                                    x0[0]),                   -math.sin(x0[0])],
                                [0., math.sin(x0[0]) / math.cos(x0[1]), math.cos(x0[0]) / math.cos(x0[1])]])
        return rotn_matrix.dot(pqr)

    def pwm_thrust_torque(self, PWM):
        # Takes in the a 4 dimensional PWM vector and returns a vector of
        # [Thrust, Taux, Tauy, Tauz] which is used for simulating rigid body dynam
        # Sources of the fit: https://wiki.bitcraze.io/misc:investigations:thrust,
        #   http://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8905295&fileOId=8905299

        # The quadrotor is 92x92x29 mm (motor to motor, square along with the built in prongs). The the distance from the centerline,

        # Thrust T = .35*d + .26*d^2 kg m/s^2 (d = PWM/65535 - normalized PWM)
        # T = (.409e-3*pwm^2 + 140.5e-3*pwm - .099)*9.81/1000 (pwm in 0,255)

        def pwm_to_thrust(PWM):
            # returns thrust from PWM
            pwm_n = PWM/65535.0
            thrust = .35*pwm_n + .26*pwm_n**2
            return thrust

        pwm_n = np.sum(PWM)/(4*65535.0)

        l = 35.527e-3   # length to motors / axis of rotation for xy
        lz = 46         # axis for tauz
        c = .05         # coupling coefficient for yaw torque

        # Torques are slightly more tricky
        # x = m2+m3-m1-m4
        # y =m1+m2-m3-m4

        # Estiamtes forces
        m1 = pwm_to_thrust(PWM[0])
        m2 = pwm_to_thrust(PWM[1])
        m3 = pwm_to_thrust(PWM[2])
        m4 = pwm_to_thrust(PWM[3])

        Thrust = pwm_to_thrust(np.sum(PWM)/(4*65535.0))
        taux = l*(m2+m3-m4-m1)
        tauy = l*(m1+m2-m3-m4)
        tauz = -lz*c*(m1+m3-m2-m4)

        return np.array([Thrust, taux, tauy, tauz])

    def simulate(self, x, PWM, t=None, addNoise = False):
        # Input structure:
        # u1 = thrust
        # u2 = torque-wx
        # u3 = torque-wy
        # u4 = torque-wz
        u = self.pwm_thrust_torque(PWM)
        dt = self.dt
        u0 = u
        x0 = x
        idx_xyz = self.idx_xyz
        idx_xyz_dot = self.idx_xyz_dot
        idx_ptp = self.idx_ptp
        idx_ptp_dot = self.idx_ptp_dot

        m = self.m
        L = self.L
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        g = self.g

        Tx = np.array([Iyy / Ixx - Izz / Ixx, L / Ixx])
        Ty = np.array([Izz / Iyy - Ixx / Iyy, L / Iyy])
        Tz = np.array([Ixx / Izz - Iyy / Izz, 1. / Izz])

        # # Add noise to input
        # u_noise_vec = np.random.normal(
        #     loc=0, scale=self.u_noise, size=(self.u_dim))
        # u = u+u_noise_vec

        # Array containing the forces
        Fxyz = np.zeros(3)
        Fxyz[0] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.cos(
            x0[idx_ptp[2]]) + math.sin(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[1] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.sin(
            x0[idx_ptp[2]]) - math.sin(x0[idx_ptp[0]]) * math.cos(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[2] = g + 1 * (math.cos(x0[idx_ptp[0]]) *
                           math.cos(x0[idx_ptp[1]])) * u0[0] / m
        # Compute the torques
        t0 = np.array([x0[idx_ptp_dot[1]] * x0[idx_ptp_dot[2]], u0[1]])
        t1 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[2]], u0[2]])
        t2 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[1]], u0[3]])
        Txyz = np.array([Tx.dot(t0), Ty.dot(t1), Tz.dot(t2)])

        x1 = np.zeros(12)
        x1[idx_xyz_dot] = x0[idx_xyz_dot] + dt * Fxyz
        x1[idx_ptp_dot] = x0[idx_ptp_dot] + dt * Txyz
        x1[idx_xyz] = x0[idx_xyz] + dt * x0[idx_xyz_dot]
        x1[idx_ptp] = x0[idx_ptp] + dt * \
            self.pqr2rpy(x0[idx_ptp], x0[idx_ptp_dot])

        # Add noise component
        if addNoise:
            x_noise_vec = np.random.normal(loc=0, scale=self.x_noise, size=(self.x_dim))
        # makes states less than 1e-12 = 0
            x1[x1 < 1e-12] = 0
            return x1+x_noise_vec
        else:
            x1[x1 < 1e-12]
            return x1

    def test_policy(self, initStates, policy, iterations, visuals, maxFrames = 500, addNoise = False, maxCond = 50):
        '''Set up other ways of gauging fitness instead of TIME'''
        if visuals:
            visualizer = GUI(10)
        time = []
        firstAddToRecent = True
        i = 0
        avgError = 0
        pitchidx = self._state_dict['pitch'][0]
        rollidx = self._state_dict['roll'][0]
        yawidx = self._state_dict['yaw'][0]
        while i < iterations:
            index = np.random.randint(0, np.shape(initStates)[0])
            state = (initStates[index, :])
            if abs(state[pitchidx])> math.radians(5) or abs(state[rollidx])> math.radians(5):
                continue #find new input
            #print("")
            #print("Beginning flight number ", i + 1)
            frames = 0
            failed = False
            X0_temp = None
            U_temp = None
            X1_temp = None
            first = True
            error = 0
            max = math.radians(maxCond)
            while not failed and frames < maxFrames:
                policy.update([math.degrees(state[pitchidx]), math.degrees(state[rollidx]), math.degrees(state[yawidx])])
                PWM = policy.chooseAction()
                newState = self.simulate(state, PWM.numpy(), addNoise = addNoise)
                if visuals:
                    pos = newState[0:3]
                    euler = np.array([newState[pitchidx], newState[rollidx], newState[yawidx]])
                    visualizer.update(self.dt, pos, euler)
                if first:
                    X0_temp = state.reshape(1,-1)
                    U_temp = PWM.numpy().reshape(1,-1)
                    X1_temp = newState.reshape(1,-1)
                    first = False
                else:
                    X0_temp = np.vstack((X0_temp, state))
                    U_temp = np.vstack((U_temp, PWM.numpy()))
                    X1_temp = np.vstack((X1_temp, newState))
                state = newState
                frames += 1
                error += newState[pitchidx] + newState[rollidx] + newState[yawidx]
                if abs(state[pitchidx]) > max or abs(state[rollidx]) > max or abs(state[yawidx]) > max:
                    failed = True
            error += (maxFrames - frames) * max * 3
            time += [frames]
            rewards = [[frames] for k in range(frames)]
            rewards = (np.array(rewards))
            flightnum = [[i] for j in range(frames)]
            flightnum = np.array(flightnum)
            result = np.hstack((X0_temp, U_temp, X1_temp, rewards, flightnum))
            i += 1
            if firstAddToRecent:
                self.recentDataSet = result
                firstAddToRecent = False
            else:
                self.recentDataSet = np.vstack((self.recentDataSet, result))
            avgError += (1/iterations) * error

        return sum(time), sum(time)/iterations, np.var(time), avgError

    def get_recent_data(self):
        return self.recentDataSet

    def stackDataset(self, dataset, dimX, dimU, dimdX, stack, rewardTime = False, minimumFrames = 200):
        '''Takes in a dataset returned by get_recent_data and stacks it in preparation for passing into the ensemble neural network
            Since this includes the position, each state would be 12 * numStacks length and action will be 4 * numStacks length. Resulting state should simply be 12.'''
        rows = np.shape(dataset)[0] - stack
        columns = dimX * stack + dimU * stack + dimdX
        result = np.zeros((1, columns)) #dummy
        X = dataset[:,:dimX]
        U = dataset[:,dimX:dimX+dimU]
        dX = dataset[:,dimX+dimU:-2] #don't want to take into account the REWARD just yet
        rewards = dataset[:, -2:-1]
        flightnum = dataset[:, -1:]
        assert np.shape(X)[1] == dimX and np.shape(dX)[1] == dimdX
        for i in range(rows):
            if flightnum[i, 0] != flightnum[i + (stack - 1), 0]:
                continue #this means the two concatenated datapoints are not from the same rollout
            currX = X[i, :]
            currU = U[i, :]
            currdX = dX[i + (stack - 1), :]
            for j in range(stack - 1):
                currX = np.hstack((currX, X[i + j + 1, :]))
                currU = np.hstack((currU, U[i + j + 1, :]))
            result = np.vstack((result, np.hstack((currX, currU, currdX))))
        extra = np.zeros((1, columns))
        if rewardTime: #this implementation should technically take constant time
            index = 0
            offsets = []
            size = []
            length = np.shape(rewards)[1]
            slices = []
            while index < length: #making offsets and size lists
                offsets += [index]
                size  += [rewards[index, 0]]
                index += rewards[index, 0]
            for i, s in enumerate(size):
                if s < minimumFrames:
                    continue
                slices += [[int(offsets[i]), int(offsets[i] + size[i])]]
            for slice in slices: #recursive step
                extraData = np.hstack(self.stackDataset(dataset[slice[0]:slice[1], :], dimX, dimU, dimdX, stack, False, minimumFrames))
                extra = np.vstack((extra, extraData))
        X = result[1:, :dimX * stack]
        U = result[1:, dimX * stack: dimX*stack + dimU * stack]
        dX = result[1:, dimX*stack + dimU * stack:]
        if not rewardTime:
            return (X,U,dX), None
        exX = extra[1:, :dimX * stack]
        exU = extra[1:, dimX * stack: dimX*stack + dimU * stack]
        exdX = extra[1:, dimX*stack + dimU * stack:]
        return (X, U, dX), (exX, exU, exdX)
