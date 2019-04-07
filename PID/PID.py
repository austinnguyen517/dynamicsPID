import numpy as np

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
