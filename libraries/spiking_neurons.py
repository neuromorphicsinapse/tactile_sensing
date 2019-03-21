'''
#-------------------------------------------------------------------------------
# NATIONAL UNIVERSITY OF SINGAPORE - NUS
# SINGAPORE INSTITUTE FOR NEUROTECHNOLOGY - SINAPSE
# Neuromorphic Engineering and Robotics Group - NER
#-------------------------------------------------------------------------------
# Description: Library for handling spiking neurons
#-------------------------------------------------------------------------------
'''
#-------------------------------------------------------------------------------
# LIBRARIES
#-------------------------------------------------------------------------------
import numpy as np #numpy
import matplotlib.pyplot as plt #plotting
import scipy.io as sio #input-output for handling files
import scipy.signal as sig #signal processing
import scipy.stats as stat #statistics
from collections import Counter #counting occurrences
#multiprocessing
from multiprocessing import Pool
from copy import copy
#-------------------------------------------------------------------------------
# DEFINES THE SINGLE NEURON MODELS
#-------------------------------------------------------------------------------
class model():
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
    #defines an izhikevich neuron model
    #izhikevich
    #v' = 0.04v^2 + 5v + 140 - u + I
    #u' = a(bv - u)
    #if v >= 30mV, v=c,u=u0+d where u0 is the initial condition of u
    #v represents the membrane potential
    #u represents a membrane recovery variable
    #the parameter a describes the time scale of the recovery variable u.
    #Smaller values result in slow recovery. A typical value is a=0.02
    #the parameter b describes the sensitivity of the recovery variable u to
    #the subthreshold fluctuations of the membrane potential v. A typical value
    #is b=0.2
    #the parameter c describes the after-spike reset value of the membrane
    #potential v caused by the fast high-threshold K+ conductances. A typical
    #value is c=-65mV.
    #the parameter d describes after-spike reset of the recovery variable u
    #caused by slow high-threshold Na+ and K+ conductances. A typical value is
    #d=2.
#-------------------------------------------------------------------------------
    class izhikevich():
        def __init__(self,A=0.04,B=5,C=140,Cm=1,a=0.02,b=0.2,c=-65,d=2,name='default'):
            #parameters that define the neuron model
            self.A = A
            self.B = B
            self.C = C
            self.Cm = Cm
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.type = 'izhikevich' #identify the neuron model
            self.name = name #name to help into identifying the neuron itself
            self.uv = []

            #Initial conditions
            self.vm = c #resting potential
            self.u0 = (self.b * self.vm) #global initial condition of recovery variable
            self.u = self.u0 #recovery variable

        #integrate the model over one step
        def integrate(self,input_current,dt):
            v_old = self.vm
            u_old = self.u
            #find the next value of the membrane voltage
            self.vm = v_old + dt*((self.A*v_old*v_old + self.B*v_old - u_old + self.C + (input_current/self.Cm)));
            #find the next value of the recovery variable
            self.u = u_old + dt*self.a*((self.b * v_old) - u_old);
            #spike event
            #if a spike is generated
            self.uv.append(self.u)
            if self.vm > 30:
                self.vm = self.c #reset membrane voltage
                self.u = self.u + self.d #reset recovery variable
                self.uv.append(self.u)
                # print(self.u, self.d)
                return [True,self.vm,31] #returns true
            else: #no spikes
                return [False,self.vm] #returns false
#-------------------------------------------------------------------------------
# DEFINES THE SIMULATION CONTROL OBJECT
#-------------------------------------------------------------------------------
class simulation():
    def __init__(self,dt=1,t0=0,tf=1000,I=[np.ones(1000)],neurons=None):
        self.dt = dt #time-step of the simulation (ms)
        self.t0 = t0 #initial time (ms)
        self.tf = tf #final time (ms)
        #input current
        #should be a matrix of NxM
        #where
        # N: number of neurons -> points which neuron should receive the current
        # M: number of samples given dt, t0 and tf
        self.I = I
        #time vector generated from t0, tf, and dt
        self.timev = np.arange(self.t0,self.tf,self.dt)
        #vector containing the neurons to be simulated
        self.neurons = neurons
        #matrix containing time vectors for each neuron
        self.timen = [[] for i in range(len(neurons))]
        #matrix containing the spike times for each neuron
        self.spikes = [[] for i in range(len(neurons))]
        #matrix containing the membrane voltage over time for each neuron
        self.vneurons = [[] for i in range(len(neurons))]

    def run(self):
        '''
        optimal integration of izhikevich model based on his matlab code.
        Warning: all the neurons should be of izhikevich type
        '''
        numNeurons = len(self.neurons)
        #model parameters
        #create numpy arrays according to the parameters of each neuron
        A = np.array([x.A for x in self.neurons],dtype='float64')
        B = np.array([x.B for x in self.neurons],dtype='float64')
        C = np.array([x.C for x in self.neurons],dtype='float64')
        a = np.array([x.a for x in self.neurons],dtype='float64')
        b = np.array([x.b for x in self.neurons],dtype='float64')
        c = np.array([x.c for x in self.neurons],dtype='float64')
        d = np.array([x.d for x in self.neurons],dtype='float64')
        #array containing the membrane voltage of each neuron
        v = np.ones(numNeurons) * c
        #array containing the membrane recovery for each neuron
        u = v * b
        #run the simulation
        #optimal integration
        for k in range(self.tf):
            #check whether there are spikes
            fired = np.where(v >= 30)[0]
            #if a spike has been triggered, reset the model
            #and save the spikes
            if(len(fired) > 0):
                #reset the membrane potential
                v[fired] = c[fired]
                #reset membrane recovery
                u[fired] += d[fired]
                #save the time of spike for the neurons that fired
                [self.spikes[idx].append(self.timev[k]) for idx in fired]
                #save the value of spike for the neurons that fired
                [self.vneurons[idx].append(31) for idx in fired]
                #save the time where the spike occurred for proper plotting
                [self.timen[idx].append(self.timev[k]) for idx in fired]

            #save the membrane voltage
            [self.vneurons[i].append(v[i]) for i in range(len(self.vneurons))]
            #save the time step
            [x.append(self.timev[k]) for x in self.timen]

            #take the current input
            im = [self.I[i][k] for i in range(numNeurons)]
            #integrate two times with dt/2 for numerical stability
            v += (self.dt/2) * (A*np.power(v,2) + B*v + C - u + im) #membrane
            v += (self.dt/2) * (A*np.power(v,2) + B*v + C - u + im) #voltage
            u += self.dt * (a*(b*v-u)) #membrane recovery
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# DEFINES METHODS FOR ANALYZING SPIKE TRAINS
#-------------------------------------------------------------------------------
class analysis():
    def raster(simulObj):
        class rasters():
            def __init__(self,xvals,yvals,yvalslbl,ylabels,spike,numNeurons):
                self.numNeurons = numNeurons
                self.xvals = xvals
                self.yvals = yvals
                self.yvalslbl = yvalslbl
                self.ylabels = ylabels
                self.spike = spike

        #get the number of neurons
        numNeurons = len(simulObj.neurons)
        #get the names of the neurons
        neuronNames = [v.name for v in simulObj.neurons]
        #get the maximum firing time
        #variable for handling absence of spikes
        maxTimes = [np.max(x) for x in simulObj.spikes if len(x) > 0]
        if len(maxTimes) > 0:
            maxSpikeTime = np.max(maxTimes)
        #y-axis values
        yvalues = []
        #x-axis values
        xvalues = []
        #determines whether neuron fired or not
        spike = []

        if len(maxTimes) > 0:
            for k in range(numNeurons):
                spk = simulObj.spikes[k]
                if len(spk) > 0:
                    spike.append(True) #neuron spiked
                    yvalues.append(np.ones(len(spk))*(k))
                else:
                    spike.append(False)
                    spk = [-100,simulObj.tf+100] #neuron didn't spike
                    yvalues.append(np.ones(len(spk))*(4.25-((k+1)*0.025)))
                xvalues.append(spk)

            yv = [np.max(x) for x in yvalues] #the values in y-axis

            return rasters(xvalues,yvalues,yv,neuronNames,spike,numNeurons)
        else:
            return False
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    #create the neurons
    n = [model.izhikevich() for k in range(6)]

    #create the input currents
    I1 = np.ones(1000) * 8
    I2 = np.ones(1000) * 10
    I3 = np.ones(1000) * 11
    I4 = np.ones(1000) * 20
    I5 = np.ones(1000) * 16
    I6 = np.ones(1000) * 19

    #prepare the simulation
    s = simulation(I=[I1,I2,I3,I4,I5,I6],neurons=n)
    #run the simulation
    s.run()
