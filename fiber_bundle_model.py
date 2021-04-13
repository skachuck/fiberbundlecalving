"""
fiber_bundle_model.py
Author: Samuel B Kachuck
Date: Feb, 2020

Methods and classes for investigating the 1D Fiber Bundle Model.

average_strain_exp and average_stress_exp compute the strains and stresses
experienced by a 1D FBM with equal load sharing (the mean field case) for a
collection of fiber strenghts supplied or a number of uniformly strengths.
The average_stress_exp additionall returns a list of avalanche sizes, as
avalanches are inevitable in an equal load sharing stress experiment.

The classes FiberBundle1D and FiberBundle1DLLS 
"""


import numpy as np
import matplotlib.pyplot as plt

def average_strain_exp(xc=None,N=100):
    """Execute an average strain experiment with critcal displacements xc.
    
    Computes the stress before and after each fiber breaks.
    Returns strains and stresses of the system as fibers break (between breaks,
    stress-strain relationship is linear).
    """
    # Initialize the critical displacements
    if xc is None:    
        xc = np.random.rand(N)
    else:
        N = len(xc)
    # Sort by fiber strength
    xc = np.sort(xc)

    # Number of unbroken fibers at each point
    Nc = np.arange(1,N+1)[::-1]

    xc = np.repeat(xc,2)
    Nc = np.r_[np.repeat(Nc,2)[1:],0]

    return xc, Nc*xc

def average_stress_exp(xc=None,N=100):
    """Executes an average stress experiment with critical displacements xc.

    Increasing the stress to break each fiber sequentially, with the
    possibility that the redistribution of stress when a fiber breaks may cause
    another to break (called an avalanche). Returns stresses, strains, and
    avalance sizes of each stress increase.
    """
    # Initialize the critical displacements
    if xc is None:    
        xc = np.random.rand(N)
    else:
        N = len(xc)
    # Sort by fiber strength
    xc = np.sort(xc)

    xs = [0]
    Fs = [0]
    avs = []

    i = 0

    # Loop through all the fibers, need to save force and displacement
    # everytime of them changes.
    while i<N:
        # Increase the force to start next burst
        F = xc[i]*(N-i)
        x = xc[i]
        # Save force and displacement
        Fs.append(F)
        xs.append(x)
        # Fiber i breaks, and the system extends
        x = F/(N-i-1)
        # Save force and displacement
        Fs.append(F)
        xs.append(x)

        # Start a new avalanche burst. The fibers are arranged in order of
        # strength, so we need only loop forward, checking if the current
        # extension of the system (Equal Load Sharing) breaks the fiber. If so,
        # break the fiber, extend and repeat. If not, the burst is done and
        # more force is needed.
        # Restart the avalanche count
        av = 1
        # Check next fiber (breaks if x>xc[i])
        i += 1
        while i<N and x>xc[i]:
            av += 1
            # Fiber i breaks, and the system extends
            x = F/(N-i-1)
            # Save force and displacement 
            Fs.append(F)
            xs.append(x)

            i += 1
        avs.append(av)

    return np.array(xs), np.array(Fs), np.array(avs)[:-1]
    #Adding print statements to see force and displacement
    print("Displacement",xs)
    print("Force",Fs)

class FiberBundle1d(object):
    def __init__(self, N, dist='uni', mode='equal'):
        self.N = N

        # Initialize the critical extensions
        if dist == 'uni':
            self.xc = np.random.rand(N)
        if dist == 'strict':
            self.xc = np.linspace(1,0,N,endpoint=False)[::-1]

        # Initialize the total force on the system
        self.F = 0

        # Initialize the stress, displacment and broken bool per fiber.
        # Proportion of the stress borne by each fiber, starts even.
        # The local stress on each fiber is ps*F
        self.ps = np.ones(N)/N
        # Displacements by fiber
        self.xs = np.zeros(N)
        # Flag for intact fiber
        self.ss = np.ones(N, dtype=bool)

        self.mode = mode

        self.dispobs = DisplacementObserver(self)
        self.avobs = AvalancheObserver(self)
        self.fibobs = FiberObserver(self)
        self.obslist = [self.dispobs, self.avobs, self.fibobs]

    @property
    def xhist(self):
        return self.dispobs.xsave
    @property
    def fhist(self):
        return self.dispobs.fsave
    @property
    def avsize(self):
        return self.avobs.size_hist

    def force_test(self):
        # Increase the force until all fibers broken
        while any(self.ss):
            # Find the weakest intact fiber
            imin = np.argwhere(self.xc == self.xc[self.ss].min())[0][0]
            # Pull to its threshhold
            self.F = self.xc[imin]/self.ps[imin]
            self.xs = self.ps*self.F

            for obs in self.obslist: obs.notify_stretch(self)
            # Break it and redistribute force
            self.break_fiber(imin)

            # Propagate the burst
            while any(self.ss) and any(self.xc[self.ss]<self.xs[self.ss]):
               breakable = np.logical_and(self.xc<self.xs, self.ss)
               # Find the weakest intact fiber over its threshhold
               imin = np.argwhere(self.xc == self.xc[breakable].min())[0][0]
               # Break it and redistribute force
               self.break_fiber(imin)               

            for obs in self.obslist: obs.notify_avalanche_end(self)

        for obs in self.obslist: obs.notify_simulation_end(self)

    def break_fiber(self, i):
        # Break fiber i and redistribute stress 
        if self.mode == 'equal':
            # Spread the force over all remaining fibers
            self.ps[self.ss] = 1./(self.ss.sum()-1.)
        elif self.mode == 'nearest':
            # Spread the force on i to the nearest neighbors
            self.ps[(i-1)%self.N] += self.ps[i]/2
            self.ps[(i+1)%self.N] += self.ps[i]/2

        # Break the fiber and zero its force
        self.ps[i] = 0
        self.ss[i] = False 
        # extend the fibers under the new local forces
        self.xs = self.ps*self.F

        for obs in self.obslist: obs.notify_break(self, i)

    def plot_av_hist(self, nbins=10):
        bins = np.logspace(0, np.log10(self.N)/2, nbins)
        plt.hist(self.avsize[:-1], log=True, 
            bins=bins)
        plt.xscale('log')
        return plt.gca()


class FiberBundle1dLLS(object):
    def __init__(self, N, dist='uni', mode='equal'):
        self.N = N

        # Initialize the critical extensions
        if dist == 'uni':
            self.xc = np.random.rand(N)
        if dist == 'strict':
            self.xc = np.linspace(1,0,N,endpoint=False)[::-1]

        # Initialize the total force on the system
        self.F = 0

        # Initialize the stress, displacment and broken bool per fiber.
        # Proportion of the stress borne by each fiber, starts even.
        # The local stress on each fiber is ps*F
        self.ps = np.ones(N)/N
        # Displacements by fiber
        self.xs = np.zeros(N)
        # Flag for intact fiber
        self.ss = np.ones(N, dtype=bool)

        # number of failed fibers to the left and right of each fiber
        self.nl = np.zeros(N)
        self.nr = np.zeros(N)

        self.mode = mode

        self.holeobs = HoleSizeObserver(self)
        self.avobs = AvalancheObserver(self)
        self.fibobs = FiberObserver(self)
        self.obslist = [self.holeobs, self.avobs, self.fibobs]

    @property
    def holebins(self):
        return self.holeobs.bins
    @property
    def holehist(self):
        return self.holeobs.hole_hist
    @property
    def avsize(self):
        return self.avobs.size_hist

    def force_test(self, maxstep=None):
        # Increase the force quasistatically (i.e., by reducing the load back
        # to zero and reapplying each time a fiber breaks) until all fibers
        # broken.

        if maxstep is None: maxstep = self.N
        step = 0
        while any(self.ss) and step<maxstep:
            # Find the weakest intact fiber
            imin = np.argmax(self.ps/self.xc)
            # Pull to its threshhold
            self.F = self.xc[imin]/self.ps[imin]
            self.xs = self.ps*self.F

            for obs in self.obslist: obs.notify_stretch(self)
            # Break it and redistribute force
            self.break_fiber(imin)
            
            step += 1

            # Propagate the burst
            while any(self.ss) and any(self.xc[self.ss]<self.xs[self.ss]) and step<maxstep:
               # Find the next intact fiber over its threshhold
               imin = np.argmax(self.ps/self.xc)
               # Break it and redistribute force
               self.break_fiber(imin)        

               step += 1

            for obs in self.obslist: obs.notify_avalanche_end(self)

        for obs in self.obslist: obs.notify_simulation_end(self)

    def break_fiber(self, i):
        # Spread the force on i to the nearest neighbors
        # Find nearest intact fiber to the left

        # Transfer the holes on the right of the breaking fiber to the new left
        # side of the hole, plus the hole from the breaking fiber.
        unbroken = np.atleast_1d(np.argwhere(self.ss).squeeze())
        # Start by finding the nearest intact fiber to the left of the breaking
        # fiber.
        ileft = unbroken[(np.argwhere(unbroken == i)-1)%len(unbroken)]
        iright = unbroken[(np.argwhere(unbroken == i)+1)%len(unbroken)]
        # Now distribute the holes to the right.
        self.nr[ileft] += self.nr[i]+1
        self.nr[i] = -1
        # Do the same for the right side of the new hole.
        self.nl[iright] += self.nl[i]+1
        self.nl[i] = -1

        # Break the fiber and zero its force
        self.ps = (1 + 0.5*(self.nl+self.nr))/self.N
        self.ss[i] = False 
        # extend the fibers under the new local forces
        self.xs = self.ps*self.F

        for obs in self.obslist: obs.notify_break(self, i)


class DisplacementObserver(object):
    def __init__(self, fiberbundle):
        self.xsave = [0]
        self.fsave = [0]

    def notify_stretch(self, fiberbundle):
        self.xsave.append(fiberbundle.F/fiberbundle.N)
        self.fsave.append(fiberbundle.F)

    def notify_break(self, fiberbundle, i):
        self.xsave.append(fiberbundle.F/fiberbundle.N)
        self.fsave.append(fiberbundle.F)

    def notify_avalanche_end(self, fiberbundle):
        pass

    def notify_simulation_end(self, fiberbundle):
        self.xsave = np.array(self.xsave)
        self.fsave = np.array(self.fsave)

class AvalancheObserver(object):
    def __init__(self, fiberbundle):
        self.current_size = 0
        self.size_hist = []

    def notify_stretch(self, fiberbundle):
        pass

    def notify_break(self, fiberbundle, i):
        self.current_size += 1

    def notify_avalanche_end(self, fiberbundle):
        self.size_hist.append(self.current_size)
        self.current_size = 0
        
    def notify_simulation_end(self, fiberbundle):
        self.size_hist = np.array(self.size_hist)

class FiberObserver(object):
    def __init__(self, fiberbundle):
        self.N = fiberbundle.N
        self.fig, self.ax = plt.subplots(1,1)
        ys = np.linspace(0,1,self.N)
        self.fibers = []
        for y in ys:
            self.fibers.append(self.ax.plot([0,1],[y,y],c='k',ls='-')[0])
        self.Ftext = self.ax.text(0,1, 'F={}'.format(fiberbundle.F))
        plt.pause(0.01)

    def notify_stretch(self, fiberbundle):
        pass

    def notify_break(self, fiberbundle, i):
        self.fibers[i].set_color((0,0,0,0.2))
        plt.pause(0.01)

    def notify_avalanche_end(self, fiberbundle):
        self.Ftext.set_text('F={}'.format(fiberbundle.F))
        plt.pause(0.5)

    def notify_simulation_end(self, fiberbundle):
        pass

class HoleSizeObserver(object):
    def __init__(self, fiberbundle):
        self.hole_hist = []
        self.nbins = int(fiberbundle.N/np.log(fiberbundle.N))
        self.bins = np.linspace(1,self.nbins,self.nbins)

    def notify_stretch(self, fiberbundle):
        pass

    def notify_break(self, fiberbundle, i):
        hist = np.histogram(fiberbundle.nr, self.bins)[0]
        self.hole_hist.append(hist)

    def notify_avalanche_end(self, fiberbundle):
        pass

    def notify_simulation_end(self, fiberbundle):
        self.hole_hist = np.array(self.hole_hist)

            


if __name__ == '__main__':
    N = 10
    xc = np.sort(np.random.rand(N))
    xc = np.linspace(1,0,N,endpoint=False)[::-1]
#    xc = np.sort(np.random.weibull(4,N))
    xstress, Fstress, avstress = average_stress_exp(xc)
    xstrain, Fstrain = average_strain_exp(xc)

    x = np.linspace(0,1,100)

    #fig, axs = plt.subplots(1,2)

    #axs[0].plot(xstress, Fstress)
    #axs[0].plot(xstrain, Fstrain)
    #mean = N*x*(1-x)
    #axs[0].plot(x, mean)
    #std = np.sqrt(N*x**3*(1-x))
#    plt.fill_between(x, mean-std, mean+std)
    #axs[0].set_xlim([0,1])

    #axs[1].hist(avstress)
    #plt.show()


