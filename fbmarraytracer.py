"""
fbmtracer.py
Author: Samuel Kachuck
Date: Sep 1, 2020

Provides the tracer particle class that contains fiber bundles for calving the
ssa1d class, along with a collection of random strength distributions and state
variable functions.
"""
from __future__ import division
import numpy as np
from util import *

class FBMFullTracer(object):
    def __init__(self, Lx, N0, Nf=10, dist=None, compState=None, stepState=None, xsep=1e16, **kwargs):
        """
        A container for tracer particles in a 1D fenics ice dynamics model.

        The tracers advect with the fluid velocity, but do not affect the flow
        until the fiber bundle models they carry break, at which point calving
        is triggered in the ice flow model.
        There are two kinds of state variables: those that are integrated
        through the simulation, and those that only depend on the instantaneous
        configuration of the ice. For the former, provide a function,
        stepState, that is used to intergate the state variable when the
        particles advect. For the latter, provide a function that computes the
        state variable when checking for calving. Only one may be specified.

        Parameters
        ----------
        Lx : the length of the system
        N0 : the initial number of particles (spread evenly through the ice
            sheet)
        dist : a function that returns a random threshold
        compState(x, ssaModel) : a function that computes the path-independent state 
            variable from the ssaModel at locations x, default None. If both
            compState and stepState are None, compState is defined as
            strain_thresh.
        stepState(x, ssaModel) : a function that computes a discrete
            time-derivative of the state variable, for integrating, default
            None
        xsep : the separation of particles when added to the system

        Data
        ----
        N : number of particles
        x : location of particles
        s : threshold of particles
        state : the state of each particle

        Methods
        -------
        advect_particles
        add_particle
        remove_particle
        check_calving
        calve
        """
        # Properties of tracers
        self.x = np.linspace(0,Lx,N0,endpoint=False)
        self.N = int(N0)
        self.xsep = xsep
        assert compState is None or stepState is None, 'Cannot specify both'
        self.compState = compState
        self.stepState = stepState
        if compState is None and stepState is None:
            self.compState = strain_thresh

        # Fiber bundles 
        self.dist = dist or strict_dist()
        # Number of fibers per bundle
        self.Nf = int(Nf)
        # Construct the fibers - random thresholds
        self.xcs = self.dist((self.N, self.Nf))
        # Force on each fiber bundle
        self.F = np.zeros(self.N)
        # Broken status of each fiber in each tracer
        # Extension of each bundle (ELS) is self.F/np.sum(self.ss, axis=1)
        self.ss = np.ones((self.N, self.Nf), dtype=bool)

        self.listform=False

    def _toList(self):
        if self.listform: return
        self.x = self.x.tolist()
        self.xcs = self.xcs.tolist()
        self.ss = self.ss.tolist()
        self.F = self.F.tolist()
        self.listform = True

    def _toArr(self):
        if not self.listform: return
        self.x = np.asarray(self.x) 
        self.xcs = np.asarray(self.xcs) 
        self.ss = np.asarray(self.ss)
        self.F = np.asarray(self.F)
        self.listform = False

    @property
    def fiber_extension(self):
        return self.F/np.sum(self.ss, axis=1)
    @property
    def exceeded_threshold(self):
        return self.xcs <= self.fiber_extension[:,None]

    @property
    def active_tracers(self):
        broken = np.where(np.logical_and(self.exceeded_threshold, self.ss))
        return np.unique(broken[0])

    @property
    def damage(self):
        return 1-np.sum(self.ss,axis=1)/self.Nf

    @property
    def data(self):
        return np.vstack([self.x, self.damage])

    def force(self, F):
        #print('Forcing with {}'.format(F))
        self.F = F
        # Find tracers with broken fibers 
        for i in self.active_tracers: 
            while any(self.ss[i]) and any(self.exceeded_threshold[i][self.ss[i]]):
                j = np.argwhere(self.exceeded_threshold[i]*self.ss[i])[0][0]
                #print('Breaking {} {}'.format(i,j))
                self.ss[i, j] = False
    def add_tracer(self, xp, state=0):
        """
        Introduce a new particle at location xp with threshold drawn from
        self.dist.
        """
        self._toList()
        for i in range(self.N):
            if self.x[i] > xp:
                index = i
                break
        self.x.insert(i,xp)
        self.xcs.insert(i, self.dist(self.Nf))
        self.F.insert(i,state)
        self.ss.insert(i,np.ones(self.Nf,dtype=bool))
        self.N += 1
        self._toArr()

    def remove_tracer(self, i):
        """
        Revmoce ith particle from the flow, used in calving.
        """
        self._toList()
        self.x.pop(i)
        self.xcs.pop(i)
        self.F.pop(i)
        self.ss.pop(i)
        self.N -= 1
        self._toArr()

    def advect_particles(self, ssaModel, dt):
        """
        Advect particles with the flow represented by vector coefficients
        Ufunc. Drop in a new particle if required.
        """
        self._toArr()
        # interpolate ice-velocities to particle positions
        U = np.array([ssaModel.U(x) for x in self.x])
        self.x += U*dt
        # If integrated state variable, increment it.
        if self.stepState is not None:
            dstate_dt = np.array([self.stepState(x, ssaModel) for x in self.x])
            F = self.F + dstate_dt*dt
        else:
            F = np.array([self.compState(x, ssaModel) for x in self.x])

        self.force(F)

        if self.x[0] > self.xsep:
            self.add_tracer(0)

    def check_calving(self):
        """Check if any FBMs have exceeded their thresholds.
        """
        if all(np.sum(self.ss, axis=1)):
            return False
        else:
            return True

    def calve(self, i=None, x=None):
        """Remove broken tracer and tracers connected to the front.
        """
        if x is not None:
            print('Calving from Lmax')
            i=np.argwhere(self.x > x)[0][0]
        elif i is None:
            assert self.check_calving(), 'No index given, none to break'
            i=np.argwhere(np.sum(self.ss,axis=1)==0)[0][0]
            #print(np.mean(self.xcs[i])) 
        xc = self.x[i]
        j = self.N - 1
        while j >= i:
            self.remove_tracer(j)
            j-=1
                
        return xc


