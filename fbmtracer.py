"""
fbmtracer.py
Author: Samuel Kachuck
Date: Sep 1, 2020

Provides the tracer particle class that contains fiber bundles for calving the
ssa1d class, along with a collection of random strength distributions and state
variable functions.
"""

class FBMTracer(object):
    def __init__(self, Lx, N0, dist=None, compState=None, stepState=None, xsep=1e16):
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
        self.x = list(np.linspace(0,Lx,N0,endpoint=False))
        self.N = N0
        self.dist = dist or retconst(1.)
        assert compState is None or stepState is None, 'Cannot specify both'
        self.compState = compState
        self.stepState = stepState
        if compState is None and stepState is None:
            self.compState = strain_thresh
        # Now create the initial thresholds
        self.s = [self.dist() for i in range(N0)]
        self.state = list(np.zeros(N0))
        self.xsep = xsep

        self.listform = True

    def advect_particles(self, ssaModel, dt):
        """
        Advect particles with the flow represented by vector coefficients
        Ufunc. Drop in a new particle if required.
        """
        self._toArr()
        # interpolate ice-velocities to particle positions
        U = np.array([ssaModel.U(x) for x in self.x])
        self.x += U*dt
        # If integrated state variable, interpolate to positions
        if self.stepState is not None:
            state_dt = np.array([self.stepState(x, ssaModel) for x in self.x])
            self.state += state_dt*dt

        if self.x[0] > self.xsep:
            self.add_particle(0)

    def add_particle(self, xp, state=0):
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
        self.s.insert(i,self.dist())
        self.state.insert(i,state)
        self.N += 1

    def _toList(self):
        if self.listform: return
        self.x = self.x.tolist()
        self.s = self.s.tolist()
        self.state = self.state.tolist()
        self.listform = True

    def _toArr(self):
        if not self.listform: return
        self.x = np.asarray(self.x)
        self.s = np.asarray(self.s)
        self.state = np.asarray(self.state)
        self.listform = False

    def remove_particle(self, i):
        """
        Revmoce ith particle from the flow, used in calving.
        """
        self._toList()
        self.x.pop(i)
        self.s.pop(i)
        self.state.pop(i)
        self.N -= 1

    def check_calving(self):
        """Check if any FBMs have exceeded their thresholds.
        """
        if self.compState is not None:
            state = self.compState(self.x, ssaModel)
        else:
            state = self.state
        
        broken = np.argwhere(self.s < state)
        if len(broken) > 0:
            # Temporarily save the lowest broken index, 
            # will be deleted after calving.
            self._minbroken = np.min(broken)
            return True
        else:
            return False

    def calve(self, i=None):
        """Remove broken FBMs and FBMs connected to the front.
        """
        self._toList()
        i = i or self._minbroken
        del self._minbroken
        xc = self.x[i]
        j = self.N - 1
        while j >= i:
            self.remove_particle(j)
            j-=1
        return xc


#### SOME THRESHOLD DISTRIBUTION GENERATORS ####
def retconst(const):
    """Returns the constant const
    """
    def f():
        return const
    return f

def uni_dist(lo,hi):
    """Uniform distribution between lo and hi
    """
    def f():
        return lo + (hi-lo)*np.random.rand()
    return f

def weib_dist(l,a):
    """Weibull distribution with factor l and shape a
    """
    def f():
        return np.random.weibull(a)*l
    return f

def norm_dist(mu,sig):
    def f():
        return (np.random.randn()+mu)*sig
    return f


#### SOME STATE VARIABLE FUNCTIONS ####
def strain_thresh(x, ssaModel):
    """
    Compute strain from grounding line.
    """
    # Interpolate to locations x
    U = np.array([ssaModel.U(i) for i in x])
    # Analytical form for strain in 1D
    strains = np.log(U/ssaModel.U0)
    return strains

def strain_ddt(x, ssaModel):
    strainFunc = project(grad(ssaModel.U)[0,0], ssaModel.Q_cg)
    return strainFunc(x)

def compute_stress(ssaModel):
    def epsilon(u):
        return 0.5*(nabla_grad(u)+nabla_grad(u).T)

    def sigma(u):
        return (2*nabla_grad(u))

    def epsII(u):
        eps = epsilon(u)
        epsII_UFL = sqrt(eps**2 + Constant(1e-16)**2)
        return epsII_UFL

    def eta(u):
        return Constant(ssaModel.B)*epsII(u)**(1./3-1.0)

    tau11 = project(eta(ssaModel.U)*grad(ssaModel.U)[0,0], ssaModel.Q_cg)

    return tau11

def strain_ddt_criticalstress(stress_c, stressFunc=compute_stress):
    def strain_ddt(x, ssaModel):
        stress = stressFunc(ssaModel)(x)
        strainFunc = project(grad(ssaModel.U)[0,0], ssaModel.Q_cg)
        return strainFunc(x)*(stress>stress_c)
    return strain_ddt
