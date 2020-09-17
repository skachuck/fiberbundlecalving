"""
util.py

Some shared utility functions.
"""

#### SOME THRESHOLD DISTRIBUTION GENERATORS ####
def retconst(const):
    """Generator for the constant const.
    """
    def f(s=None):
        if s is None:
            return const
        else:
            return np.ones(s)*const
    return f

def uni_dist(lo,hi):
    """Generator for uniform distribution between lo and hi.
    """
    def f(s=None):
        if s is None:
            return lo + (hi-lo)*np.random.rand()
        elif len(np.atleast_1d(s))==1: 
            return lo + (hi-lo)*np.random.rand(s)
        else:
            return lo + (hi-lo)*np.random.rand(*s)
    return f

def weib_dist(l,a):
    """Weibull distribution with factor l and shape a
    """
    def f(s=None):
        if s is None:
            return np.random.weibull(a)*l
        elif len(np.atleast_1d(s))==1: 
            return np.random.weibull(a)*l
        else:
            return np.random.weibull(a)*l 
    return f

def norm_dist(mu,sig):
    """Normal distribution, positive only
    """
    def f(s=None):
        if s is None:
            return np.abs((np.random.randn()+mu)*sig)
        elif len(np.atleast_1d(s))==1: 
            return np.abs((np.random.randn(s)+mu)*sig)
        else:
            return np.abs((np.random.randn(*s)+mu)*sig)
    return f

def strict_dist(lo=0,hi=1):
    """Generator for strictly increasing thresholds lo to hi, excluding lo.
    """
    def f(s=None):
        assert s is not None, 'No way to set one strict threshold'
        if len(np.atleast_1d(s))==1:
            return np.linspace(lo,hi,s+1)[1:]
        else:
            return np.repeat(np.linspace(lo,hi,s[1]+1)[1:][None,:],s[0],0)

#### SOME STATE VARIABLE FUNCTIONS ####
def strain_thresh(x, ssaModel):
    """
    Compute strain from grounding line.
    """
    # Interpolate to locations x
    U = ssaModel.U(x)
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
