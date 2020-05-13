"""
Class to solve SSA equations for 1D ice shelf, including advection, with a
stochastic model for calving that uses passive tracers.
"""

from fenics import *
import numpy as np
import pylab as plt

import numerics
reload(numerics)
from numerics import *

time_factor = 86400.0*365.24
parameters['allow_extrapolation'] = True

class ssa1D:
    def __init__(self,mesh,U0=8e-6,H0=800,order=1,beta=0.,m=3,
                advect_front=False, calve_flag=False,fbmkwargs={}):
        """
        Evolves a 1D ice shelf using the shallow shelf approximation.

        mesh = FEM mesh for model
        U0 = Inflow velocity at the grounding line (m/s)
        H0 = Ice thickness at the grounding line (m)
        beta = Drag coefficient
        m = Drag exponent
        order = Order of the function space (typically 1 or 2)

        advect_front = if True, front advects with the front velocity, 
            mesh evolves using Arbitrary Lagrange-Euler (ALE) method.
        calve_flag = if True,



        We use CG function spaces for velocity and DG function spaces
        for ice thickness and solve the continuity equation simultaneously with
        the SSA equation so that we can take implicit time steps
        """
        self.U0 = U0
        self.H0 = H0

        # Basal drag parameters
        self.beta = beta
        if isinstance(beta, (float, int)):
            self.beta = Constant(beta)
        self.m = m

        # Initialize constants
        self.rho_i = 920.0   # Density of ice (kg/m^3)
        self.rho_w = 1030.0  # Density of ambient ocean water (kg/m^3)
        self.rho_f = 1000.0  # Density of freshwater (kg/m^3)

        # Acceleration due to gravity
        self.g = 9.81 # acceleration due to gravity m/s^2
        self.n = 3.0 # flow law exponent
        self.B = 0.5e8 # rate factor of ice--can be adjusted
        # Constant used later

        self.C = (self.rho_i*self.g*(self.rho_w-self.rho_i)/4/self.B/self.rho_w)**(self.n)
        # Setup evenly spaced grid (This need to change for 2D solution)
        self.Nx=Nx; self.Lx=Lx
        #mesh = IntervalMesh(Nx, 0.0, Lx)
        self.mesh = Mesh(mesh)
        
        self.order = order

        # Initialize function spaces
        #self.init_function_space(self.mesh,order)

        self.Q_sys, self.Q, self.Q_vec, self.Q_cg, self.Q_cg_vec, self.ncell, self.hcell, self.v, self.v_vec, self.phi, self.phi_vec = self.init_function_space(self.mesh,order)

        #self.obslist = [UniformCalvingFrontObserver(self, 100000, 0.01)]
        self.obslist = [CalveObserver(self)]
        self.advect_front = advect_front
        self.calve_flag = calve_flag 

        if fbmkwargs:
            self.fbm = FBMTracer(**fbmkwargs)
        else:
            self.fbm = None

        self.t = 0


    def init_function_space(self,mesh,order):
        # vector CG element for velocity
        P2 = VectorElement("CG", mesh.ufl_cell(), order)
        # scalar DG element for ice thickness
        P1 = FiniteElement("DG", mesh.ufl_cell(), order)

        # Define mixed element and then mixed function space
        P_sys = MixedElement([P1, P2])
        Q_sys = FunctionSpace(mesh, P_sys)

        Q = Q_sys.sub(0).collapse()
        Q_vec = Q_sys.sub(1).collapse()
        Q_cg = FunctionSpace(mesh,'CG',1)
        Q_cg_vec = VectorFunctionSpace(mesh,'CG',1)

        # Facet normals and cell sizes
        ncell = FacetNormal(mesh)
        hcell = CellDiameter(mesh)

        # Test and trial functions
        # Trial functions for thickness and velocity
        v,v_vec = TrialFunctions(Q_sys)#self.v, self.phi
        # Test functions for thickness and velocity
        phi,phi_vec = TestFunctions(Q_sys)

        return Q_sys, Q, Q_vec, Q_cg, Q_cg_vec, ncell, hcell, v, v_vec, phi, phi_vec

    def steady_state(self,accum_rate):
        """
        Input: accum_rate = accumulation (m/s)
        """
        # Analytic solution for steady-state ice shelf with positive accumulation
        gdim = self.mesh.geometry().dim()
        Q= FunctionSpace(self.mesh, "CG", self.order)
        x = Q.tabulate_dof_coordinates().reshape((-1, gdim))
        n = self.n
        rho_i = self.rho_i
        rho_w = self.rho_w
        U0 = self.U0
        H0 = self.H0
        D = U0*(1-self.C/accum_rate*H0**(n+1))
        H = (self.C/accum_rate - (U0**(n+1)*(self.C/accum_rate*H0**(n+1)-1))/(accum_rate*x+H0*U0)**(n+1))**(-1/(n+1))
        U = (H0*U0+accum_rate*x)/H
        return x,H,U

    def init_shelf(self,accum_rate):
        """
        Initialize ice thickness and velocity to analytic solution
        """
        # Find analytic steady-state for given accumulation
        x,Hi,Ui = self.steady_state(accum_rate)

        # Define ice thickness and velocity on grid nodes
        H = Function(self.Q_cg);U = Function(self.Q_vec)

        # Set ice thickness and velocity based on analytic solution
        H.vector().set_local(Hi);U.vector().set_local(Ui)

        # Interpolate to function spaces
        H=interpolate(H,self.Q);

        return H,U

    def set_bcs(self):
        """
        Apply Dirichlet BCs at grounding line
        """
        inflow= CompiledSubDomain("near(x[0], 0.0)")
        bc1 = DirichletBC(self.Q_sys.sub(0), Constant(self.H0), inflow,"pointwise")
        bc2 = DirichletBC(self.Q_sys.sub(1), (Constant(self.U0),), inflow,"pointwise")
        bcs = [bc1,bc2]
        return bcs

    def velocity(self,H):
        """
        Variational form for the velocity
            Fully implicit now so we don't actually need to
            feed in the ice thickness

        """

        # Test and trial functions
        q,u  = TrialFunctions(self.Q_sys); w,v  = TestFunctions(self.Q_sys)

        #u  = self.v_vec; v  = self.phi_vec;q=self.v;w=self.phi

        def epsilon(u):
            return 0.5*(nabla_grad(u)+nabla_grad(u).T)

        def sigma(u):
            return (2*nabla_grad(u))

        def epsII(u):
            eps = epsilon(u)
            epsII_UFL = sqrt(eps**2 + Constant(1e-16)**2)
            return epsII_UFL

        def eta(u):
            return Constant(self.B)*epsII(u)**(1./3-1.0)

        # Effective acceleration due to gravity
        f = Constant(self.rho_i*self.g*(1-self.rho_i/self.rho_w))


        b = -self.rho_i/self.rho_w*H


        # SSA variational form
        Hmid = (H+q)/2
        F1 = inner(2*eta(u)*Hmid*grad(u), grad(v))*dx
        F2 = inner(0.5*f*Hmid**2, nabla_div(v))*dx
        # Drag
        F3 = inner(self.beta*u,v)*dx
        F = F1-F2+F3

        return F


    def advect(self,h0,uv,dt,accum=0.0):
        """
        Variational form for continuity equation
        h0 = ice thickness from previous step
        uv = velocity from previous step
        accum = accumulation rate (m/s)
        """

        n = self.ncell


        # Need to figure out CFL criterion to set maximum time step???
        dtc = Constant(dt)   # Make time step size a constant so we don't have to recompile

        # Define trial and vector spaces
        v,v_vec   = TrialFunctions(self.Q_sys); phi,phi_vec   = TestFunctions(self.Q_sys)

        # Flux at mid point for DG form of continuity equation
        #q = 0.5*(v_vec*v + uv*h0)
        #b = flux(phi, q, n)
        u = v_vec
        un = 0.5*(dot(u, n) + abs(dot(u, n)))
        t1 = v*div(phi*u)*dx + dot(u,grad(phi*v))*dx
        t2 = - 2*conditional(dot(u, n) > 0, phi*dot(u, n)*v, 0.0)*ds
        t3 = - 2*(phi('+') - phi('-'))*(un('+')*v('+') - un('-')*v('-'))*dS
        b = t1+t2+t3
        M = ((v-h0)/Constant(dt)*phi*dx)

        s = source(phi,accum)

        L1 = M - (b + s)

        return L1


    def integrate(self,H,U,dt=86400.0,Nt=1,accum=1e-16):
        """
        Integrate systems of equations for Nt steps.
        H = ice thickness
        U = velocity
        dt = time step (seconds)
        Nt = number of time steps to take
        accum = default accumulation
        """
        self.H, self.U = H, U
        for i in xrange(Nt):
            # Advect the front
            if self.advect_front:
                self.advect_mesh(self.U, dt)
            # Compute the new U and H fields
            self.step(dt,accum)
            # Advect the FBM tracer particles
            if self.fbm is not None:
                self.fbm.advect_particles(self.U, dt)
            # Advance the time
            self.t += dt
            # Check if calving-criterion is met anywhere
            if self.calve_flag and self.fbm.check_calving(self.U, self.U0):
                # If so, and if calve_flag is True, calve
                xc = self.fbm.calve()
                self.calve(xc)

            for obs in self.obslist: obs.notify_step(self, dt)

        return self.H,self.U

    def step(self,dt=86400.0,accum=1e-16):
        """Step forward in time once.
        """
        # Define variational forms
        L1 = self.velocity(self.H)
        L2 = self.advect(self.H,self.U,dt=dt,accum=accum)
        F = L1 + L2

        # Apply bcs
        bcs = self.set_bcs()

        # Solution vector
        dq = Function(self.Q_sys)
        assign(dq,[self.H,self.U])

        # Jacobian of non-linear systems
        R = action(F,dq)
        DR = derivative(R,  dq)

        # Setup variational form and solve
        problem = NonlinearVariationalProblem(R, dq, bcs, DR)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm["newton_solver"]["relative_tolerance"]=1e-6
        prm["newton_solver"]["absolute_tolerance"]=1.0

        solver.solve()
        # Store the solutions internally.
        self.H,self.U=dq.split(deepcopy = True)

    def advect_mesh(self, U, dt):
        """Advectt he mesh with the velocity field U over timestep U.
        """
        # First, create the new boundary by moving the calving front only.
        boundary = BoundaryMesh(self.mesh, "exterior")
        for x in boundary.coordinates():
            # Janky check, works in 1D
            if x[0] != 0:
                Lx = x[0] + U.vector().get_local()[0]*dt
                x[0] = Lx
                self.Lx = Lx

        # Copty the mesh, just in case
        old_mesh = Mesh(self.mesh)
        # Stretch the mesh using the new boundary.
        ALE.move(self.mesh, boundary) 
        # Do some checks!?
        # ?????
        # Delete the old mesh
        del old_mesh
        # Rebuild the mesh box.
        self.mesh.bounding_box_tree().build(self.mesh)

    def continuousH(self, H=None):
        """Convenience function for DG H (stored) to CG H (useful).
        """
        H = H or self.H
        return interpolate(self.H, self.Q_cg)

    @property
    def data(self):
        """Convenience function for saving (x, H, U)
        """
        return np.vstack([self.mesh.coordinates().squeeze()[::-1],
                            self.continuousH().vector().get_local(),
                            self.U.vector().get_local()])


    def calve(self, xc):
        """Calve the ice shelf at xc, and remesh.
        """
        assert xc < self.Lx

        # Make post-calving mesh
        new_mesh = IntervalMesh(self.Nx, 0.0, xc)

        # Create new function spaces on this mesh
        Q_sys, Q, Q_vec, Q_cg, Q_cg_vec, ncell, hcell, v, v_vec, phi, phi_vec = self.init_function_space(new_mesh,self.order)

        # Create new functions on the mesh
        Hnew = Function(Q_cg);Unew = Function(Q_vec)

        # Set ice thickness and velocity based on data interpolated from
        # pre-calved mesh
        Hcg = interpolate(self.H, self.Q_cg)
        Hnew_arr = np.array([Hcg(i) for i in new_mesh.coordinates()][::-1])

        Unew_arr = np.array([self.U(i) for i in new_mesh.coordinates()][::-1])

        Hnew.vector().set_local(Hnew_arr);Unew.vector().set_local(Unew_arr)

        # Replace functions
        self.H = interpolate(Hnew,Q);
        self.U = Unew.copy(deepcopy=True)


        self.Q_sys, self.Q, self.Q_vec, self.Q_cg, self.Q_cg_vec, self.ncell, self.hcell, self.v, self.v_vec, self.phi, self.phi_vec = Q_sys, Q, Q_vec, Q_cg, Q_cg_vec, ncell, hcell, v, v_vec, phi, phi_vec 
        # Dispose of original mesh and replace
        del self.mesh
        self.mesh = new_mesh

        for obs in self.obslist: obs.notify_calve(self.Lx, xc, self.t)

        self.Lx = xc

    def plot_hu(self,U0=None,H0=None):

        fig=plt.figure(1)
        fig.clf()
        plt.subplot(2,1,1)
        if U0 is not None:
            plot(U0[0]*time_factor,color='r',linestyle='--',label='analytic')
        plot(self.U[0]*time_factor,color='k',label='numeric')
        plt.xlabel('Distance (m)')
        plt.ylabel('Velocity (m/a)')
        plt.subplot(2,1,2)
        if H0 is not None: 
            plot(H0,color='r',linestyle='--',label='analytic')
        plot(self.H,color='k',label='numerical solution')
        plt.legend()
        plt.xlabel('Distance (m)')
        plt.ylabel('Ice shelf elevation (m.a.s.l.)')
        plt.ylim([0,self.H0])
        
        plt.plot()
        plt.show()


class FBMTracer(object):
    def __init__(self, Lx, N0, dist, xsep=1e16):
        """
        A container for tracer particles in a 1D fenics ice dynamics model.

        The tracers advect with the fluid velocity, but do not affect the flow
        until the fiber bundle models they carry break, at which point calving
        is triggered in the ice flow model.
        """
        self.x = list(np.linspace(0,Lx,N0,endpoint=False))
        self.s = [dist() for i in range(N0)]
        self.N = N0
        self.dist = dist
        self.xsep = xsep

    def advect_particles(self, Ufunc, dt):
        """
        Advect particles with the flow represented by vector coefficients
        Ufunc. Drop in a new particle if required.
        """
        for i in range(self.N):
            x = self.x[i]
            dx = Ufunc(x)*dt
            self.x[i] += dx

        if self.x[0] > self.xsep:
            self.add_particle(0)

    def add_particle(self, xp):
        """
        Introduce a new particle at location xp.
        """
        for i in range(self.N):
            if self.x[i] > xp:
                index = i
                break
        self.x.insert(i,xp)
        self.s.insert(i,self.dist())
        self.N += 1

    def remove_particle(self, i):
        """
        Revmoce ith particle from the flow, used in calving.
        """
        self.x.pop(i)
        self.s.pop(i)
        self.N -= 1

    def check_calving(self, Ufunc, U0):
        """Check if any FBMs have exceeded their thresholds.
        """
        U = np.array([Ufunc(x) for x in self.x])
        strains = np.log(U/U0)
        broken = np.argwhere(self.s < strains)
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
        i = i or self._minbroken
        del self._minbroken
        xc = self.x[i]
        j = self.N - 1
        while j >= i:
            self.remove_particle(j)
            j-=1
        return xc


#### SOME THRESHOLD DISTRIBUTION FUNCTIONS ####
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

def determine_F(ssamodel):
    grad_h = project(grad(ssamodel.H),ssamodel.Q_cg_vec)

def grav_prod(H, U, accum=1e-16):
    return H*(dot(U, nabla_grad(H)) + accum)

def compute_force(ssamodel):
    def epsilon(u):
        return 0.5*(nabla_grad(u)+nabla_grad(u).T)

    def sigma(u):
        return (2*nabla_grad(u))

    def epsII(u):
        eps = epsilon(u)
        epsII_UFL = sqrt(eps**2 + Constant(1e-16)**2)
        return epsII_UFL

    def eta(u):
        return Constant(ssamodel.B)*epsII(u)**(1./3-1.0)

    return eta(ssamodel.U)

class FrontObserver(object):
    def __init__(self, ssaModel):
        self.xc = [ssaModel.Lx]
        self.ts = [0]

    def notify_step(self, ssaModel, dt):
        self.xc.append(ssaModel.Lx)
        self.ts.append(self.ts[-1]+dt)

    def notify_calve(self, Lx, xc, t):
        pass

class CalveObserver(object):
    def __init__(self, ssaModel):
        self.ts = [0]
        self.xs = [ssaModel.Lx]

    def notify_step(self, ssaModel, dt):
        pass

    def notify_calve(self, Lx, xc, t):
        self.ts += [t, t]
        self.xs += [Lx, xc]

    @property
    def data(self):
        return np.vstack([self.ts, self.xs])


def plot_profiles(H, hnew, U, unew):
    fig=plt.figure(2)
    fig.clf()
    plt.subplot(2,1,1)
    plot(unew[0]*time_factor,color='k',label='numeric')
    plot(U[0]*time_factor,color='r',linestyle='--',label='analytic')
    plt.xlabel('Distance (m)')
    plt.ylabel('Velocity (m/a)')
    plt.subplot(2,1,2)
    
    plot(hnew,color='k',label='numerical solution')
    plot(H,color='r',linestyle='--',label='analytic')
    plt.legend()
    plt.xlabel('Distance (m)')
    plt.ylabel('Ice shelf elevation (m.a.s.l.)')
    plt.plot()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import sys, pickle

    # Setup some fenics log stuff to output diagnostic information
    set_log_level(20)
    set_log_active(False)
    
    # Example showing how to use the model
    # Example 1: Resolution too low to accurately resolve the velocity field
    Nx = 2**9    # Number of points in the x-direction
    Lx = 200e3/1 # Length of domain in the x-direction
    
    # Setup ice shelf parameters
    accum = 0.3/time_factor # m/s
    H0 = 500.0          # Ice thickness at the grounding line (m)
    U0 = 50/time_factor # Velocity of ice at the grounding line (m/a)
    
    # Initialize model
    mesh = IntervalMesh(Nx, 0.0, Lx)
    
    
    fbmkwargs={'Lx':Lx,
                   'N0':100,
                   'xsep':2000,
                   'dist':retconst(2.8)}
    ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True, 
                            calve_flag=True,fbmkwargs=fbmkwargs) 
    
    #ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True,calve_flag=True)
    del mesh
    x,H,U = ssaModel.steady_state(accum)
    H,U=ssaModel.init_shelf(accum)
    ssaModel.H, ssaModel.U = H, U
    

    # Example 1: Resolution too low to accurately resolve the velocity field
    if 'test' in sys.argv:
    	Nx = 160*8    # Number of points in the x-direction
    	Lx = 100e3/4 # Length of domain in the x-direction
    	
    	# Setup evenly spaced grid (This need to change for 2D solution)
    	mesh = IntervalMesh(Nx, 0.0, Lx)
    	
    	# Setup ice shelf parameters
    	accum = 0.1/time_factor
    	H0 = 1000.0          # Ice thickness at the grounding line (m)
    	U0 = 250/time_factor # Velocity of ice at the grounding line (m/a)
    	
    	
    	# Initialize model
    	ssaModel = ssa1D(mesh,order=1)
    	x,H,U = ssaModel.steady_state(accum)
    	H,U=ssaModel.init_shelf(accum)
    	
    	# Setup some fenics log stuff to output diagnostic information
    	set_log_level(20)
    	set_log_active(True)
    	
    	# Integrate model for 10 time steps
    	hnew,unew = ssaModel.integrate(H,U,dt=86400.*100,Nt=10,accum=Constant(accum))
    	H,U=ssaModel.init_shelf(accum)
    	
    	fig=plt.figure(2)
    	fig.clf()
    	plt.subplot(2,1,1)
    	plot(unew[0]*time_factor,color='k',label='numeric')
    	plot(U[0]*time_factor,color='r',linestyle='--',label='analytic')
    	plt.xlabel('Distance (m)')
    	plt.ylabel('Velocity (m/a)')
    	plt.subplot(2,1,2)
    	
    	plot(hnew,color='k',label='numerical solution')
    	plot(H,color='r',linestyle='--',label='analytic')
    	plt.legend()
    	plt.xlabel('Distance (m)')
    	plt.ylabel('Ice shelf elevation (m.a.s.l.)')
    	plt.plot()
    	plt.show()

    #H,U = ssaModel.integrate(H,U,dt=864000.,Nt=50000,accum=Constant(1*accum)) 
    #import pickle
    #pickle.dump(ssaModel.obslist[0].data,
    #            open('./frontev_xsep2000_const_2p8_dt_864000_Nt_50000', 'w'))
 
    #fronts = []
    #hnew,unew = ssaModel.integrate(H,U,dt=86400.,Nt=1,accum=Constant(1*accum))
    #fronts.append(ssaModel.Lx)
    #ssaModel.calve(Lx)
    # Integrate model for 10 time steps
    
    # Calving check: advect a few timesteps, calve back to x=200000, 10 times
    #ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True,calve_flag=True)
    #for i in range(10):
    #    hnew,unew = ssaModel.integrate(ssaModel.H,ssaModel.U,dt=86400.,Nt=100,accum=Constant(1*accum))
    #    #hnew,unew = ssaModel.integrate(H,U,dt=86400.*100,Nt=20,
    #    #                        accum=Expression('accum - (Lx-x[0])*25/3e7/Lx', degree=1, 
    #    #                        Lx=Lx, accum=accum))
    #    # Redfien initial, equilibrium, geometries
    #    fronts.append(ssaModel.Lx)
    #    ssaModel.calve(Lx)
    #    H,U=ssaModel.init_shelf(accum)

    # Mesh-refinement convergence test for time-stepping
    if 'dxconv' in sys.argv: 
        for fac in [9,10,11,12,13]:
            DT = 8640000
            Nx = 2**fac    # Number of points in the x-direction
            Lx = 200e3/1 # Length of domain in the x-direction
            DX = Lx/Nx
            print('CFL cond numb: {}'.format(U0*DT/DX))
            
            # Setup ice shelf parameters
            accum = 0.3/time_factor # m/s
            H0 = 500.0          # Ice thickness at the grounding line (m)
            U0 = 50/time_factor # Velocity of ice at the grounding line (m/a)
            
            # Initialize model
            mesh = IntervalMesh(Nx, 0.0, Lx)
            fbmkwargs={'Lx':Lx,
                       'N0':1000,
                       'xsep':20,
                       'dist':retconst(2.8)}
            ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=False, 
                                calve_flag=False,fbmkwargs=fbmkwargs)  
            del mesh
            x,H,U = ssaModel.steady_state(accum)
            H,U=ssaModel.init_shelf(accum)
        
            
            H,U = ssaModel.integrate(H,U,dt=DT,Nt=2000,accum=Constant(accum))
            DIR = './tests/mr_conv/'
            FPROFBASE ='profs_noadv_dt_{}_dx_{}_Nt_2000.txt' 
            np.savetxt(DIR+FPROFBASE.format(DT,DX), ssaModel.data)
    
    
    # Mesh-refinement convergence test for advection
    if 'dxconvadv' in sys.argv:
    	for fac in [9,10,11,12,13]:
    	    DT = 8640000
    	    Nx = 2**fac    # Number of points in the x-direction
    	    Lx = 200e3/1 # Length of domain in the x-direction
    	    DX = Lx/Nx
    	    print('CFL cond numb: {}'.format(U0*DT/DX))
    	    
    	    # Setup ice shelf parameters
    	    accum = 0.3/time_factor # m/s
    	    H0 = 500.0          # Ice thickness at the grounding line (m)
    	    U0 = 50/time_factor # Velocity of ice at the grounding line (m/a)
    	    
    	    # Initialize model
    	    mesh = IntervalMesh(Nx, 0.0, Lx)
    	    fbmkwargs={'Lx':Lx,
    	               'N0':1000,
    	               'xsep':20,
    	               'dist':retconst(2.8)}
    	    ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True, 
    	                        calve_flag=False,fbmkwargs=fbmkwargs)  
    	    del mesh
    	    x,H,U = ssaModel.steady_state(accum)
    	    H,U=ssaModel.init_shelf(accum)
    	
    	    
    	    H,U = ssaModel.integrate(H,U,dt=DT,Nt=2000,accum=Constant(accum))
    	    DIR = './tests/mr_conv/'
    	    FPROFBASE ='profs_adv_dt_{}_dx_{}_Nt_2000.txt' 
    	    np.savetxt(DIR+FPROFBASE.format(DT,DX), ssaModel.data)
     
    # Mesh-refinement convergence test for calving
    if 'dxconvcav' in sys.argv:
    	for fac in [9,10,11,12,13]:
    	    DT = 8640000
    	    Nx = 2**fac    # Number of points in the x-direction
    	    Lx = 200e3/1 # Length of domain in the x-direction
    	    DX = Lx/Nx
    	    print('CFL cond numb: {}'.format(U0*DT/DX))
    	    
    	    # Setup ice shelf parameters
    	    accum = 0.3/time_factor # m/s
    	    H0 = 500.0          # Ice thickness at the grounding line (m)
    	    U0 = 50/time_factor # Velocity of ice at the grounding line (m/a)
    	    
    	    # Initialize model
    	    mesh = IntervalMesh(Nx, 0.0, Lx)
    	    fbmkwargs={'Lx':Lx,
    	               'N0':1000,
    	               'xsep':20,
    	               'dist':retconst(2.8)}
    	    ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True, 
    	                        calve_flag=True,fbmkwargs=fbmkwargs)  
    	    del mesh
    	    x,H,U = ssaModel.steady_state(accum)
    	    H,U=ssaModel.init_shelf(accum)
    	
    	    
    	    H,U = ssaModel.integrate(H,U,dt=DT,Nt=2000,accum=Constant(accum))
    	    DIR = './tests/mr_conv/'
    	    FPROFBASE ='profs_calv_uni_2p8_N0_1000_xsep_20_dt_{}_dx_{}_Nt_2000.txt' 
    	    FFRONTBASE ='front_calv_uni_2p8_N0_1000_xsep_20_dt_{}_dx_{}_Nt_2000.txt' 
    	    np.savetxt(DIR+FPROFBASE.format(DT,DX), ssaModel.data)
    	    np.savetxt(DIR+FFRONTBASE.format(DT,DX), ssaModel.obslist[0].data)
    
    
    #for fac in [9,10,11,12,13]:
    #    DT = 8640000
    #    Nx = 2**fac
    #    Lx = 200e3
    #    DX = Lx/Nx
    #    DIR = './tests/mr_conv/' 
    #    FFRONTBASE ='front_calv_uni_2p8_N0_1000_xsep_20_dt_{}_dx_{}_Nt_2000.txt' 
    #    t, front = np.loadtxt(DIR+FFRONTBASE.format(DT,DX))
    #    plt.plot(t, front)
    #
    #plt.show()
    
    
    
    # Mesh-refinement convergence test result plot
    if 'mrconvplot' in sys.argv:
    	errs_cav = []
    	dens_cav = []
    	errs_adv = []
    	dens_adv = []
    	errs_noadv = []
    	dens_noadv = []
    	
    	plt.figure()
    	plt.gca().set_prop_cycle(None)
    	for fac in [9,10,11,12,13]:
    	    DT = 8640000
    	    Nx = 2**fac
    	    Lx = 200e3
    	    DX = Lx/Nx
    	    DIR = './tests/mr_conv/'
    	    FPROFBASE ='profs_calv_uni_2p8_N0_1000_xsep_20_dt_{}_dx_{}_Nt_2000.txt'
    	
    	
    	    x,H,U = np.loadtxt(DIR+FPROFBASE.format(DT,DX))
    	    C, n = ssaModel.C, ssaModel.n
    	    D = U0*(1-C/accum*H0**(n+1))
    	    Han = (C/accum - (U0**(n+1)*(C/accum*H0**(n+1)-1))/(accum*x+H0*U0)**(n+1))**(-1/(n+1))
    	    Uan = (H0*U0+accum*x)/Han
    	
    	    plt.plot(x, np.abs(Uan-U))
    	    errs_cav.append(np.sum((Uan-U)**2)/2**fac)
    	    dens_cav.append(x.max()/2**fac)
    	
    	plt.gca().set_prop_cycle(None)
    	for fac in [9,10,11,12,13]:
    	    DT = 8640000
    	    Nx = 2**fac
    	    Lx = 200e3
    	    DX = Lx/Nx
    	    DIR = './tests/mr_conv/'
    	    FPROFBASE ='profs_adv_dt_{}_dx_{}_Nt_2000.txt' 
    	    x,H,U = np.loadtxt(DIR+FPROFBASE.format(DT,DX))
    	    C, n = ssaModel.C, ssaModel.n
    	    D = U0*(1-C/accum*H0**(n+1))
    	    Han = (C/accum - (U0**(n+1)*(C/accum*H0**(n+1)-1))/(accum*x+H0*U0)**(n+1))**(-1/(n+1))
    	    Uan = (H0*U0+accum*x)/Han
    	
    	    plt.plot(x, np.abs(Uan-U), '--')
    	    errs_adv.append(np.sum((Uan-U)**2)/2**fac)
    	    dens_adv.append(x.max()/2**fac)
    	
    	plt.gca().set_prop_cycle(None)
    	for fac in [9,10,11,12,13]:
    	    DT = 8640000
    	    Nx = 2**fac
    	    Lx = 200e3
    	    DX = Lx/Nx
    	    DIR = './tests/mr_conv/'
    	    FPROFBASE ='profs_noadv_dt_{}_dx_{}_Nt_2000.txt' 
    	
    	    x,H,U = np.loadtxt(DIR+FPROFBASE.format(DT,DX))
    	    C, n = ssaModel.C, ssaModel.n
    	    D = U0*(1-C/accum*H0**(n+1))
    	    Han = (C/accum - (U0**(n+1)*(C/accum*H0**(n+1)-1))/(accum*x+H0*U0)**(n+1))**(-1/(n+1))
    	    Uan = (H0*U0+accum*x)/Han
    	
    	    plt.plot(x, np.abs(Uan-U), '-.')
    	    errs_noadv.append(np.sum((Uan-U)**2)/2**fac)
    	    dens_noadv.append(x.max()/2**fac)
    	
    	
    	plt.xlabel('Distance from GL (m)')
    	plt.ylabel('Velocity Err')
    	plt.show()
    	
    	plt.figure()
    	plt.subplot(211)
    	plt.title('Velocity MSE after 2000 steps')
    	plt.loglog(2**np.array([9,10,11,12,13]), errs_cav, 'k-', label='Calve,Adv')
    	plt.loglog(2**np.array([9,10,11,12,13]), errs_adv, 'k--', label='Adv')
    	plt.loglog(2**np.array([9,10,11,12,13]), errs_noadv, 'k-.', label='Stationary')
    	plt.xlabel('Num Points')
    	plt.ylabel('MSE (m/yr)^2')
    	
    	plt.subplot(212)
    	plt.loglog(dens_cav, errs_cav, 'k-', label='Calve,Adv')
    	plt.loglog(dens_adv, errs_adv, 'k--', label='Adv')
    	plt.loglog(dens_noadv, errs_noadv, 'k-.', label='Stationary')
    	plt.xlabel('Average point separation')
    	plt.ylabel('MSE (m/yr)^2')
    	plt.legend()
    	
    	plt.tight_layout()
    	plt.show()
    
    
    
    #
    #
    # Higher time resolution with highes space resolution, for good measure.
    #mesh = IntervalMesh(Nx, 0.0, Lx)
    #ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True,calve_flag=True)
    #del mesh
    #x,H,U = ssaModel.steady_state(accum)
    #H,U=ssaModel.init_shelf(accum)
    #
    #H,U = ssaModel.integrate(H,U,dt=432000,Nt=1000,accum=Constant(1*accum))
    #front_evs.append(ssaModel.obslist[0].xc)
    #
    #dxs = [390, 195, 97, 48, 24, 24]
    #dts = [864000, 864000, 864000, 864000, 864000, 432000]
    #
    #for front_ev, dx, dt in zip(front_evs, dxs, dts):
    #    plt.plot(np.array(ssaModel.obslist[0].ts)/time_factor, front_ev, 
    #        label='dx={}, dt={}')
    #plt.xlabel('Time (yr)')
    #plt.ylabel('Front (m)')
    #plt.legend()
    #plt.show()
    
    
    # Uniform Distribution, number of particles experiment
    #fronts_2000 = []
    #for i in range(10):
    #    Nx = 2**13    # Number of points in the x-direction
    #    Lx = 200e3/1 # Length of domain in the x-direction
    #    
    #    # Setup ice shelf parameters
    #    accum = 0.3/time_factor # m/s
    #    H0 = 500.0          # Ice thickness at the grounding line (m)
    #    U0 = 50/time_factor # Velocity of ice at the grounding line (m/a)
    #    
    #    # Initialize model
    #    mesh = IntervalMesh(Nx, 0.0, Lx)
    #    fbmkwargs={'Lx':Lx,
    #               'N0':100,
    #               'xsep':2000,
    #               'dist':uni_dist(2.8,3.0)}
    #    ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True, 
    #                        calve_flag=True,fbmkwargs=fbmkwargs) 
    #
    #    del mesh
    #    x,H,U = ssaModel.steady_state(accum)
    #    H,U=ssaModel.init_shelf(accum)
    #
    #    
    #    H,U = ssaModel.integrate(H,U,dt=86400.*10,Nt=5000,accum=Constant(1*accum))
    #    fronts_2000.append([ssaModel.obslist[0].ts, ssaModel.obslist[0].xs])
    #
    #fronts_200 = []
    #for i in range(10):
    #    Nx = 2**13    # Number of points in the x-direction
    #    Lx = 200e3/1 # Length of domain in the x-direction
    #    
    #    # Setup ice shelf parameters
    #    accum = 0.3/time_factor # m/s
    #    H0 = 500.0          # Ice thickness at the grounding line (m)
    #    U0 = 50/time_factor # Velocity of ice at the grounding line (m/a)
    #    
    #    # Initialize model
    #    mesh = IntervalMesh(Nx, 0.0, Lx)
    #    fbmkwargs={'Lx':Lx,
    #               'N0':1000,
    #               'xsep':200,
    #               'dist':uni_dist(2.8,3.0)}
    #    ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True, 
    #                        calve_flag=True,fbmkwargs=fbmkwargs) 
    #
    #    del mesh
    #    x,H,U = ssaModel.steady_state(accum)
    #    H,U=ssaModel.init_shelf(accum)
    #
    #    
    #    H,U = ssaModel.integrate(H,U,dt=86400.*10,Nt=5000,accum=Constant(1*accum))
    #    fronts_200.append([ssaModel.obslist[0].ts, ssaModel.obslist[0].xs])
    
    
    
    
    #t = Expression('15000000*exp(-pow((Lx/2-x[0])/(Lx/16),2))',degree=1,Lx=Lx)
    
    
    
