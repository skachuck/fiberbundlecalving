"""
Simplified script for running and exploring the Fiber-Bundle enabled shallow
shelf model.
Recommended use is to run this script in an ipython session or notebook:
>> %run ssaRun.py
Can plot the calving events
>> plt.plot(ssaModel.obslist[0].ts, ssaModel.obslist[0].xs)
"""
# Import the models
from ssaModel import *
import pylab as plt

# Setup some fenics log stuff to output diagnostic information
set_log_level(20)
set_log_active(False)

# Set simulation parameters
DT = 8640000
Nx = 2**8    # Number of points in the x-direction
Lx = 200e3/1 # Length of domain in the x-direction
DX = Lx/Nx
NT = 200

# Set ice shelf parameters
accum = -2./time_factor # m/s
H0 = 434.          # Ice thickness at the grounding line (m)
U0 = 95./time_factor # Velocity of ice at the grounding line (m/a)
B = (2.54e-17)**(-1./3.)

# Set ice shel parameters
accum = 0.5/time_factor
H0 = 500.
U0 = 50./time_factor
B = 0.5e8


print('CFL cond numb: {}'.format(U0*DT/DX))

# Initialize model
mesh = IntervalMesh(Nx, 0.0, Lx)
fbmkwargs={'Lx':Lx,
           'N0':1000,
           'Nf':10,
           'xsep':200}

# In the large-Nf limit, the maximum stress per fiber of a bundle of fibers 
# with strictly increasing strengths (with maximum xmax) is Fmax/Nf = xmax^2/4.
# For Fmax=2.8, we need xmax=sqrt(4*Fmax/Nf)=1.058
# Accounting for fluctuations due to finite N (Fmax_actual = Fmax + d N**1/3),
# with d~0.3138 for the strict distribution over [0,1], get xmax=0.923
fbmkwargs['dist'] = strict_dist(0,0.923)
#fbmkwargs['dist'] = uni_dist(0,0.923)
ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,B=B,
                    advect_front=True, calve_flag=True,
                    fbm_type='full', fbmkwargs=fbmkwargs) ; 

del mesh
x,H,U = ssaModel.steady_state(accum)
H,U=ssaModel.init_shelf(accum)

# Run the model in time
H,U = ssaModel.integrate(H,U,dt=DT,Nt=NT,accum=Constant(accum));

# get damage of fibers over time with ssaModel.fbmobs.data
# get front position over time with ssaModel.frontobs.data
# get calving events with ssaModel.calveobs.data

