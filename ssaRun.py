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

# Set simulation parameters
DT = 8640000
Nx = 2**8    # Number of points in the x-direction
Lx = 200e3/1 # Length of domain in the x-direction
DX = Lx/Nx

# Set ice shelf parameters
accum = 0.3/time_factor # m/s
H0 = 500.0          # Ice thickness at the grounding line (m)
U0 = 50/time_factor # Velocity of ice at the grounding line (m/a)

print('CFL cond numb: {}'.format(U0*DT/DX))

# Initialize model
mesh = IntervalMesh(Nx, 0.0, Lx)
fbmkwargs={'Lx':Lx,
           'N0':1000,
           'xsep':20,
           #'dist':retconst(2.8)}
           'dist':uni_dist(2.5, 4)}
ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=True, 
                    calve_flag=True,fbmkwargs=fbmkwargs) ; 
del mesh
x,H,U = ssaModel.steady_state(accum)
H,U=ssaModel.init_shelf(accum)

# Run the model in time
H,U = ssaModel.integrate(H,U,dt=DT,Nt=2000,accum=Constant(accum));

# plot with plt.plot(
