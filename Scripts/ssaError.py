"""
Program to calculate the error between the numerical and analytical solution of the SSA model
"""
#from array import *
#import matplotlib.pyplot as plt
#import numpy as np
from ssaModel import *


# Set simulation parameters

DT = 8640000
#Nx = 2**5   # Number of points in the x-direction
Lx = 200e3/1 # Length of domain in the x-direction
#DX = Lx/Nx

# Set ice shelf parameters
accum = 0.3/time_factor # m/s
H0 = 500.0          # Ice thickness at the grounding line (m)
U0 = 50/time_factor # Velocity of ice at the grounding line (m/a)

# Initialize model
fbmkwargs={'Lx':Lx,
            'N0':1000,
            'xsep':20,
            'dist':retconst(2.8)}
            #'dist':uni_dist(2.5, 4)}
                    
Usave = []
Nxsave = []

for i in range(1,3):
    Nx = 160*i
    mesh = IntervalMesh(Nx, 0.0, Lx)
    ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,advect_front=False,
    calve_flag=False,fbmkwargs=fbmkwargs) ;
    del mesh
    x,H,U = ssaModel.steady_state(accum)
    H,U=ssaModel.init_shelf(accum)
    ssaModel.H, ssaModel.U = H, U
    H,U = ssaModel.integrate(H,U,dt=DT,Nt=100,accum=Constant(accum));
    x,Hanalytic,Uanalytic = ssaModel.steady_state(accum)
    Uerr = U.vector().get_local() - Uanalytic.squeeze()
    Usave.append(Uerr)
    Nxsave.append(Nx)
    plt.plot(x,Uerr,label='error')

#Unorm = [np.sum(Ui**2) for Ui in Usave]
#plt.plot(Nxsave, Unorm)
    
plt.xlabel('Velocity')
plt.ylabel('Total Error')
plt.legend()
plt.show()

       

