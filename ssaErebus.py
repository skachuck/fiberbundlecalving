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
from analytictongue import AnalyticTongue

# Setup some fenics log stuff to output diagnostic information
set_log_level(50)
set_log_active(False)

# Set simulation parameters
DT = 864000
Nx = 2**12    # Number of points in the x-direction
Lx = 200e3/1 # Length of domain in the x-direction

NT = 100

# Set ice shelf parameters
accum = -2./time_factor # m/s
H0 = 434.          # Ice thickness at the grounding line (m)
U0 = 95./time_factor # Velocity of ice at the grounding line (m/a)
B = (2.54e-17/time_factor)**(-1./3.)
Lx = -H0*U0/accum - 1e-6
Lx = 19.5e3

DX = Lx/Nx

erebus = AnalyticTongue(mdot=-2, A=2.54e-17, h0=434, u0=95, rhoi=920, rhow=1030)
# Set ice shel parameters
#accum = 0.5/time_factor
#H0 = 500.
#U0 = 50./time_factor
#B = 0.5e8

print('CFL cond numb: {}'.format(U0*DT/DX))

# Initialize model
mesh = IntervalMesh(Nx, 0.0, Lx)
fbmkwargs={'Lx':Lx,
           'N0':1000,
           'Nf':10,
           'xsep':200,
           'fbm_type':'full'}

#t = Expression('1.5e7*exp(-pow((Lx/2-x[0])/(Lx/16),2))',degree=1,Lx=Lx)
# In the large-Nf limit, the maximum stress per fiber of a bundle of fibers 
# with strictly increasing strengths (with maximum xmax) is Fmax/Nf = xmax^2/4.
# For Fmax=2.8, we need xmax=sqrt(4*Fmax/Nf)=1.058
# Accounting for fluctuations due to finite N (Fmax_actual = Fmax + d N**1/3),
# with d~0.3138 for the strict distribution over [0,1], get xmax=0.923
#fbmkwargs['dist'] = strict_dist(0,0.923)
fbmkwargs['dist'] = strict_dist(0,0.169)
#fbmkwargs['dist'] = strict_dist(0,0.168)
#fbmkwargs['dist'] = strict_dist(0,0.16)
#fbmkwargs['dist'] = uni_dist(0,0.179)
results = []
lmaxs = []
umaxs = []
for i in range(1):
    mesh = IntervalMesh(Nx, 0.0, Lx)
    ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,B=B,
                        #beta=1.5e7,
                        #beta=t,
                        #min_thk=1., calve_flag=True,
                        #advect_front=True, calve_flag=False,
                        #fbm_type='full', fbmkwargs=fbmkwargs,
                        Lmax=-H0*U0/accum) ; 
    del mesh
    x,H,U = ssaModel.steady_state(accum)
    H,U=ssaModel.init_shelf(accum)
    ssaModel.H = H
    ssaModel.U = U
    

#    H,U = ssaModel.integrate(H,U,dt=DT,Nt=NT,accum=accum);
    lmaxs.append(ssaModel.Lx)
    umaxs.append(ssaModel.data[2].max())
#    ssaModel.advect_front=False
    # Run the model in time
    for j in range(NT):
        H,U = ssaModel.integrate(H,U,dt=DT,Nt=1,accum=accum);
        lmaxs.append(ssaModel.Lx)
        umaxs.append(ssaModel.data[2].max())

    
    lmaxs = np.array(lmaxs)
    umaxs = np.array(umaxs)
    results.append(ssaModel.frontobs.data)


plt.plot(ssaModel.data[0], ssaModel.data[1])
plt.plot(ssaModel.data[0], erebus.h(ssaModel.data[0]))
plt.xlabel('Distance from GL (m)')
plt.ylabel('Thickness (m)')
plt.show()

plt.plot(ssaModel.data[0], ssaModel.data[2]*time_factor)
plt.plot(ssaModel.data[0], erebus.u(ssaModel.data[0]))
plt.xlabel('Distance from GL (m)')
plt.ylabel('Velocity (m/yr)')
plt.show()

#plt.plot(*ssaModel.fbmobs.data, marker='o')
#plt.xlabel('Position of particles (m)')
#plt.ylabel('Damage of particles (dimless)')
#plt.show()
#
#for entry in results:
#    plt.plot(entry[0]/time_factor, entry[1])
#plt.xlabel('Time (yrs)')
#plt.ylabel('Front pos (m)')
#plt.show()

# get latest damage of fibers with ssaModel.fbmobs.data
# get front position over time with ssaModel.frontobs.data
# get calving events with ssaModel.calveobs.data


#fbmkwargs={'Lx':Lx,
#           'N0':1000,
#           'xsep':200,
#           'fbm_type':'max',
#           'dist':retconst(2.8)}
