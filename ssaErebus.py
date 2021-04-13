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

# Setup some fenics log stuff to output diagnostic information
set_log_level(00)
set_log_active(False)

#NT = 365 --> 100 years
#NT = 456 --> 125 years
#NT = 100 --> 27.5 years
# Set simulation parameters
DT = 8640000
Nx = 2**8    # Number of points in the x-direction
Lx = 200e3/1 # Length of domain in the x-direction
DX = Lx/Nx
NT = 365

# Set ice shelf parameters
accum = -2./time_factor # m/s
H0 = 434.          # Ice thickness at the grounding line (m)
U0 = 95./time_factor # Velocity of ice at the grounding line (m/a)
B = (2.54e-17/time_factor)**(-1./3.)
Lx = -H0*U0/accum - 1e-6

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
           'xsep':20,
           'fbm_type':'full'}

# In the large-Nf limit, the maximum stress per fiber of a bundle of fibers 
# with strictly increasing strengths (with maximum xmax) is Fmax/Nf = xmax^2/4.
# For Fmax=2.8, we need xmax=sqrt(4*Fmax/Nf)=1.058
# Accounting for fluctuations due to finite N (Fmax_actual = Fmax + d N**1/3),
# with d~0.3138 for the strict distribution over [0,1], get xmax=0.923
#fbmkwargs['dist'] = strict_dist(0,0.923)

# fbmkwargs['dist'] = strict_dist(0,0.179)
fbmkwargs['dist'] = uni_dist(0,0.179)
results = []
for i in range(10):
    mesh = IntervalMesh(Nx, 0.0, Lx)
    ssaModel = ssa1D(mesh,order=1,U0=U0,H0=H0,B=B,
                        advect_front=True, calve_flag=True,
                        fbm_type='full', fbmkwargs=fbmkwargs,
                        Lmax=-H0*U0/accum) ; 
    del mesh
    x,H,U = ssaModel.steady_state(accum)
    H,U=ssaModel.init_shelf(accum)
    ssaModel.H = H
    ssaModel.U = U

    # Run the model in time
    H,U = ssaModel.integrate(H,U,dt=DT,Nt=NT,accum=accum); #getting error here when trying to run the model in time
    #self.x doesn't have any particles and it's trying to pull from an empty list in self.x.pop(i)

    
    results.append(ssaModel.frontobs.data)

#plt.plot(*ssaModel.fbmobs.data, marker='o')
#plt.xlabel('Position of particles (m)')
#plt.ylabel('Damage of particles (dimless)')
#plt.show()

event_counts = []
for entry in results:
    event_count = 0
    plt.plot(entry[0]/time_factor, entry[1])
    for i in range(1,len(entry[1])):
        current_value = entry[1][i]
        previous_value = entry[1][i-1]
        if current_value < previous_value:
            event_count +=1
    event_counts.append(event_count)

print('Number of calving events per run:',event_counts)
print('Average number of calving events in 10 runs:',sum(event_counts)/len(event_counts))

for entry in results:
   plt.plot(entry[0]/time_factor, entry[1])
plt.xlabel('Time (yrs)')
plt.ylabel('Front pos (m)')
plt.show()

# get latest damage of fibers with ssaModel.fbmobs.data
# get front position over time with ssaModel.frontobs.data
# get calving events with ssaModel.calveobs.data


#fbmkwargs={'Lx':Lx,
#           'N0':1000,
#           'xsep':200,
#           'fbm_type':'max',
#           'dist':retconst(2.8)}
