"""
 event_driven_tongue.py

 Author: Samuel B. Kachuck

 An FBM-calving ice tongue using an event-driven technique, for rapid testing.
 The tongue adfvances from the grounding line, following the analytical
 solution for an ice tongue.

 Limitations:
 - The FBM tracers don't advcect with the ice velocity, but march along at
 constant steps, keeping the distance between them the same (could fix
 somehow)
 - Requires steady state melt profile - no climatic forcing.

"""

import numpy as np
from scipy.optimize import bisect

from fbmtracer import FBMMaxStrengthTracer
from util import *
from analytictongue import *

time_factor = 86400.0*365.24

# The number of calving events to observe in the model
NUM_CALVE = 100

# Form the ice tongue profiles
H0 = 434.
U0 = 95./time_factor
MDOT = -2./time_factor
A = 2.54e-17
tongue = AnalyticTongue(h0=H0, u0=U0, mdot=MDOT, A=A)

# Form the initial fiber bundles
DX_fbm = 10.

def int_strain(x, tongue):
    return np.log(tongue.u(x)/tongue.u0)

def thresh_dist_from_eps(eps, tongue):
    f = lambda x: int_strain(x,tongue)-eps
    x0 = bisect(f,0,tongue.lmax-1e-10)
    return x0

Nx0 = int(tongue.lmax/DX_fbm)
fbms = FBMMaxStrengthTracer(tongue.lmax, Nx0, dist=uni_dist(4.5, 4.8))
# It puts them at the wrong place, so replace them
fbms._toArr()
fbms.x = np.arange(Nx0)*DX_fbm
# Convert strengths to distances along integrated strain curve
for i, s in enumerate(fbms.s):
    fbms.s[i] = thresh_dist_from_eps(s, tongue)

fbms.state = fbms.x[:]

# Lop off already broken end
if fbms.check_calving():
    L = fbms.calve()
else:
    L = tongue.lmax

# Stuff to store
calving_sizes = []
lengths_at_calving = []
steps_at_calving = []

ncalve = 0
nsteps = 0
while ncalve<NUM_CALVE:
    fbms._toArr()
    # Intermediate event in the case that no FBMs remain
    if len(fbms.s) == 0:
        fbms.add_particle(0.)
        # and convert its strength to distence along integrated strain curve
        fbms.s[0] = thresh_dist_from_eps(fbms.s[0], tongue)
        # Compute new fbm states
        fbms.state = fbms.x[:]
    # Event type 1 - need to insert particle to keep DX_fbm between them.
    elif (1-(L/DX_fbm % 1.))*DX_fbm < np.min(fbms.s - fbms.state): 
        # All distances move up to new fbm insertion
        dx = (1-(L/DX_fbm % 1.))*DX_fbm
        L += dx
        fbms.x += dx
        # Add particle
        fbms.add_particle(0)
        # and convert its strength to distence along integrated strain curve
        fbms.s[0] = thresh_dist_from_eps(fbms.s[0], tongue)
        # Compute new fbm states
        fbms.state = fbms.x[:]

    # Event type 2 - bundle is close to breaking, so move to break it.
    else:
        # Find the index of the FB nearest to breaking
        i = np.argmin(fbms.s - fbms.state)
        dx = np.min(fbms.s - fbms.state)
        # Move all FBs up
        fbms.x += dx
        # And calve
        fbms._minbroken = None
        xc = fbms.calve(i)
        # Store new ice front position
        print("Step {}: Calving from {} to {}, distance of {}".format(nsteps, L, xc, L - xc))

        # Stuff to store
        calving_sizes.append(L-xc)
        lengths_at_calving.append(L)
        steps_at_calving.append(nsteps)
        L = xc
        # Compute new fbm states for remaining FBs
        fbms._toArr()
        fbms.state = fbms.x[:]
        ncalve+=1

    nsteps += 1
