"""
analyticTongue.py
Author: Samuel B. Kachuck
Date: Aug 7 2020

Class containing the analytic solution for an ice tongue.
"""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# Defining constants
RHOI = 910.
RHOW = 1028.
RN = RHOI/RHOW/2.
G = 9.81
A = 1.294e-17
N = 3.

H0 = 340.
U0 = 100.
MDOT = -1. # m/yr (melting is negative)

class AnalyticTongue(object):
    def __init__(self, h0=H0, u0=U0, mdot=MDOT, n=N, A=A, rhow=RHOW, rhoi=RHOI, g=G):
        self.h0=h0; self.u0=u0; self.mdot=mdot
        self.n=n; self.A=A;
        self.rhow=rhow; self.rhoi=rhoi; self.g=g

    def h(self, x):
        """
        """
        h0=self.h0; u0=self.u0; mdot=self.mdot
        n=self.n; A=self.A;
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        
        c = A*(rhoi*g*(rhow-rhoi)/4/rhow)**n
        n1 = n+1

        # No melt or ablation or refreeze
        if mdot==0:
            h = ((n+1)*c*x/h0/u0 + h0**(-(n+1)))**(-1./(n+1))
        # Melt
        elif mdot<0:
            mdot=abs(mdot)
            cm = c/mdot
            h = ( u0**n1*(1 + cm*h0**n1)/(h0*u0 - mdot*x)**n1 - cm)**(-1./n1)
        # refreeze
        elif mdot>0:
            cm = c/mdot
            h = ( cm - u0**n1*(cm*h0**n1 - 1.)/(h0*u0 + mdot*x)**n1 )**(-1./n1)
        return h

    def u(self, x):
        h0=self.h0; u0=self.u0; mdot=self.mdot
        n=self.n; A=self.A;
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        
        thk = self.h(x)
        return (u0*h0+mdot*x)/thk

    def epsxx(self, x):
        h0=self.h0; u0=self.u0; mdot=self.mdot
        n=self.n; A=self.A;
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        
        thk = self.h(x)
        return A*(rhoi*g*thk * (1-rhoi/rhow)/4)**n

    def tauxx(self, x):
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        return (rhow-rhoi)/(4*rhow)*rhoi*g*self.h(x)

    def r(self, x):
        h0=self.h0; u0=self.u0; mdot=self.mdot
        n=self.n; A=self.A;
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        
        c = A*(rhoi*g*(rhow-rhoi)/4/rhow)**n
        lmax = h0*u0/-mdot
        xcr = lmax*(1-((-mdot+c*h0**(n+1))/((n+1)*c*h0**(n+1)))**(1/(n+1)))
        rn = rhoi/rhow/2.
        ucr = self.u(xcr)
        u = self.u(x)

        dam = rn*np.ones_like(x)
        dam[np.argwhere(x>xcr)] *= (h0*u0+mdot*xcr)/(h0*u0+mdot*x[np.argwhere(x>xcr)])*(ucr/u[np.argwhere(x>xcr)])**n
        return np.maximum(np.minimum(dam, 1.), rn)

    def s(self, x):
        h0=self.h0; u0=self.u0; mdot=self.mdot
        n=self.n; A=self.A;
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        
        h = self.h(x)
        epsxx = self.epsxx(x)
        src = (-3*epsxx-mdot/h)
        return np.maximum(src, 0)
    
    @property
    def lmax(self):
        assert self.mdot<0, 'Must be melting'
        h0=self.h0; u0=self.u0; mdot=self.mdot
        n=self.n; A=self.A;
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        lmax = h0*u0/abs(mdot)
        return lmax
    
    @property
    def xcr(self):
        assert self.mdot<0, 'Must be melting'
        h0=self.h0; u0=self.u0; mdot=self.mdot
        n=self.n; A=self.A;
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        lmax = self.lmax
        c = A*(rhoi*g*(rhow-rhoi)/4/rhow)**n
        xcr = lmax*(1-((-mdot+c*h0**(n+1))/((n+1)*c*h0**(n+1)))**(1/(n+1)))
        
        return xcr

    @property
    def xu(self):
        h0=self.h0; u0=self.u0; mdot=self.mdot
        n=self.n; A=self.A;
        rhow=self.rhow; rhoi=self.rhoi; g=self.g
        
        rn = rhoi/rhow/2.
        ucr = self.u(self.xcr)
        def crossing(x):
            return rn*(h0*u0+mdot*self.xcr)/(h0*u0+mdot*x)-(self.u(x)/ucr)**n
        return root(crossing, self.lmax-1e-2)['x'][0]

