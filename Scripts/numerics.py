"""
Definitions of DG flux and source term
"""

__author__ = "Jeremy N. Bassis (jbassis@umich.edu)"
__copyright__ = "Copyright (C) 2019 %s" % __author__

from dolfin import *

def flux(phi, q, n):
    """
    Define the Discontinuous Galerkin form of the
    flux

    q is the flux
    phi is Trial function

    n is the cell normal
    """

    # Define `upwind' flux`
    #qn = abs(dot(q('+'), n('+')))
    qn = 0.5*(dot(q, n) + abs(dot(q, n)))

    # Integral of values over each cell
    a_cell = dot(q,grad(phi))*dx

    # Inflow and outflow boundary condition (should mark domain)
    #a_outflow = -phi*dot(q, n)*ds


    a_outflow = - conditional(dot(q, n) > 0, phi*dot(q, n), 0.0)*ds

    #a_outflow = -qn*phi*ds
    #a_inflow =   phi*dot(q, n)*ds(2)

    # Jump condition across interface between interior cells
    a_jump = - (phi('+') - phi('-'))*(qn('+') - qn('-'))*dS

    a = a_cell + a_outflow + a_jump

    return a

def mass_matrix(phi,v,dtc):
    """
    Define consistent mass matrix for
    Forward Euler time step

    phi is Trial function
    v is the Test function
    """
    return dot(phi,v)/dtc*dx

def source(phi,f):
    return dot(f,phi)*dx#(metadata={'quadrature_degree': 2})


def diffusion(phi, v, kappa, n, h, alpha=1.0,theta=1.0):
    """
    Diffusion used for eddy viscosity

    phi is Trial function
    v is the Test function

    kappa is the eddy diffusivity

    alpha is a regularization constant??
    n is the cell normal
    h is the cell size
    """

    # Contribution from the cells
    a_cell = kappa*dot(grad(phi), grad(v))*dx

    # Contribution from the interior facets
    a_int0 = theta*kappa('+')*alpha('+')/h('+')*dot(jump(v, n), jump(phi, n))*dS
    a_int1 = - theta*kappa('+')*dot(avg(grad(v)), jump(phi, n))*dS
    a_int2 = - theta*kappa('+')*dot(jump(v, n), avg(grad(phi)))*dS
    a_int = a_int0 + a_int1 + a_int2

    # Contribution from the exterior facets?

    a = a_cell + a_int


     # Bilinear form
    #a_int = dot(grad(v), kappa*grad(phi))*dx

    #a_fac = (alpha/avg(h))*dot(jump(kappa*v, n), jump(phi, n))*dS \
    #        - dot(avg(grad(kappa*v)), jump(phi, n))*dS \
    #        - dot(jump(kappa*v, n), avg(grad(phi)))*dS


    #a = a_int + a_fac
    return a
