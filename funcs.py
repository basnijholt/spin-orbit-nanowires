# 1. Standard library imports
import operator
from types import SimpleNamespace

# 2. External package imports
import holoviews as hv
import kwant
from kwant.continuum.discretizer import discretize
from kwant.digest import uniform
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, m_e, eV, physical_constants, e
import sympy
from sympy.physics.quantum import TensorProduct as kr

# 3. Internal imports
from combine import combine

# Parameters taken from arXiv:1204.2792
# All constant parameters, mostly fundamental constants, in a SimpleNamespace.
constants = SimpleNamespace(
    m_eff=0.015 * m_e,  # effective mass in kg
    hbar=hbar,
    m_e=m_e,
    eV=eV,
    e=e,
    meV=eV * 1e-3,
    c=1e18 / (eV * 1e-3))  # to get to meV * nm^2

constants.t = (hbar ** 2 / (2 * constants.m_eff)) * constants.c
constants.mu_B = physical_constants['Bohr magneton'][0] / constants.meV


# Hamiltonian and system definition

def discretized_hamiltonian(a):
    sx, sy, sz = [sympy.physics.matrices.msigma(i) for i in range(1, 4)]
    s0 = sympy.eye(2)
    k_x, k_y, k_z = kwant.continuum.momentum_operators
    x, y, z = kwant.continuum.position_operators
    B_x, B_y, B_z, Delta, mu, alpha, g, mu_B, hbar = sympy.symbols(
        'B_x B_y B_z Delta mu alpha g mu_B hbar', real=True)
    m_eff, mu_sc, mu_sm = sympy.symbols(
        'm_eff, mu_sc, mu_sm', commutative=False)
    c, c_tunnel = sympy.symbols('c, c_tunnel')  # c should be (1e18 / constants.meV) if in nm and meV
    kin = (1 / 2) * hbar**2 * (k_x**2 + k_y**2 + k_z**2) / m_eff * c
    ham = ((kin - mu) * kr(s0, sz) +
           alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
           0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
           Delta * kr(s0, sx))

    args = dict(lattice_constant=a)
    subs_sm = [(Delta, 0), (mu, mu_sm)]
    subs_sc = [(g, 0), (alpha, 0), (mu, mu_sc)]
    subs_barrier = [(c, c * c_tunnel), (alpha, 0), (mu, (mu_sc + mu_sm) / 2)]

    templ_sm = discretize(ham.subs(subs_sm), **args)
    templ_sc = discretize(ham.subs(subs_sc), **args)
    templ_barrier = discretize(ham.subs(subs_barrier), **args)
    return templ_sm, templ_sc, templ_barrier


def add_disorder_to_template(template):
    def onsite_dis(site, disorder, salt):
        identity = np.eye(4)
        return disorder * (uniform(repr(site), repr(salt)) - .5) * identity

    for site, onsite in template.site_value_pairs():
        onsite = template[site]
        template[first_site] = combine(onsite, onsite_dis, operator.add, 1)

    return template


def delta(site1, site2):
    """This is still being used in tunnel_hops. Should not depend on
    this to make the algo more robust."""
    return np.argmax(site2.pos - site1.pos)


def phase(site1, site2, B_x, B_y, B_z, orbital, e, hbar):
    x, y, z = site1.tag
    vec = site2.tag - site1.tag
    lat = site1[0]
    a = np.max(lat.prim_vecs)  # lattice_contant
    A = [B_y * z - B_z * y, 0, B_x * y]
    A = np.dot(A, vec) * a**2 * 1e-18 * e / hbar
    phi = np.exp(-1j * A)
    if orbital:
        if lat.norbs == 2:  # No PH degrees of freedom
            return phi
        elif lat.norbs == 4:
            return np.array([phi, phi.conj(), phi, phi.conj()], dtype='complex128')
    else:  # No orbital phase
        return 1


def apply_peierls_to_template(template):
    """Adds p.orbital argument to the hopping functions."""
    for (site1, site2), hop in template.hopping_value_pairs():
        lat = site1[0]
        a = np.max(lat.prim_vecs)
        template[site1, site2] = combine(hop, phase, operator.mul, 2)
    return template


# Shape functions

def square_sector(r_out, r_in=0, L=1, L0=0, phi=360, angle=0, a=10):
    """Returns the shape function and start coords of a wire
    with a square cross section, for -r_out <= x, y < r_out.

    Parameters
    ----------
    r_out : int
        Outer radius in nm.
    r_in : int
        Inner radius in nm.
    L : int
        Length of wire from L0 in nm, -1 if infinite in x-direction.
    L0 : int
        Start position in x.
    phi : ignored
        Ignored variable, to have same arguments as cylinder_sector.
    angle : ignored
        Ignored variable, to have same arguments as cylinder_sector.
    a : int
        Discretization constant in nm.

    Returns
    -------
    (shape_func, *(start_coords))
    """
    r_in /= 2
    r_out /= 2 
    if r_in > 0:
        def shape(site):
            x, y, z = site.pos
            shape_yz = -r_in <= y < r_in and r_in <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return shape, np.array([L / a - 1, 0, r_in / a + 1], dtype=int)
    else:
        def shape(site):
            x, y, z = site.pos
            shape_yz = -r_out <= y < r_out and -r_out <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return shape, (int((L - a) / a), 0, 0)


def cylinder_sector(r_out, r_in=0, L=1, L0=0, phi=360, angle=0, a=10):
    """Returns the shape function and start coords for a wire with
    as cylindrical cross section.

    Parameters
    ----------
    r_out : int
        Outer radius in nm.
    r_in : int, optional
        Inner radius in nm.
    L : int, optional
        Length of wire from L0 in nm, -1 if infinite in x-direction.
    L0 : int, optional
        Start position in x.
    phi : int, optional
        Coverage angle in degrees.
    angle : int, optional
        Angle of tilting from top in degrees.
    a : int, optional
        Discretization constant in nm.

    Returns
    -------
    (shape_func, *(start_coords))
    """
    phi *= np.pi / 360
    angle *= np.pi / 180
    r_out_sq, r_in_sq = r_out**2, r_in**2

    def shape(site):
        x, y, z = site.pos
        n = (y + 1j * z) * np.exp(1j * angle)
        y, z = n.real, n.imag
        rsq = y**2 + z**2
        shape_yz = r_in_sq <= rsq < r_out_sq and z >= np.cos(phi) * np.sqrt(rsq)
        return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
    
    r_mid = (r_out + r_in) / 2
    start_coords = np.array([L - a, 
                             r_mid * np.sin(angle), 
                             r_mid * np.cos(angle)])

    return shape, np.round(start_coords / a).astype(int)

def at_interface(site1, site2, shape1, shape2):
    return ((shape1[0](site1) and shape2[0](site2)) or
            (shape2[0](site1) and shape1[0](site2)))


# System construction

def make_3d_wire(a, L, r1, r2, phi, angle, onsite_disorder,
                 with_leads, with_shell, shape):
    """Create a cylindrical 3D wire covered with a
    superconducting (SC) shell, but without superconductor in 
    leads.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    L : int
        Length of wire (the scattering part with SC shell.)
    r1 : int
        Radius of normal part of wire in nm.
    r2 : int
        Radius of superconductor in nm.
    phi : int
        Coverage angle of superconductor in degrees.
    angle : int
        Angle of tilting of superconductor from top in degrees.
    onsite_disorder : bool
        When True, disorder in SM and requires `disorder` and `salt` aguments.
    with_leads : bool
        If True it adds infinite semiconducting leads.
    with_shell : bool
        Adds shell to the scattering area. If False no SC shell is added and
        only a cylindrical or square wire will be created.
    shape : str
        Either `circle` or `square` shaped cross section.

    Returns
    -------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(a=10, angle=0, site_disorder=False,
    ...                    L=30, phi=185, r1=50, r2=70, shape='square',
    ...                    with_leads=True, with_shell=True)
    >>> syst, hopping = make_3d_wire(**syst_params)

    """
    templ_sm, templ_sc, templ_barrier = map(apply_peierls_to_template, discretized_hamiltonian(a))
    symmetry = kwant.TranslationalSymmetry((a, 0, 0))
    syst = kwant.Builder()

    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise(NotImplementedError('Only square or circle wire cross section allowed'))

    shape_normal = shape_function(r_out=r1, angle=angle, L=L, a=a)
    shape_normal_lead = shape_function(r_out=r1, angle=angle, L=-1, a=a)
    shape_sc = shape_function(r_out=r2, r_in=r1, phi=phi, angle=angle, L0=0, L=L, a=a)

    if onsite_disorder:
        templ_sm = add_disorder_to_template(templ_sm)

    syst.fill(templ_sm, *shape_normal)

    if with_shell:
        syst.fill(templ_sc, *shape_sc)
        
        # Adding a tunnel barrier between SM and SC
        tunnel_hops = {delta(s1, s2): hop for
                       (s1, s2), hop in templ_barrier.hopping_value_pairs()}
        for (site1, site2), hop in syst.hopping_value_pairs():
            if at_interface(site1, site2, shape_normal, shape_sc):
                syst[site1, site2] = tunnel_hops[delta(site1, site2)]

    if with_leads:
        sz = np.array([[1, 0], [0, -1]])
        cons_law = np.kron(np.eye(2), -sz)
        lead = kwant.Builder(symmetry, conservation_law=cons_law)
        lead.fill(templ_sm, *shape_normal_lead)
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())
    return syst.finalized()


def make_lead(a, r1, r2, phi, angle, with_shell, shape):
    """Create an infinite cylindrical 3D wire partially covered with a
    superconducting (SC) shell.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    r1 : int
        Radius of normal part of wire in nm.
    r2 : int
        Radius of superconductor in nm.
    phi : int
        Coverage angle of superconductor in degrees.
    angle : int
        Angle of tilting of superconductor from top in degrees.
    with_shell : bool
        Adds shell to the scattering area. If False no SC shell is added and
        only a cylindrical or square wire will be created.
    shape : str
        Either `circle` or `square` shaped cross section.

    Returns
    -------
    syst : kwant.builder.InfiniteSystem
        The finilized kwant system.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(a=10, angle=0, phi=185, r1=50,
    ...                    r2=70, shape='square', with_shell=True)
    >>> syst, hopping = make_lead(**syst_params)

    """
    templ_sm, templ_sc, templ_barrier = map(apply_peierls_to_template, discretized_hamiltonian(a))
    symmetry = kwant.TranslationalSymmetry((a, 0, 0))

    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise(NotImplementedError('Only square or circle wire cross section allowed'))

    shape_normal_lead = shape_function(r_out=r1, angle=angle, L=-1, a=a)
    shape_sc_lead = shape_function(r_out=r2, r_in=r1, phi=phi, angle=angle, L=-1, a=a)

    lead = kwant.Builder(symmetry)
    lead.fill(templ_sm, *shape_normal_lead)
    if with_shell:
        lead.fill(templ_sc, *shape_sc_lead)
        
        # Adding a tunnel barrier between SM and SC
        tunnel_hops = {delta(s1, s2): hop for
                       (s1, s2), hop in templ_barrier.hopping_value_pairs()}
        for (site1, site2), hop in lead.hopping_value_pairs():
            if at_interface(site1, site2, shape_normal_lead, shape_sc_lead):
                lead[site1, site2] = tunnel_hops[delta(site1, site2)]

    return lead.finalized()


# Physics functions

def andreev_conductance(syst, params, E=100e-3, verbose=False):
    """Conductance is N - R_ee + R_he"""
    smatrix = kwant.smatrix(syst, energy=E, params=params)
    r_ee = smatrix.submatrix((0, 0), (0, 0))
    r_eh = smatrix.submatrix((0, 0), (0, 1))
    r_hh = smatrix.submatrix((0, 1), (0, 1))
    r_he = smatrix.submatrix((0, 1), (0, 0))
    c = lambda r: np.trace(r @ r.T.conj()).real

    if verbose:
        r_ = {'r_ee': c(r_ee), 'r_eh': c(r_eh), 'r_he': c(r_he), 'r_hh': c(r_hh)}
        for key, val in r_.items():
            print('{val}: {key}'.format(val=val, key=key))

    N_e = r_ee.shape[0]

    return N_e - c(r_ee) + c(r_he)


def bands(lead, params, ks=None):
    if ks is None:
        ks = np.linspace(-3, 3)
        
    bands = kwant.physics.Bands(lead, params=params)
    return np.array([bands(k) for k in ks])


def modes(h_cell, h_hop, tol=1e6):
    """Compute the eigendecomposition of a translation operator of a lead.
    Adapted from kwant.physics.leads.modes such that it returns the eigenvalues.
    Parameters:
    ----------
    h_cell : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    h_hop : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.
    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=exp(i*k).
    """
    m = h_hop.shape[1]
    n = h_cell.shape[0]

    if (h_cell.shape[0] != h_cell.shape[1] or
            h_cell.shape[0] != h_hop.shape[0]):
        raise ValueError("Incompatible matrix sizes for h_cell and h_hop.")

    # Note: np.any(h_hop) returns (at least from numpy 1.6.1 - 1.8-devel)
    #       False if h_hop is purely imaginary
    if not (np.any(h_hop.real) or np.any(h_hop.imag)):
        v = np.empty((0, m))
        return (kwant.physics.PropagatingModes(np.empty((0, n)), np.empty((0,)),
                                               np.empty((0,))),
                kwant.physics.StabilizedModes(np.empty((0, 0)),
                                              np.empty((0, 0)), 0, v))

    # Defer most of the calculation to helper routines.
    matrices, v, extract = kwant.physics.leads.setup_linsys(
        h_cell, h_hop, tol, None)
    ev = kwant.physics.leads.unified_eigenproblem(*(matrices + (tol,)))[0]

    return ev


def find_gap(lead, params, E_bias=0, tol=1e-3):
    """Finds the gapsize by peforming a binary search of the modes with a
    tolarance of tol.
    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    tol : float
        The precision of the binary search.

    Returns:
    --------
    gap : float
        Size of the gap.
    """
    def gap_minimizer(lead, params, energy):
        """Function that minimizes a function to find the band gap.
        This objective function checks if there are progagating modes at a
        certain energy. Returns zero if there is a propagating mode.
        Parameters:
        -----------
        energy : float
            Energy at which this function checks for propagating modes.
        h0 : numpy array
            Onsite Hamiltonian, syst.cell_hamiltonian(params=params)
        t0 : numpy array
            Hopping matrix, syst.inter_cell_hopping(params=params)
        Returns:
        --------
        minimized_scalar : float
            Value that is zero when there is a propagating mode.
        """
        h, t = lead.cell_hamiltonian(params=params), lead.inter_cell_hopping(params=params)
        ev = modes(h - energy * np.identity(len(h)), t)
        norm = (ev * ev.conj()).real
        return np.sort(np.abs(norm - 1))[0]

    bands = kwant.physics.Bands(lead, params=params)
    band_k_0 = np.abs(bands(k=0)).min()
    lim = [0, band_k_0]
    if gap_minimizer(lead, params, energy=0) < 1e-15:
        gap = 0
    else:
        while lim[1] - lim[0] > tol:
            energy = sum(lim) / 2.
            par = gap_minimizer(lead, params, energy)
            if par < 1e-10:
                lim[1] = energy
            else:
                lim[0] = energy
        gap = sum(lim) / 2.
    return gap