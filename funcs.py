
# 1. Standard library imports
from copy import deepcopy
import operator
from types import SimpleNamespace

# 2. External package imports
import holoviews as hv
import kwant
from kwant.continuum.discretizer import discretize
from kwant.digest import uniform
import numpy as np
import scipy.constants

# 3. Internal imports
from combine import combine
from common import *

# Parameters taken from arXiv:1204.2792
# All constant parameters, mostly fundamental constants, in a SimpleNamespace.
constants = SimpleNamespace(
    m_eff=0.015 * scipy.constants.m_e,  # effective mass in kg
    hbar=scipy.constants.hbar,
    m_e=scipy.constants.m_e,
    eV=scipy.constants.eV,
    e=scipy.constants.e,
    c=1e18 / (scipy.constants.eV * 1e-3),  # to get to meV * nm^2
    mu_B=scipy.constants.physical_constants['Bohr magneton in eV/T'][0] * 1e3)

constants.t = (constants.hbar ** 2 / (2 * constants.m_eff)) * constants.c


# Hamiltonian and system definition
@memoize
def discretized_hamiltonian(a, as_lead=False):
    ham = ("(0.5 * hbar**2 * (k_x**2 + k_y**2 + k_z**2) / m_eff * c - mu + V(x, y, z)) * kron(sigma_0, sigma_z) + "
           "alpha * (k_y * kron(sigma_x, sigma_z) - k_x * kron(sigma_y, sigma_z)) + "
           "0.5 * g * mu_B * (B_x * kron(sigma_x, sigma_0) + B_y * kron(sigma_y, sigma_0) + B_z * kron(sigma_z, sigma_0)) + "
           "Delta * kron(sigma_0, sigma_x)")

    lead = {'mu': 'mu_lead'} if as_lead else {}

    subst_sm = {'Delta': 0, **lead}
    subst_sc = {'g': 0, 'alpha': 0, **lead}
    subst_interface = {'c': 'c * c_tunnel', 'alpha': 0, **lead}
    subst_barrier = {'mu': 'mu - V_barrier', 'Delta': 0, **lead}

    templ_sm = discretize(ham, locals=subst_sm, grid_spacing=a)
    templ_sc = discretize(ham, locals=subst_sc, grid_spacing=a)
    templ_interface = discretize(ham, locals=subst_interface, grid_spacing=a)
    templ_barrier = discretize(ham, locals=subst_barrier, grid_spacing=a)

    return templ_sm, templ_sc, templ_interface, templ_barrier


def add_disorder_to_template(template):
    # Only works with particle-hole + spin DOF or only spin.
    template = deepcopy(template)  # Needed because kwant.Builder is mutable
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    s0sz = np.kron(s0, sz)
    norbs = template.lattice.norbs
    mat = s0sz if norbs == 4 else s0

    def onsite_disorder(site, disorder, salt):
        return disorder * (uniform(repr(site), repr(salt)) - .5) * mat

    for site, onsite in template.site_value_pairs():
        onsite = template[site]
        template[site] = combine(onsite, onsite_disorder, operator.add, 1)

    return template


def apply_peierls_to_template(template, xyz_offset=(0, 0, 0)):
    """Adds p.orbital argument to the hopping functions."""
    template = deepcopy(template)  # Needed because kwant.Builder is mutable
    x0, y0, z0 = xyz_offset
    lat = template.lattice
    a = np.max(lat.prim_vecs)  # lattice contant

    def phase(site1, site2, B_x, B_y, B_z, orbital, e, hbar):
        if orbital:
            x, y, z = site1.tag
            direction = site2.tag - site1.tag
            A = [B_y * (z - z0) - B_z * (y - y0), 0, B_x * (y - y0)]
            A = np.dot(A, direction) * a**2 * 1e-18 * e / hbar
            phase = np.exp(-1j * A)
            if lat.norbs == 2:  # No PH degrees of freedom
                return phase
            elif lat.norbs == 4:
                return np.array([phase, phase.conj(), phase, phase.conj()],
                                dtype='complex128')
        else:  # No orbital phase
            return 1

    for (site1, site2), hop in template.hopping_value_pairs():
        template[site1, site2] = combine(hop, phase, operator.mul, 2)
    return template


def get_offset(shape, start, lat):
    a = np.max(lat.prim_vecs)
    coords = [site.pos for site in lat.shape(shape, start)()]
    xyz_offset = np.mean(coords, axis=0)
    return xyz_offset

# Shape functions

def square_sector(r_out, r_in=0, L=1, L0=0, coverage_angle=360, angle=0, a=10):
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
    coverage_angle : ignored
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
            try:
                x, y, z = site.pos
            except AttributeError:
                x, y, z = site
            shape_yz = -r_in <= y < r_in and r_in <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return shape, np.array([L / a - 1, 0, r_in / a + 1], dtype=int)
    else:
        def shape(site):
            try:
                x, y, z = site.pos
            except AttributeError:
                x, y, z = site
            shape_yz = -r_out <= y < r_out and -r_out <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return shape, (int((L - a) / a), 0, 0)


def cylinder_sector(r_out, r_in=0, L=1, L0=0, coverage_angle=360, angle=0, a=10):
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
    coverage_angle : int, optional
        Coverage angle in degrees.
    angle : int, optional
        Angle of tilting from top in degrees.
    a : int, optional
        Discretization constant in nm.

    Returns
    -------
    (shape_func, *(start_coords))
    """
    coverage_angle *= np.pi / 360
    angle *= np.pi / 180
    r_out_sq, r_in_sq = r_out**2, r_in**2

    def shape(site):
        try:
            x, y, z = site.pos
        except AttributeError:
            x, y, z = site
        n = (y + 1j * z) * np.exp(1j * angle)
        y, z = n.real, n.imag
        rsq = y**2 + z**2
        shape_yz = (r_in_sq <= rsq < r_out_sq and
                    z >= np.cos(coverage_angle) * np.sqrt(rsq))
        return (shape_yz and L0 <= x < L) if L > 0 else shape_yz

    r_mid = (r_out + r_in) / 2
    start_coords = np.array([L - a,
                             r_mid * np.sin(angle),
                             r_mid * np.cos(angle)])

    return shape, start_coords


def at_interface(site1, site2, shape1, shape2):
    return ((shape1[0](site1) and shape2[0](site2)) or
            (shape2[0](site1) and shape1[0](site2)))


def change_hopping_at_interface(syst, template, shape1, shape2):
    for (site1, site2), hop in syst.hopping_value_pairs():
        if at_interface(site1, site2, shape1, shape2):
            syst[site1, site2] = template[site1, site2]
    return syst


# System construction

@memoize
def make_3d_wire(a, L, r1, r2, coverage_angle, angle, onsite_disorder,
                 with_leads, with_shell, shape, A_correction):
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
    coverage_angle : int
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
    ...                    L=30, coverage_angle=135, r1=50, r2=70,
                           shape='square', with_leads=True, with_shell=True)
    >>> syst, hopping = make_3d_wire(**syst_params)

    """
    sz = np.array([[1, 0], [0, -1]])
    cons_law = np.kron(np.eye(2), -sz)
    syst = kwant.Builder(conservation_law=cons_law)

    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise(NotImplementedError('Only square or circle wire cross'
                                  'section allowed'))

    shape_normal = shape_function(r_out=r1, angle=angle, L0=a, L=L, a=a)
    shape_barrier = shape_function(r_out=r1, angle=angle, L=a, a=a)
    shape_sc = shape_function(r_out=r2, r_in=r1, coverage_angle=coverage_angle,
                              angle=angle, L0=a, L=L, a=a)

    templ_sm, templ_sc, templ_interface, templ_barrier = discretized_hamiltonian(a)

    templ_sm = apply_peierls_to_template(templ_sm)
    templ_barrier = apply_peierls_to_template(templ_barrier)

    if onsite_disorder:
        templ_sm = add_disorder_to_template(templ_sm)

    syst.fill(templ_sm, *shape_normal)
    syst.fill(templ_barrier, *shape_barrier)

    if with_shell:

        if A_correction:
            lat = templ_sc.lattice
            xyz_offset = get_offset(*shape_sc, lat=lat)
        else:
            xyz_offset = (0, 0, 0)

        templ_sc = apply_peierls_to_template(templ_sc, xyz_offset=xyz_offset)
        syst.fill(templ_sc, *shape_sc)

        # Adding a tunnel barrier between SM and SC
        templ_interface = apply_peierls_to_template(templ_interface)
        syst = change_hopping_at_interface(syst, templ_interface,
                                           shape_normal, shape_sc)

    if with_leads:
        lead = make_lead(a, r1, r2, coverage_angle, angle, A_correction=False,
                         with_shell=False, shape=shape)
        # The lead at the side of the tunnel barrier.
        syst.attach_lead(lead.reversed())
        # The second lead on the other side.
        syst.attach_lead(lead)

    return syst.finalized()


@memoize
def make_lead(a, r1, r2, coverage_angle, angle, A_correction, with_shell, shape):
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
    coverage_angle : int
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

    >>> syst_params = dict(a=10, angle=0, coverage_angle=185, r1=50,
    ...                    r2=70, A_correction=True, shape='square', with_shell=True)
    >>> syst, hopping = make_lead(**syst_params)

    """
    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise NotImplementedError('Only square or circle wire cross section allowed')

    shape_normal_lead = shape_function(r_out=r1, angle=angle, L=-1, a=a)
    shape_sc_lead = shape_function(r_out=r2, r_in=r1, coverage_angle=coverage_angle, angle=angle, L=-1, a=a)

    sz = np.array([[1, 0], [0, -1]])
    cons_law = np.kron(np.eye(2), -sz)
    symmetry = kwant.TranslationalSymmetry((a, 0, 0))
    lead = kwant.Builder(symmetry, conservation_law=cons_law)

    templ_sm, templ_sc, templ_interface, _ = discretized_hamiltonian(a, as_lead=True)
    templ_sm = apply_peierls_to_template(templ_sm)
    lead.fill(templ_sm, *shape_normal_lead)

    if with_shell:
        lat = templ_sc.lattice
        # Take only a slice of SC instead of the infinite shape_sc_lead
        shape_sc = shape_function(r_out=r2, r_in=r1, coverage_angle=coverage_angle, angle=angle, L=a, a=a)

        if A_correction:
            xyz_offset = get_offset(*shape_sc, lat)
        else:
            xyz_offset = (0, 0, 0)

        templ_sc = apply_peierls_to_template(templ_sc, xyz_offset=xyz_offset)
        templ_interface = apply_peierls_to_template(templ_interface)
        lead.fill(templ_sc, *shape_sc_lead)

        # Adding a tunnel barrier between SM and SC
        lead = change_hopping_at_interface(lead, templ_interface,
                                           shape_normal_lead, shape_sc_lead)

    return lead


# Physics functions
@memoize
def andreev_conductance(syst, params, E=100e-3, verbose=False):
    """The Andreev conductance is N - R_ee + R_he."""
    smatrix = kwant.smatrix(syst, energy=E, params=params)
    r_ee = smatrix.transmission((0, 0), (0, 0))
    r_eh = smatrix.transmission((0, 0), (0, 1))
    r_hh = smatrix.transmission((0, 1), (0, 1))
    r_he = smatrix.transmission((0, 1), (0, 0))

    if verbose:
        r_ = {'r_ee': r_ee, 'r_eh': r_eh, 'r_he': r_he, 'r_hh': r_hh}
        for key, val in r_.items():
            print('{val}: {key}'.format(val=val, key=key))

    N_e = smatrix.submatrix((0, 0), (0, 0)).shape[0]

    return {'G_Andreev': N_e - r_ee + r_he,
            'G_01': smatrix.transmission(0, 1)}


def bands(lead, params, ks=None):
    if ks is None:
        ks = np.linspace(-3, 3)

    bands = kwant.physics.Bands(lead, params=params)

    if isinstance(ks, (float, int)):
        return bands(ks)
    else:
        return np.array([bands(k) for k in ks])


def translation_ev(h, t, tol=1e6):
    """Compute the eigenvalues of the translation operator of a lead.

    Adapted from kwant.physics.leads.modes.

    Parameters
    ----------
    h : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    t : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.

    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=r*exp(i*k),
        for |r|=1 they are propagating modes.
    """
    a, b = kwant.physics.leads.setup_linsys(h, t, tol, None).eigenproblem
    ev = kwant.physics.leads.unified_eigenproblem(a, b, tol=tol)[0]
    return ev


def cell_mats(lead, params, bias=0):
    h = lead.cell_hamiltonian(params=params)
    h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params)
    return h, t


@memoize
def gap_minimizer(lead, params, energy):
    """Function that minimizes a function to find the band gap.
    This objective function checks if there are progagating modes at a
    certain energy. Returns zero if there is a propagating mode.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    energy : float
        Energy at which this function checks for propagating modes.

    Returns
    -------
    minimized_scalar : float
        Value that is zero when there is a propagating mode.
    """
    h, t = cell_mats(lead, params, bias=energy)
    ev = translation_ev(h, t)
    norm = (ev * ev.conj()).real
    return np.min(np.abs(norm - 1))


@memoize
def find_gap(lead, params, tol=1e-6):
    """Finds the gapsize by peforming a binary search of the modes with a
    tolarance of tol.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    tol : float
        The precision of the binary search.

    Returns
    -------
    gap : float
        Size of the gap.
    """
    lim = [0, np.abs(bands(lead, params, ks=0)).min()]
    if gap_minimizer(lead, params, energy=0) < 1e-15:
        # No band gap
        gap = 0
    else:
        while lim[1] - lim[0] > tol:
            energy = sum(lim) / 2
            par = gap_minimizer(lead, params, energy)
            if par < 1e-10:
                lim[1] = energy
            else:
                lim[0] = energy
        gap = sum(lim) / 2
    return gap


def get_cross_section(syst, pos, direction):
    coord = np.array([s.pos for s in syst.sites if s.pos[direction] == pos])
    cross_section = np.delete(coord, direction, 1)
    return cross_section


def get_densities(lead, k, params):

    xy = get_cross_section(lead, pos=0, direction=0)
    h, t = lead.cell_hamiltonian(params=params), lead.inter_cell_hopping(params=params)
    h_k = h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)

    vals, vecs = np.linalg.eigh(h_k)
    indxs = np.argsort(np.abs(vals))
    vecs = vecs[:, indxs]
    vals = vals[indxs]
    
    norbs = lat_from_syst(lead).norbs
    densities = np.linalg.norm(vecs.reshape(-1, norbs, len(vecs)), axis=1)**2
    return xy, vals, densities.T


def plot_wfs_in_cross_section(lead, params, k, num_bands=40):
    xy, energies, densities = get_densities(lead, k, params)
    wfs = [kwant.plotter.mask_interpolate(xy, density, oversampling=1)[0]
                                          for density in densities[:num_bands]]
    ims = {E: hv.Image(wf) for E, wf in zip(energies, wfs)}
    return hv.HoloMap(ims, kdims=[hv.Dimension('E', unit='meV')])
