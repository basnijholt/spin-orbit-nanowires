"""Common functions for doing stuff."""

import collections
import functools
from itertools import product
import subprocess

import kwant
import numpy as np


def lat_from_syst(syst):
    lats = set(s.family for s in syst.sites)
    if len(lats) > 1:
        raise Exception('No unique lattice in the system.')
    return list(lats)[0]


def unique_rows(coor):
    coor_tuple = [tuple(x) for x in coor]
    unique_coor = sorted(set(coor_tuple), key=lambda x: coor_tuple.index(x))
    return np.asarray(unique_coor)


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


def named_product(**items):
    names = items.keys()
    vals = items.values()
    return [dict(zip(names, res)) for res in product(*vals)]


def get_git_revision_hash():
    """Get the git hash to save with data to ensure reproducibility."""
    git_output = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return git_output.decode("utf-8").replace('\n', '')


def find_nearest(array, value):
    """Find the nearest value in an array to a specified `value`."""
    idx = np.abs(np.array(array) - value).argmin()
    return array[idx]


def remove_unhashable_columns(df):
    df = df.copy()
    for col in df.columns:
        if not hashable(df[col].iloc[0]):
            df.drop(col, axis=1, inplace=True)
    return df


def hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def drop_constant_columns(df):
    """Taken from http://stackoverflow.com/a/20210048/3447047"""
    df = remove_unhashable_columns(df)
    df = df.reset_index(drop=True)
    return df.loc[:, (df != df.ix[0]).any()]


def sparse_diag(matrix, k, sigma, **kwargs):
    """Call sla.eigsh with mumps support.

    Please see scipy.sparse.linalg.eigsh for documentation.
    """
    class LuInv(sla.LinearOperator):

        def __init__(self, matrix):
            instance = kwant.linalg.mumps.MUMPSContext()
            instance.analyze(matrix, ordering='pord')
            instance.factor(matrix)
            self.solve = instance.solve
            sla.LinearOperator.__init__(self, matrix.dtype, matrix.shape)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    opinv = LuInv(matrix - sigma * identity(matrix.shape[0]))
    return sla.eigsh(matrix, k, sigma=sigma, OPinv=opinv, **kwargs)


def sort_eigvals(eigvals, eigvecs):
    eigvals, eigvecs = map(np.asarray, [eigvals, eigvecs])
    eigvals_sorted = np.copy(eigvals)
    eigvecs_sorted = np.copy(eigvecs)
    not_swapped = collections.defaultdict(list)

    for i in range(eigvals.shape[0] - 1):
        overlap = np.abs(eigvecs_sorted[i].conj().T @ eigvecs_sorted[i+1])
        idx_max_overlap = overlap.argmax(axis=1)
        max_overlap = overlap.max(axis=1)

        swap_idx = []
        for j, found_overlap in enumerate(max_overlap > 1/np.sqrt(2)):
            if found_overlap:
                swap_idx.append(j)
            else:
                not_swapped[j].append(i+1)

        # Swap all the values that have an overlap > 1/√2
        eigvals_sorted[i+1, swap_idx] = eigvals_sorted[i+1, idx_max_overlap[swap_idx]]
        eigvecs_sorted[i+1, :, swap_idx] = eigvecs_sorted[i+1, :, idx_max_overlap[swap_idx]]

    # See how many points were not swapped
    N = sum(len(l) for l in not_swapped.values()) + len(not_swapped)

    # Add N rows with NaN
    eigvals_sorted = np.pad(eigvals_sorted, ((0, 0), (0, N)),
                            mode='constant', constant_values=(np.nan, np.nan))

    # Some eigvals had no matching eigvecs to swap with, these
    # are in `not_swapped`.
    # For example we have this discontineous matrix with jumps in two places:
    # eigvals_sorted = [0, 0, 1, 1, 0, 0]
    # So we have three bands, therefore extend we `eigvals_sorted`
    # with two rows of NaNs.
    # We get:
    # eigvals_sorted = [[0, 0, 1, 1, 0, 0],
    #                   [nan, nan, nan, nan, nan, nan],
    #                   [nan, nan, nan, nan, nan, nan]]
    # we want to change this to
    # eigvals_sorted = [[0, 0, nan, nan, nan, nan],
    #                   [nan, nan, 1, 1, nan, nan],
    #                   [nan, nan, nan, nan, 0, 0]]

    swap_j = eigvals.shape[1] + 1
    for i, (swap_i, ks) in enumerate(not_swapped.items()):
        ks = ks + [None]
        for j in range(len(ks) - 1):
            swap_j += 1
            row = eigvals_sorted[ks[j]:ks[j+1]]
            A, B = row[:, swap_i].copy(), row[:, swap_j].copy()
            row[:, swap_i], row[:, swap_j] = B, A

    return eigvals_sorted


def spherical_coords(r, theta, phi, degrees=True):
    """Transform spherical coordinates to Cartesian.

    Parameters
    ----------
    r, theta, phi : float or array
        radial distance, polar angle θ, azimuthal angle φ.
    degrees : bool
        Degrees when True, radians when False.
    """
    r, theta, phi = [np.reshape(x, -1) if np.isscalar(x)
                     else x for x in (r, theta, phi)]

    if degrees:
        theta, phi = map(np.deg2rad, (theta, phi))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) + 0 * phi
    return np.array([x, y, z]).T


def cartesian_coords(x, y, z, degree=True):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x)

    if degree:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)

    return r, theta, phi


def spherical_coords_vec(rs, thetas, phis, degrees=True, unique=False):
    """Spherical coordinates to Cartesian, combinations of the arguments.

    Parameters
    ----------
    rs, thetas, phis : numpy array
        radial distance, polar angle θ, azimuthal angle φ.
    degrees : bool
        Degrees when True, radians when False.
    unique : bool
        Only return the unique combinations of coordinates. Useful e.g.
        when theta=0 or to avoid double values.
    """
    rs = np.reshape(rs, (-1, 1, 1))
    thetas = np.reshape(thetas, (1, -1, 1))
    phis = np.reshape(phis, (1, 1, -1))
    vec = spherical_coords(rs, thetas, phis, degrees).reshape(-1, 3)
    if unique:
        vec = unique_rows(vec)
    return vec.round(15)
