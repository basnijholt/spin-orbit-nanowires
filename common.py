"""Common functions for doing stuff."""

import asyncio
from copy import copy
import collections
from datetime import timedelta
import functools
from glob import glob
import gzip
from itertools import product
import os
import pickle
import time
import subprocess
import sys

import kwant
import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla
import scipy.optimize
from scipy.sparse import identity
from skimage import measure

assert sys.version_info >= (3, 6), 'Use Python ≥3.6'


def run_simulation(lview, func, vals, parameters, fname_i, N=None,
                   overwrite=False):
    """Run a simulation where one loops over `vals`. The simulation
    yields len(vals) results, but by using `N`, you can split it up
    in parts of length N.

    Parameters
    ----------
    lview : ipyparallel.client.view.LoadBalancedView object
        LoadBalancedView for asynchronous map.
    func : function
        Function that takes a list of arguments: `vals`.
    vals : list
        Arguments for `func`.
    parameters : dict
        Dictionary that is saved with the data, used for constant
        parameters.
    fname_i : str
        Name for the resulting HDF5 files. If the simulation is
        split up in parts by using the `N` argument, it needs to
        be a formatteble string, for example 'file_{}'.
    N : int
        Number of results in each pandas.DataFrame.
    overwrite : bool
        Overwrite the file even if it already exists.
    """
    if N is None:
        N = 1000000
        if len(vals) > N:
            raise Exception('You need to split up vals in smaller parts')

    N_files = len(vals) // N + (0 if len(vals) % N == 0 else 1)
    print('`vals` will be split in {} files.'.format(N_files))
    time_elapsed = 0
    parts_done = 0

    parameters_new = {}
    for k, v in parameters.items():
        if callable(v):
            warnings.warn('parameters["{}"] is a function and is not saved!'.format(k))
        else:
            parameters_new[k] = v
    parameters = parameters_new

    if N < len(vals) and fname_i.format('1') == fname_i.format('2'):
        raise Exception('Use a formattable string for `fname_i`.')

    for i, chunk in enumerate(partition_all(N, vals)):
        fname = fname_i.replace('{}', '{:03d}').format(i)
        print('Busy with file: {}.'.format(fname))
        if not os.path.exists(fname) or overwrite:
            map_async = lview.map_async(func, chunk)
            map_async.wait_interactive()
            result = map_async.result()
            df = pd.DataFrame(result)

            common_keys = common_elements(df.columns, parameters.keys())
            if common_keys:
                raise Exception('Parameters in both function result and function input',
                                ': {}'.format(common_keys))
            else:
                df = df.assign(**parameters)

            #df = df.assign(git_hash=get_git_revision_hash())
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            df.to_hdf(fname, 'all_data', mode='w', complib='zlib', complevel=9)

            # Print useful information
            N_files_left = N_files - (i + 1)
            parts_done += 1
            time_elapsed += map_async.elapsed
            time_left = timedelta(seconds=(time_elapsed / parts_done) *
                                  N_files_left)
            print_str = ('Saved {}, {} more files to go, {} time left '
                         'before everything is done.')
            print(print_str.format(fname, N_files_left, time_left))
        else:
            print('File: {} was already done.'.format(fname))


def common_elements(list1, list2):
    return [element for element in list1 if element in list2]


def parse_params(params):
    for k, v in params.items():
        if isinstance(v, str):
            try:
                params[k] = eval(v)
            except NameError:
                pass
    return params


def combine_dfs(pattern, fname=None):
    files = glob(pattern)
    df = pd.concat([pd.read_hdf(f) for f in sorted(files)])
    df = df.reset_index(drop=True)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        df.to_hdf(fname, 'all_data', mode='w', complib='zlib', complevel=9)

    return df


def lat_from_syst(syst):
    lats = set(s.family for s in syst.sites)
    if len(lats) > 1:
        raise Exception('No unique lattice in the system.')
    return list(lats)[0]


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


def sort_spectrum(energies, psis):
    psi = psis[:][0]
    e = energies[0]
    sorted_levels = [e]
    for i in range(len(energies) - 1):
        e2, psi2 = energies[i+1], psis[:][i+1]
        perm, line_breaks = best_match(psi, psi2)
        e2 = e2[perm]
        intermediate = (e + e2) / 2
        intermediate[line_breaks] = None
        psi = psi2[:, perm]
        e = e2

        sorted_levels.append(intermediate)
        sorted_levels.append(e)
    sorted_levels = np.array(sorted_levels)

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

    NaNs = np.isnan(sorted_levels)
    N = np.count_nonzero(NaNs)
    levels_padded = np.pad(sorted_levels, ((0, 0), (0, N)), mode='constant',
                           constant_values=(np.nan, np.nan))
    swaps = collections.defaultdict(list)
    for i, j in np.argwhere(NaNs):
        swaps[j].append(i)

    swap_to = sorted_levels.shape[1]
    for swap_from, xs in swaps.items():
        xs = xs + [None]
        for j in range(len(xs) - 1):
            row = levels_padded[xs[j]:xs[j+1]]
            A, B = row[:, swap_from].copy(), row[:, swap_to].copy()
            row[:, swap_from], row[:, swap_to] = B, A
            swap_to += 1

    return levels_padded[::2]


def best_match(psi1, psi2, threshold=None):
    """Find the best match of two sets of eigenvectors.

    Parameters:
    -----------
    psi1, psi2 : numpy 2D complex arrays
        Arrays of initial and final eigenvectors.
    threshold : float, optional
        Minimal overlap when the eigenvectors are considered belonging to the same band.
        The default value is :math:`1/sqrt(2N)`, where :math:`N` is the length of each eigenvector.

    Returns:
    --------
    sorting : numpy 1D integer array
        Permutation to apply to ``psi2`` to make the optimal match.
    diconnects : numpy 1D bool array
        The levels with overlap below the ``threshold`` that should be considered disconnected.
    """
    if threshold is None:
        threshold = (2 * psi1.shape[1])**-0.5
    Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
    orig, perm = scipy.optimize.linear_sum_assignment(-Q)
    return perm, Q[orig, perm] < threshold



def unique_rows(coor):
    coor_tuple = [tuple(x) for x in coor]
    unique_coor = sorted(set(coor_tuple), key=lambda x: coor_tuple.index(x))
    return np.asarray(unique_coor)


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
    xyz = np.array([x, y, z]).T
    return xyz.round(15).squeeze()


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
    return vec


def add_direction(row):
    from copy import copy
    d = copy(row)
    xyz = spherical_coords(1, row.pop('theta'), row.pop('phi'))
    if np.any(np.count_nonzero(xyz) > 1):
        raise Exception('Cannot determine direction. Only fields in purley B_x, B_y, or B_z can be used.')
    row['direction'] = np.argmax(xyz)
    return row


def save_DataSaver_extra_data(learner, N=10000, folder='tmp'):
    os.makedirs(folder, exist_ok=True)
    for i, chunk in enumerate(partition_all(N, learner.extra_data.items())):
        with gzip.open(f'{folder}/extra_data_{i:04d}.pickle', 'wb') as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_DataSaver_extra_data(learner, folder='tmp'):
    extra_data = []
    for fname in sorted(glob(f'{folder}/extra_data_*')):
        with gzip.open(fname, 'rb') as f:
            extra_data += pickle.load(f)
    learner.extra_data = collections.OrderedDict(extra_data)


def gaussian(x, a, mu, sigma):
    factor = a  #/ (sigma * np.sqrt(2 * np.pi))
    return factor * np.exp(-0.5 * (x - mu)**2 / sigma**2)


def loss(ip):
    from adaptive.learner.learner2D import deviations, areas
    A = np.sqrt(areas(ip))
    dev = deviations(ip)[0]
    loss = A * dev + 0.3*A**2
    if len(ip.values) < 2000:
        loss *= 100
    return loss


def get_contours_from_image(image):
    data = np.rot90(image.data, 3)
    lbrt = image.bounds.lbrt()
    contours = measure.find_contours(data, 0.5)
    dx = (lbrt[2] - lbrt[0]) / len(data)
    dy = (lbrt[3] - lbrt[1]) / len(data)
    for c in contours:
        c[:, 0] = c[:, 0] * dx + lbrt[0]
        c[:, 1] = c[:, 1] * dy + lbrt[1]
    return contours


def select_keys(d, keys):
    return {k: v for k, v in d.items() if k in keys}


def smooth_bump(params, pot_params):
    """A modified Gaussian that starts at y=V_l and ends at y=V_r.

    Parameters
    ----------
    params : dict
        With keys `sigma`, `V_l`, and `V_r`
    pot_params : dict
        With keys `x0` and `V_0`. This dict is obtained by
        calling `get_smooth_bump_params`.

    Returns
    -------
    V : function
        Function of position `x`.
    """
    sigma = params['sigma']
    V_l = params['V_l']
    V_r = params['V_r']
    V_0 = pot_params['V_0']
    x0 = pot_params['x0']
    V = lambda x: (
        gaussian(x, V_0, x0, sigma)
        + V_l + (V_r - V_l) * 0.5 * (1 + np.tanh((x - x0) / sigma))
    )
    return V


@memoize
def get_smooth_bump_params(params):
    """Get the parameters for the `smooth_bump` function.
    
    Parameters
    ----------
    params : dict
        With keys `sigma`, `V_l`, `V_r`, `x0`, and `V_0_top`.
    
    Returns
    -------
    smooth_bump_params : dict
        A dictionary with `V_0` and `x0`, which is needed for
        the `smooth_bump` function.

    Notes
    -----
    This awesome plot indicates the parameters.
    +------------------------------------------------+
    |             x0                                 |
    |        |----------|                            |
    |                   ..            __             |
    |                  .  .           |              |
    |                 .    .........  | V_0  ___     |
    |                .                |       |      |
    |    ___ ........                 __      |      |
    |     |         |---|                     | V_r  |
    | V_l |         sigma                     |      |
    |    -- %                                ---     |
    |       |                                        |
    |   (% indicated the origin, (x=0, y=0))         |
    +------------------------------------------------+
    """
    def minimizer(V_0, params, pot_params):
        pot_params['V_0'] = V_0
        V = smooth_bump(params, pot_params)
        f_min = lambda x: params['V_0_top'] + params['V_r'] - V(x)
        op = scipy.optimize.minimize(f_min, pot_params['x0'])
        return op
    pot_params = {'x0': params['x0']}
    V_0 = scipy.optimize.minimize(
            lambda x: abs(minimizer(x, params, pot_params).fun),
            x0=params['V_0_top']).x[0]
    x0 = minimizer(V_0, params, pot_params).x[0]
    x0 = 2*pot_params['x0'] - x0
    return {'V_0': V_0, 'x0': x0}
