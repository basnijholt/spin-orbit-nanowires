
import hpc05
client = hpc05.Client(profile='pbs3', timeout=60)
print("Connected to hpc05")
from time import sleep
sleep(2)
dview = client[:]
dview.use_dill()
lview = client.load_balanced_view()
print('Connected to {} engines.'.format(len(dview)))
while len(dview) < 100:
    sleep(2)
    print(len(dview))
get_ipython().magic("px import sys, os; sys.path.append(os.path.expanduser('~/Work/induced_gap_B_field/'))")


# 1. Standard library imports
import os.path

# 2. External package imports
import holoviews as hv
import kwant
import numpy as np
import pandas as pd
from toolz import partition_all

# 3. Internal imports
import funcs
from funcs import constants

print(kwant.__version__)

params = dict(Delta=60, V=0, c_tunnel=5/8, **constants.__dict__)

syst_pars = dict(a=10, onsite_disorder=False,
                 L=2000, phi=135, r1=50, r2=70, shape='circle',
                 with_leads=True, with_shell=True)

def cond(val, syst_pars=syst_pars, params=params):
    import funcs
    val['B_x'], val['B_y'], val['B_z'] = val.pop('B')
    params = dict(**params, **val)
    params['mu_lead'] = val['mu']
    
    for x in ['angle']:
        syst_pars[x] = params.pop(x)

    syst = funcs.make_3d_wire(**syst_pars)
    return dict(**funcs.andreev_conductance(syst, params, E=val['V_bias']), **params)


Bs = funcs.spherical_coords_vec(rs=np.linspace(0, 2, 51),
                                thetas=[0, 90],
                                phis=[0, 90])

vals = funcs.named_product(B=Bs,
                           V_bias=np.linspace(-0.25, 0.25, 51),
                           V_barrier=[15],
                           g=[0, 50],
                           alpha=[0, 20],
                           orbital=[False, True],
                           mu=[5, 10, 15, 20],
                           angle=[0])
print(len(vals))

for i, chunk in enumerate(partition_all(200000, vals)):
    fname = 'conductance_{:03d}_sc_correction.hdf'.format(i)
    if not os.path.exists(fname):
        G = lview.map_async(cond, chunk)
        G.wait_interactive()
        G = G.result()

        df = pd.DataFrame(G)

        df = df.assign(**syst_pars)
        df = df.assign(git_hash=funcs.get_git_revision_hash())
        df.to_hdf(fname, 'all_data', mode='w')