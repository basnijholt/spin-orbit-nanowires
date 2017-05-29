import hpc05
from time import sleep
import sys
client = hpc05.Client(profile='pbs3', timeout=60)
print("Connected to hpc05")
sleep(2)
dview = client[:]
dview.use_dill()
lview = client.load_balanced_view()
print('Connected to {} engines.'.format(len(dview)))

if len(dview) < 100:
    print(len(dview))
    sys.exit(1)

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

print(kwant.__version__)

params = dict(Delta=60, c_tunnel=5/8, V=lambda x: 0, **funcs.constants.__dict__)

syst_pars = dict(a=10, onsite_disorder=False,
                 L=2000, coverage_angle=135, r1=50, r2=70, shape='circle',
                 with_leads=True, with_shell=True)

def cond(val, syst_pars=syst_pars, params=params):
    import funcs
    val['mu_lead'] = val['mu']
    val['B_x'], val['B_y'], val['B_z'] = funcs.spherical_coords(val['B'],
                                                                val['theta'],
                                                                val['phi'])

    params = dict(**params, **val)

    for x in ['angle', 'A_correction']:
        syst_pars[x] = params.pop(x)

    syst = funcs.make_3d_wire(**syst_pars)
    return dict(**funcs.andreev_conductance(syst, params, E=val['V_bias']), **val)



vals = funcs.named_product(B=np.linspace(0, 2, 51),
                           theta=[0, 90],
                           phi=[0, 90],
                           V_bias=np.linspace(-0.25, 0.25, 51),
                           V_barrier=[15],
                           g=[0, 50],
                           alpha=[0, 20],
                           orbital=[False, True],
                           mu=[5, 10, 15, 20],
                           angle=[0],
                           A_correction=[True, False])

vals = [val for val in vals if (not (val['theta'] == 0 and val['phi'] != 0)) and
                               (not (val['g'] == 0 and val['orbital'] == False))]

print(len(vals))

for i, chunk in enumerate(partition_all(20000, vals)):
    fname = 'conductance_{:03d}_sc_correction.hdf'.format(i)
    if not os.path.exists(fname):
        G = lview.map_async(cond, chunk)
        G.wait_interactive()
        G = G.result()

        df = pd.DataFrame(G)

        to_save_params = {k: v for k, v in params.items() if k is not callable(k)}
        df = df.assign(**syst_pars).assign(**to_save_params)
        df = df.assign(git_hash=funcs.get_git_revision_hash())
        df.to_hdf(fname, 'all_data', mode='w')