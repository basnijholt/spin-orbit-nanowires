#!/usr/bin/env python

import time
import os.path

from hpc05 import Client
from hpc05.utils import (start_ipcluster,
                         start_remote_ipcluster,
                         kill_remote_ipcluster)
from ipyparallel.error import NoEnginesRegistered

def connect_ipcluster(n, profile='pbs', folder=None, timeout=60):
    client = Client(profile=profile, timeout=timeout)
    print("Connected to hpc05")
    print(f'Initially connected to {len(client)} engines.')
    time.sleep(2)
    try:
        dview = client[:]
    except NoEnginesRegistered:
        # This can happen, we just need to wait a little longer.
        pass


    t_start = time.time()
    done = len(client) == n
    while not done:
        dview = client[:]
        done = len(client) == n
        if time.time() - t_start > timeout:
            raise Exception(f'Not all connected after {timeout} seconds.')
        time.sleep(1)

    print(f'Connected to all {len(client)} engines.')
    dview.use_dill()
    lview = client.load_balanced_view()

    if folder is None:
        folder = os.path.dirname(os.path.realpath(__file__))

    print(f'Adding {folder} to path.')
    get_ipython().magic(f"px import sys, os; sys.path.append(os.path.expanduser('{folder}'))")

    return client, dview, lview


def start_and_connect(n, profile='pbs', folder=None):
    start_ipcluster(n, profile)
    return connect_ipcluster(n, profile, folder)


def start_remote_and_connect(n, profile='pbs', folder=None, hostname='hpc05',
                             username=None, password=None,
                             del_old_ipcluster=True):
    if del_old_ipcluster:
        kill_remote_ipcluster(hostname, username, password)

    start_remote_ipcluster(n, profile, hostname, username, password)
    time.sleep(2)
    return connect_ipcluster(n, profile, folder)
