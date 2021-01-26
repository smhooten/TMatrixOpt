"""
TMatrixOpt: A fast and modular transfer-matrix optimization package 
for 1D optical devices.
Copyright (C) 2021 Sean Hooten & Zunaid Omair

TMatrixOpt/parallel_toolkit.py

ATTN: This module contains redistributed and modified code (under the
GNU General Public License, Version 3.0) from EMopt, 
copyright Andrew Michaels, which may be found here:
    https://github.com/anstmichaels/emopt.git

Module that provides useful tools for implementing parallel MPI operations.
"""

__author__ = 'Sean Hooten'
__version__ = '1.0'
__license__ = 'GPL 3.0'

import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def parallel_partition(num_sims):
    """
    Returns curated 1D partition for each processor given the total number of
    simulations required (emphasis on *total*, not the number of simulations that
    each node will run individually)
    """
    simdisc = int(num_sims / SIZE)
    extra = int(num_sims % SIZE)

    partition = list(range(simdisc*RANK, simdisc*(RANK+1)))

    if RANK < extra:
        partition.append(simdisc*(RANK+1))

    partition = [x+RANK if RANK<extra else x+extra for x in partition]

    return(partition)

def parallel_partition2(rank, num_sims):
    """
    Same function as previous, but allows rank as an argument. This can be
    used to create partitions on the head node only before being sent
    to the worker nodes (reduces overall memory usage at potential cost of some
    computational speed).
    """
    simdisc = int(num_sims / SIZE)
    extra = int(num_sims % SIZE)

    partition = list(range(simdisc*rank, simdisc*(rank+1)))

    if rank < extra:
        partition.append(simdisc*(rank+1))

    partition = [x+rank if rank<extra else x+extra for x in partition]

    return(partition)

def run_on_master(func):
    """
    AUTHOR CREDIT: Andrew Michaels
    See header above

    Function decorator that will cause the function to only run on the
    head node, return None on the worker nodes.
    """
    def wrapper(*args, **kwargs):
        if(RANK == 0):
            return func(*args, **kwargs)
        else:
            return

    return(wrapper)

def parallel_integral(integrand, x, axis):
    """
    Run a 1D integral in parallel using np.trapz (works with arbitrarily
    shaped integrands, but an integration axis must be provided).
    The overhead of this calculation is slow, so only use when the integrals
    are very large.
    """
    if RANK == 0:
        bins = integrand.shape[axis]
        integrands = []
        x_pass = []
        for j in range(SIZE):
            partition = parallel_partition2(j, bins)
            if j == 0:
                integrands.append(get_axis(integrand, axis, partition[0], partition[-1]+1))
                x_pass.append(x[partition])
            else:
                integrands.append(get_axis(integrand, axis, partition_prev[-1], partition[-1]+1))
                par = [partition_prev[-1]] + partition
                x_pass.append(x[par])
            partition_prev = partition
    else: 
        integrands = None
        x_pass = None

    integrands = COMM.scatter(integrands, root=0)
    x_pass = COMM.scatter(x_pass, root=0)
    integral = np.trapz(integrands, x=x_pass, axis=axis)

    integral = COMM.gather(integral, root=0)
    if RANK == 0:
        return sum(integral)
    else: 
        return MathDummy()

def get_axis(m, axis, start, end):
    slc = [slice(None)] * len(m.shape)
    slc[axis] = slice(start, end)
    return m[slc]
