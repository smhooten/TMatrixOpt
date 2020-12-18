"""
Module that provides useful tools for implementing parallel MPI operations.
"""

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
    See license given in head directory for emopt

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


class MathDummy(np.ndarray):
    """
    AUTHOR CREDIT: Andrew Michaels
    See license given in head directory for emopt

    Define a MathDummy.

    A MathDummy is an empty numpy.ndarray which devours all mathematical
    operations done by it or on it and just spits itself back out. This is
    used by emopt in order simplify its interface in the presence of MPI. For
    example, in many instances, you will need to calculate a quantity which
    need only be known on the master node, however the function performing the
    computation will be run on all nodes. Rather than having to worry about
    putting in if(NOT_PARALLEL) statements everywhere, we can just sneakily
    replace quantities involved in the calculation with MathDummies on all
    nodes but the master node.  You can then do any desired calculations
    without worying about what's going on in the other nodes.
    """
    def __new__(cls):
        obj = np.asarray([]).view(cls)
        return obj

    def __add__(self, other): return self
    def __sub__(self, other): return self
    def __mul__(self, other): return self
    def __matmul__(self, other): return self
    def __truediv__(self, other): return self
    def __floordiv__(self, other): return self
    def __mod__(self, other): return self
    def __divmod__(self, other): return self
    def __pow__(self, other, modulo=2): return self
    def __lshift__(self, other): return self
    def __rshift__(self, other): return self
    def __and__(self, other): return self
    def __xor__(self, other): return self
    def __or__(self, other): return self
    def __radd__(self, other): return self
    def __rsub__(self, other): return self
    def __rmul__(self, other): return self
    def __rmatmul__(self, other): return self
    def __rtruediv__(self, other): return self
    def __rfloordiv__(self, other): return self
    def __rmod__(self, other): return self
    def __rdivmod__(self, other): return self
    def __rpow__(self, other): return self
    def __rlshift__(self, other): return self
    def __rrshift__(self, other): return self
    def __rand__(self, other): return self
    def __rxor__(self, other): return self
    def __ror__(self, other): return self
    def __iadd__(self, other): return self
    def __isub__(self, other): return self
    def __imul__(self, other): return self
    def __imatmul__(self, other): return self
    def __itruediv__(self, other): return self
    def __ifloordiv__(self, other): return self
    def __imod__(self, other): return self
    def __ipow__(self, other, modulo=2): return self
    def __ilshift__(self, other): return self
    def __irshift__(self, other): return self
    def __iand__(self, other): return self
    def __ixor__(self, other): return self
    def __ior__(self, other): return self
    def __neg__(self): return self
    def __pos__(self): return self
    def __abs__(self): return self
    def __invert__(self): return self
    def __complex__(self): return self
    def __int__(self): return self
    def __float__(self): return self
    def __round__(self, n): return self
    def __index__(self): return 0
    def __getitem__(self, key): return self
    def __setitem__(self, key, value): return self
