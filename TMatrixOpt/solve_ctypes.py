from ctypes import *
import os
import numpy as np
from numpy.ctypeslib import ndpointer

dir_path = os.path.dirname(os.path.realpath(__file__))

so_path = ''.join([dir_path, '/Solve.so'])
lib = cdll.LoadLibrary(so_path)

# useful defs
c_int_p = ndpointer(np.int32, ndim=1, flags='C')
c_double_p = ndpointer(np.double, ndim=1, flags='C')
c_complex_p = ndpointer(np.complex128, ndim=1, flags='C')

lib.solve.argtypes = [c_double,
                      c_double,
                      c_int,
                      c_int,
                      c_double_p,
                      c_complex_p,
                      c_int_p,
                      c_complex_p,
                      c_complex_p,
                      c_complex_p,
                      c_complex_p]
lib.solve.restype = None

lib.solve_forward.argtypes = [c_double,
                              c_double,
                              c_int,
                              c_int,
                              c_double_p,
                              c_complex_p,
                              c_int_p,
                              c_complex_p,
                              c_complex_p]
lib.solve_forward.restype = None

"""
lib.reflectivity_grads.argtypes = [c_complex_p,
                                   c_complex_p,
                                   c_complex_p,
                                   c_complex_p,
                                   c_complex_p,
                                   c_complex_p,
                                   c_complex_p,
                                   c_complex_p,
                                   c_int,
                                   c_int,
                                   c_int,
                                   c_double_p,
                                   c_double_p]
lib.reflectivity_grads.restype = None
"""
