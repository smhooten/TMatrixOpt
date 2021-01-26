"""
TMatrixOpt: A fast and modular transfer-matrix optimization package 
for 1D optical devices.
Copyright (C) 2021 Sean Hooten & Zunaid Omair

TMatrixOpt/optimizer.py

ATTN: This module is a redistributed and modified version (under the
GNU General Public License, Version 3.0) of a similar module 
from EMopt, copyright Andrew Michaels, which may be found here:
    https://github.com/anstmichaels/emopt.git

This module is a wrapper around scipy.optimize.minimize allowing
the user to easily invoke conventional gradient-based optimization
methods. The Optimizer manages MPI communication automatically,
allowing for parallelized calculation of a merit function and
its gradient.
"""

__author__ = 'Andrew Michaels, Sean Hooten'
__version__ = '1.0'
__license__ = 'GPL 3.0'

import numpy as np
from math import pi
from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from scipy.optimize import basinhopping
from .parallel_toolkit import MPI, COMM, RANK, SIZE

class Optimizer:
    class RunCommands:
        """Run command codes used during message passing.

        We need a way to signal the non-master nodes to perform different
        operations during the optimization.  We do this by sending integers
        from the master node to the other nodes containing a command code. The
        commands are specified using an enum-like class.

        Attributes
        ----------
        FOM : int
            Tells worker nodes to compute the figure of merit
        GRAD : int
            Tells the worker nodes to compute the gradient of the figure of
            merit
        EXIT : int
            Tells the worker nodes to finish.
        """
        FOM = 0
        GRAD = 1
        EXIT = 2

    def __init__(self, tm, p0, callback_func=None, opt_method='BFGS', \
                 Nmax=1000, tol=1e-5, bounds=None, scipy_verbose=True, additional_options=None):
        self.tm = tm

        self.p0 = p0

        if(callback_func is None):
            self.callback = lambda p : None
        else:
            self.callback = callback_func

        self.opt_method = opt_method
        self.Nmax = Nmax
        self.tol = tol
        self.bounds = bounds
        self.scipy_verbose = scipy_verbose
        self.additional_options = additional_options

        self._comm = MPI.COMM_WORLD

    def run(self):
        """Run the optimization.

        Returns
        -------
        float
            The final figure of merit
        numpy.array
            The optimized design parameters
        """
        command = None
        running = True
        params = np.zeros(self.p0.shape)
        if(RANK == 0):
            fom, params = self.run_sequence(self.tm)
        else:
            while(running):
                # Wait for commands from the master node
                command = self._comm.bcast(command, root=0)

                if(command == self.RunCommands.FOM):
                    params = self._comm.bcast(params, root=0)
                    self.tm.fom(params)
                elif(command == self.RunCommands.GRAD):
                    params = self._comm.bcast(params, root=0)
                    self.tm.gradient(params)
                elif(command == self.RunCommands.EXIT):
                    running = False

            fom = None
            params = None

        # share the final fom and parameters with all processes
        fom = COMM.bcast(fom, root=0)
        params = COMM.bcast(params, root=0)

        return fom, params

    def __fom(self, params):
        # Execute the figure of merit in parallel
        command = self.RunCommands.FOM
        self._comm.bcast(command, root=0)
        self._comm.bcast(params, root=0)
        return self.tm.fom(params)

    def __gradient(self, params):
        # Execute the figure of merit in parallel
        command = self.RunCommands.GRAD
        self._comm.bcast(command, root=0)
        self._comm.bcast(params, root=0)
        return self.tm.gradient(params)

    def run_sequence(self, tm):
        """Sequential optimization code.

        In general, the optimization itself is run in parallel.  Instead, only
        the calculation of the figure of merit and gradient takes advantage of
        paralellism (which is where the bulk of the computational complexity
        comes in).  This function defines the sequential optimization code and
        makes calls to the parallel components.

        Notes
        -----
        Override this method for custom functionality!

        Parameters
        ----------
        am : :class:`emopt.adjoint_method.AdjointMethod`
            The adjoint method object responsible for FOM and gradient
            calculations.

        Returns
        -------
        (float, numpy.ndarray)
            The optimized figure of merit and the corresponding set of optimal
            design parameters.
        """
        self.__fom(self.p0)
        self.callback(self.p0)

        if self.additional_options is not None:
            options = self.additional_options.copy()
            options.update({'maxiter':self.Nmax,
                            'disp': self.Nmax})
        else:
            options = {'maxiter':self.Nmax, 
                       'disp': self.scipy_verbose}
                       

        result = minimize(self.__fom, self.p0, method=self.opt_method,
                          jac=self.__gradient, callback=self.callback,
                          tol=self.tol, bounds=self.bounds,
                          options=options)

        command = self.RunCommands.EXIT
        self._comm.bcast(command, root=0)

        return result.fun, result.x


class basin_hopping(Optimizer):
    def __init__(self, am, p0, callback_func=None, opt_method='BFGS', \
                 Nmax=1000, tol=1e-5, bounds=None, scipy_verbose=True, additional_options=None):
        super(basin_hopping, self).__init__(am, p0, callback_func, opt_method, \
                 Nmax, tol, bounds, scipy_verbose, additional_options)

    def __fom(self, params):
        # Execute the figure of merit in parallel
        command = self.RunCommands.FOM
        self._comm.bcast(command, root=0)
        self._comm.bcast(params, root=0)
        return self.tm.fom(params)

    def __gradient(self, params):
        # Execute the figure of merit in parallel
        command = self.RunCommands.GRAD
        self._comm.bcast(command, root=0)
        self._comm.bcast(params, root=0)
        return self.tm.gradient(params)

    def run_sequence(self, am):
        self.__fom(self.p0)
        self.callback(self.p0)

        if self.additional_options is not None:
            options = self.additional_options.copy()
            options.update({'maxiter':self.Nmax,
                            'disp': self.scipy_verbose})
        else:
            options = {'maxiter':self.Nmax, 
                       'disp': self.scipy_verbose}

        minimizer_kwargs = dict(method=self.opt_method, jac=self.__gradient, callback=self.callback, 
                                tol=self.tol, options=options)

        result = basinhopping(self.__fom, self.p0, niter=40, T=0.01, stepsize=0.05, minimizer_kwargs = minimizer_kwargs)
        #result = basinhopping(self.__fom, self.p0, niter=10, T=0.3, stepsize=0.01, minimizer_kwargs = minimizer_kwargs)

        command = self.RunCommands.EXIT
        self._comm.bcast(command, root=0)

        return result.fun, result.x
    

def momentum_gradient_descent(fun, x0, jac=None, tol=1e-4, stepsize=0.1, \
                              beta=0.1, disp=True, maxiter=100, callback=None, **options):
    x = x0
    y = fun(x0)
    funcalls = 1
    niter = 0
    stop = False

    grad = jac(x)
    del_x = -1*stepsize*grad

    while not stop and niter<maxiter:
        niter += 1
        funcalls += 1
        x = x+del_x
        y = fun(x)

        grad = jac(x)
        del_x = -1*stepsize*grad + beta*del_x
        
        if callback is not None:
            callback(x)


    return OptimizeResult(fun=y, x=x, nit=niter,
                          nfev=funcalls, success=(niter > 1))
