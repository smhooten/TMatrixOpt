"""
Module that manages layers, pulls their params, solves the T-Matrix
system, provides gradients to the user, and updates the T-Matrix geometry.
"""

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import importlib
import inspect

from parallel_toolkit import MPI, COMM, RANK, SIZE, \
                             parallel_partition, parallel_partition2, \
                             run_on_master, MathDummy
from physical_constants import *
from geometry import Geometry, Material
import fomutils

from solve_ctypes import lib

def filter_kwargs(f):
    arg_names = inspect.getargspec(f).args
    def _f(*args, **kwargs):
        return f(*args, **{k: v for k, v in kwargs.items() if k in arg_names})
    return _f

class TMatrix:
    """
    TMatrix solver base class. Inherit this class for custom
    functionality, and define figure of merit and gradient calculations
    at a higher level. This class is built with the purpose of making
    parallel calculations over many input photon energies and incident
    angles fast and easy using arbitrarily defined geometrical layers.

    The user is expected to define figure of merit and gradient function
    that return a single figure of merit and gradient given data from
    all photon energies and incident angles.
    """
    acceptable_results = ['rTE',         # TE reflection coefficient
                          'rTM',         # TM reflection coefficient
                          'tTE',         # TE transmission coefficient
                          'tTM',         # TM transmission coefficient
                          'M_TE',        # System matrix, TE
                          'M_TM',        # System matrix, TM
                          'dM_dp_TE',    # System matrix gradient, TE
                          'dM_dp_TM',    # System matrix gradient, TM
                          'dM11_dp_TE',  # Element 11 of system matrix gradient, TE
                          'dM11_dp_TM',  # Element 11 of system matrix gradient, TM
                          'dM21_dp_TE',  # Element 21 of system matrix gradient, TE
                          'dM21_dp_TM',  # Element 21 of system matrix gradient, TM
                          'dM12_dp_TE',  # Element 12 of system matrix gradient, TE
                          'dM12_dp_TM',  # Element 12 of system matrix gradient, TM
                          'dM22_dp_TE',  # Element 22 of system matrix gradient, TE
                          'dM22_dp_TM']  # Element 22 of system matrix gradient, TM

    def __init__(self, fom_setting = 'Reflectivity', return_results = None):
        self.__layers = []
        self.__build = False
        self.__d = None
        self.__n = None
        self.__fom = None
        self.__gradient = None

        if fom_setting in dir(fomutils):
            fom_class = getattr(fomutils, fom_setting)
            self.__fom_class = fom_class
            return_results = fom_class.return_results
        elif fom_setting == 'Custom':
            for result in return_results:
                assert result in self.acceptable_results
            self.__fom_class = None
        else:
            Exception('Not an acceptable fom setting. Use "Custom", or one of the classes \
                       defined in fomutils.py')

        self.__return_results = dict.fromkeys(return_results, [])
        self.__return_results_keys = return_results

        functions = []
        for key in return_results:
            if key == 'rTE':
                functions.append(self._rTE)
            elif key == 'rTM':
                functions.append(self._rTM)
            elif key == 'tTE':
                functions.append(self._tTE)
            elif key == 'tTM':
                functions.append(self._tTM)
            elif key == 'M_TE':
                functions.append(self._M_TE)
            elif key == 'M_TM':
                functions.append(self._M_TM)
            elif key == 'dM_dp_TE':
                functions.append(self._dM_dp_TE)
            elif key == 'dM_dp_TM':
                functions.append(self._dM_dp_TM)
            elif key == 'dM11_dp_TE':
                functions.append(self._dM11_dp_TE)
            elif key == 'dM11_dp_TM':
                functions.append(self._dM11_dp_TM)
            elif key == 'dM21_dp_TE':
                functions.append(self._dM21_dp_TE)
            elif key == 'dM21_dp_TM':
                functions.append(self._dM21_dp_TM)
            elif key == 'dM12_dp_TE':
                functions.append(self._dM12_dp_TE)
            elif key == 'dM12_dp_TM':
                functions.append(self._dM12_dp_TM)
            elif key == 'dM22_dp_TE':
                functions.append(self._dM22_dp_TE)
            elif key == 'dM22_dp_TM':
                functions.append(self._dM22_dp_TM)
            else:
                Exception("Unexpected Error")

        self.__functions = functions
                              
    ############################
    # USER CONVENIENCE FUNCTIONS
    ############################
    def add_layers(self, new_layers, app_idx = None):
        """
        Add list of geometrical layers to the current layer stack.
        Can add them at a specific location ussing app_idx.
        """
        self.check_layers(new_layers)

        if app_idx is None:
            self.__layers = self.__layers + new_layers
        else:
            self.__layers = self.__layers[:app_idx] + new_layers + self.__layers[app_idx:]

        self.__build = False
        
    def remove_layers(self, rem_idxs):
        """
        Remove layers given list of indices from the stack.
        """
        for rem_idx in rem_indx:
            self.__layers.pop(rem_idx)

        self.__build = False

    def check_layers(self, layers):
        """
        Checks layers to make sure they are compatible with TMatrix
        """
        for layer in layers:
            assert isinstance(layer, Geometry)

    def print_info(self):
        """
        Prints information from each Geometry layer in order.
        Note that the layer index is Layer_Number-1.
        """
        count = 1
        for layer in self.__layers:
            print('T-Matrix Layer: %s' % str(count))
            layer.print_info()
            print('\n')
            count += 1

    def param_vec(self):
        """
        Returns a vector of zeros that is the same shape
        as the system parameters.
        """
        assert self.__build
        return np.zeros(len(self.__param_idcs))

    #def plot_R(self):
    #    import matplotlib.pyplot as plt
    #    f = plt.figure()
    #    ax = f.add_subplot(111)
    #    p = ax.imshow(self.RTM[:,:,0], aspect='auto')
    #    cbar = plt.colorbar(p)
    #    plt.show()

    @property
    def d(self):
        """
        Get current thickness values of all layers in order
        """
        assert self.__build
        return np.copy(self.__d)

    @property
    def n(self):
        """
        Get current refractive index values in order
        """
        assert self.__build
        return self.__n

    @property
    def FOM(self):
        """
        Get current FOM value
        """
        return self.__fom

    @property
    def GRADIENT(self):
        """
        Get current gradients
        """
        return self.__gradient

    @property
    def return_results(self):
        """
        Return dictionary containing all current results and values
        """
        return self.__return_results

    def build(self):
        """
        DO NOT CHANGE
        Build the system after Geometry layers have been provided.
        Must use this method before optimization.
        """
        d = []
        n = []
        idx = []
        layer_idcs = []
        param_idcs = []

        count = 0
        layer_count = 0
        for layer in self.__layers:
            ds, ns, designable, num_params = layer.get_params() # get the parameters from geometry.py

            if any([designable]):
                param_idcs += [layer_count for p in range(num_params)]
            
            if type(ds) is list:
                d.extend(ds)
                n.extend(ns)
                for i in range(len(ds)):
                    if designable[i]:
                        idx += [count]
                        layer_idcs += [layer_count]
                    count += 1
            else:
                d.append(ds)
                n.append(ns)
                if designable:
                    idx += [count]
                    layer_idcs += [layer_count]
                count += 1

            layer_count += 1

        self.__d = np.array(d)
        self.__n = n
        self.__idx = idx
        self.__layer_idcs = np.array(layer_idcs, dtype=np.int)
        self.__param_idcs = np.array(param_idcs, dtype=np.int)
        self.__build = True

    def check_gradient(self, params, step=5e-10):
        """
        Check numerical accuracy of calculated gradient
        compared to a finite difference calculation (e.g.
        changing each parameter individually and calculating
        the derivative.
        """
        # Calculate nominal figure of merit and grads
        if type(step) is float:
            steps = step*np.ones(params.shape[0])
        else:
            assert params.shape[0] == step.shape[0]
            steps = step

        fom0 = self.fom(params)
        grads0 = self.gradient(params)

        # Calculate central differences
        foms_fd1 = np.zeros(params.shape[0])
        foms_fd2 = np.zeros(params.shape[0])
        for i in range(params.shape[0]):
             params_new = np.copy(params)
             params_new[i] = params_new[i] + steps[i]
             fom_fd1 = self.fom(params_new)

             params_new = np.copy(params)
             params_new[i] = params_new[i] - steps[i]
             fom_fd2 = self.fom(params_new)

             if RANK == 0:
                 foms_fd1[i] = fom_fd1
                 foms_fd2[i] = fom_fd2

        grads_fd = (foms_fd1 - foms_fd2) / (2*steps)

        if RANK == 0:
             # Calculate relative error
             reldiff = np.mean(np.abs((grads_fd - grads0)/grads_fd))
             print('Relative error between finite difference and calculated gradient: %s' % reldiff)

             # plot
             import matplotlib.pyplot as plt
             f = plt.figure()
             ax1 = f.add_subplot(311)
             ax1.bar(np.arange(params.shape[0]), grads0)
             ax1.set_title('Calculated Gradient')
             ax1.set_xlabel('Param Number')
             ax2 = f.add_subplot(312)
             ax2.bar(np.arange(params.shape[0]), grads_fd)
             ax2.set_title('Finite Difference Gradient')
             ax2.set_xlabel('Param Number')
             ax3 = f.add_subplot(313)
             ax3.bar(np.arange(params.shape[0]), (grads_fd-grads0)/grads_fd)
             ax3.set_title('Relative Differences')
             ax3.set_xlabel('Param Number')
             plt.show()


    ############################
    # OVERRIDEABLE METHODS:
    # USER MUST OVERRIDE THESE
    ############################
    @abstractmethod
    def input_func(self):
        """
        Provide photon energy and incident angle inputs.
        Can change these iteratively.
        """
        pass

    @abstractmethod
    def calc_fom(self, **results):
        """
        Calculate overall figure of merit 
        """
        pass

    @abstractmethod
    def calc_grads(self, **results):
        """
        Calculate overall gradients
        """
        pass

    ############################
    # INTERNAL METHODS
    # DO NOT CHANGE
    ############################
    def update_system(self, params):
        layer_count = 0
        for layer in self.__layers:
            idcs = self.__param_idcs == layer_count
            if any(idcs):
                layer.update(params[idcs])
            layer_count += 1
        self.build()

    def fom(self, params):
        assert self.__build
        self.update_system(params)

        photon_energies, thetas = self.input_func()
        self.solve(photon_energies, thetas)

        fom = self.__calc_fom()
        grads = self.__calc_grads()

        self.__fom = fom
        self.__grads = grads

        return self.__fom

    def gradient(self, params):
        return self.__grads

    #def __calc_fom(self, results):
    #    if RANK == 0:
    #        rTE = results[0]
    #        rTM = results[2]
    #        RTE = (rTE*np.conj(rTE)).real
    #        RTM = (rTM*np.conj(rTM)).real
    #    else:
    #        RTE = MathDummy()
    #        RTM = MathDummy()

    #    return self.calc_fom(RTE, RTM)

    @run_on_master
    def __calc_fom(self):
        results = self.__return_results
        if self.__fom_class is None:
            return filter_kwargs(self.calc_fom)(**results)
        else:
            int_results = filter_kwargs(self.__fom_class.calc_fom)(**results)

        return filter_kwargs(self.calc_fom)(**int_results)

    #def __calc_grads(self, results):
    #    if RANK == 0:
    #        rTE = results[0]
    #        tTE = results[1]
    #        rTM = results[2]
    #        tTM = results[3]
    #        dM11_dd_TE = results[4] 
    #        dM21_dd_TE = results[5]
    #        dM11_dd_TM = results[6]
    #        dM21_dd_TM = results[7]        

    #        dM11_dp_TE, dM21_dp_TE, dM11_dp_TM, dM21_dp_TM = \
    #            self.get_layer_gradients(dM11_dd_TE, dM21_dd_TE, dM11_dd_TM, dM21_dd_TM)
    #        #dM11_dp_TE = dM11_dd_TE
    #        #dM21_dp_TE = dM21_dd_TE
    #        #dM11_dp_TM = dM11_dd_TM
    #        #dM21_dp_TM = dM21_dd_TM

    #        drTE_dp = tTE * dM21_dp_TE - rTE * tTE * dM11_dp_TE
    #        drTM_dp = tTM * dM21_dp_TM - rTM * tTM * dM11_dp_TM

    #        dRTE_dp = 2*(np.conj(rTE)*drTE_dp).real
    #        dRTM_dp = 2*(np.conj(rTM)*drTM_dp).real
    #    else:
    #        dRTE_dp = MathDummy()
    #        dRTM_dp = MathDummy()
    #    
    #    return self.calc_grads(dRTE_dp, dRTM_dp)

    @run_on_master
    def __calc_grads(self):
        results = self.__return_results
        if self.__fom_class is None:
            return filter_kwargs(self.calc_grads)(**results)
        else:
            int_results = filter_kwargs(self.__fom_class.calc_grads)(**results)
        return filter_kwargs(self.calc_grads)(**int_results)

    def get_layer_gradients_BETA(self, dM_dd):
        layer_count = 0
        flag = True
        for layer in self.__layers:
            idcs = self.__layer_idcs == layer_count
           
            if any(idcs):
                
                dM_dp_n = layer.get_layer_gradient(dM_dd[..., idcs])
                if flag:
                    dM_dp = dM_dp_n
                    flag = False
                else:
                    dM_dp = np.concatenate((dM_dp, dM_dp_n), axis=-1)
            layer_count += 1
            
        return dM_dp

    def solve(self, photon_energies, thetas):
        # Mostly data manipulation done here, distributes
        # necessary data for each process
        assert self.__build
        len_idx = len(self.__idx)
        len_pe = len(photon_energies)
        len_theta = len(thetas)
        num_sims = len_pe * len_theta
        
        if RANK == 0:
            simulation = np.array([(PE,IA) for PE in photon_energies for IA in thetas])
            simulations = []
            for j in range(SIZE):
                partition = parallel_partition2(j, num_sims)
                simulations.append(simulation[partition])
        else:
            simulations = None

        simulations = COMM.scatter(simulations, root=0)
        len_sim = len(simulations)

        M_TE = np.empty((len_sim, 2, 2), dtype=PRECISION)
        M_TM = np.empty((len_sim, 2, 2), dtype=PRECISION)
        dM_dd_TE = np.empty((len_sim, 2, 2, len_idx), dtype=PRECISION)
        dM_dd_TM = np.empty((len_sim, 2, 2, len_idx), dtype=PRECISION)

        count = 0
        for p in simulations:
            PE = p[0]
            IA = p[1]
            output = self.__solve_ALPHA(PE, IA)
            M_TE[count, ...] = output[0]
            M_TM[count, ...] = output[1]
            dM_dd_TE[count, ...] = output[2]
            dM_dd_TM[count, ...] = output[3]
            count += 1

        retvals = []
        for f in self.__functions:
            retvals.append(f(M_TE, M_TM, dM_dd_TE, dM_dd_TM))

        retvals_gather = []
        for retval in retvals:
            retvals_gather.append(COMM.gather(retval, root=0))

        if RANK == 0:
            count = 0
            for key in self.__return_results_keys:
                retval = retvals_gather[count]

                if key[:2] == 'M_':
                    # M_TE, M_TM
                    self.__return_results[key] = \
                        np.concatenate(retval, axis=0).reshape(len_pe,len_theta,2,2)
                elif key[:5] == 'dM_dp':
                    # dM_dp_TE, dM_dp_TM
                    self.__return_results[key] = \
                        np.concatenate(retval, axis=0).reshape(len_pe,len_theta,2,2,len(self.__param_idcs))
                        #np.concatenate(retval, axis=0).reshape(len_pe,len_theta,2,2,len_idx)
                elif key[:2] == 'dM':
                    # dMxy_dp_TE, dMxy_dp_TM
                    self.__return_results[key] = \
                        np.concatenate(retval, axis=0).reshape(len_pe,len_theta,len(self.__param_idcs))
                        #np.concatenate(retval, axis=0).reshape(len_pe,len_theta,len_idx)
                else:
                    # rTE, rTM, tTE, tTM
                    self.__return_results[key] = \
                        np.concatenate(retval, axis=0).reshape(len_pe,len_theta,1)

                count += 1
        return

    def __solve_BETA(self, photon_energy, theta):
        assert self.__build

        d = self.__d
        idx = self.__idx
        n = np.empty((len(self.__n), 1), dtype=PRECISION)

        for i in range(len(self.__n)):
            if type(self.__n[i]) is Material:
                n[i] = self.__n[i].get_index(photon_energy)
            else:
                n[i] = self.__n[i]

        wavelength = h*c/photon_energy
        # Calculating the angle in each layer, given the angle in layer 0
        
        angles = np.empty((d.shape[0],1),dtype=PRECISION)
        angles[0] = theta
        for i in range(1,d.shape[0]):
            angles[i] = np.arcsin(n[i-1]*np.sin(angles[i-1])/n[i])

        # horizontal wave-vector
        Kx = 2*np.pi*n/wavelength*np.cos(angles)
        L = Kx[1:]/Kx[:-1]

        M_TE = np.asarray([[1,0],[0,1]],dtype=PRECISION)
        M_TM = np.asarray([[1,0],[0,1]],dtype=PRECISION)
        
        X_TE = np.ndarray((2,2,len(idx)),dtype=PRECISION)
        X_TM = np.ndarray((2,2,len(idx)),dtype=PRECISION)

        ### calculating the forward t-matrix
        count = 0
        for i in range(d.shape[0]-1):
            T_TE = (np.asarray([[1+L[i],1-L[i]],[1-L[i],1+L[i]]],
                                dtype=PRECISION)).reshape(2,2)*0.5
            
            T_TM = (np.asarray([[L[i]*n[i]/n[i+1]+n[i+1]/n[i],
                                 L[i]*n[i]/n[i+1]-n[i+1]/n[i]],
                                [L[i]*n[i]/n[i+1]-n[i+1]/n[i],
                                 L[i]*n[i]/n[i+1]+n[i+1]/n[i]]],
                                 dtype=PRECISION)).reshape(2,2)*0.5
                
            P = (np.asarray([[np.exp(-1j*Kx[i+1]*d[i+1]),0],
                             [0,np.exp(1j*Kx[i+1]*d[i+1])]],
                             dtype=PRECISION)).reshape(2,2)
            
            B_TE = T_TE @ P
            B_TM = T_TM @ P

            if i+1 in idx:
                X_TE[:,:,count] = M_TE @ T_TE
                X_TM[:,:,count] = M_TM @ T_TM
                count += 1
            
            M_TE = M_TE @ B_TE
            M_TM = M_TM @ B_TM

        # NOTE: Need to cap off transmission matrices?
        # Don't need this if d[-1]=0.0

        ## calculating the derivative
        dM_dd_TE = np.ndarray((2,2,len(idx)),dtype=PRECISION)
        dM_dd_TM = np.ndarray((2,2,len(idx)),dtype=PRECISION)   

        Idash = (np.asarray([[-1,0],
                             [0, 1]], 
                             dtype=PRECISION)).reshape(2,2)

        count = 0
        for i in idx:
            try:
                A_TE = np.linalg.solve(X_TE[:,:,count], M_TE)
                A_TM = np.linalg.solve(X_TM[:,:,count], M_TM)
            except:
                A_TE, _, _, _ = np.linalg.lstsq(X_TE[:,:,count], M_TE, rcond=None)
                A_TM, _, _, _ = np.linalg.lstsq(X_TM[:,:,count], M_TM, rcond=None)
                
            #Xinv_TE = np.linalg.pinv(X_TE[:,:,count])
            #Xinv_TM = np.linalg.pinv(X_TM[:,:,count])

            dM_dd_TE[:,:,count] = 1j*Kx[i]*X_TE[:,:,count] @ Idash @ A_TE
            dM_dd_TM[:,:,count] = 1j*Kx[i]*X_TM[:,:,count] @ Idash @ A_TM
            #dM_dd_TE[:,:,count] = 1j*Kx[i]*X_TE[:,:,count] @ Idash @ Xinv_TE @ M_TE
            #dM_dd_TM[:,:,count] = 1j*Kx[i]*X_TM[:,:,count] @ Idash @ Xinv_TM @ M_TM

            count += 1

        return M_TE, M_TM, dM_dd_TE, dM_dd_TM

    def __solve(self, photon_energy, theta):
        assert self.__build

        d = self.__d
        idx = self.__idx
        n = np.empty((len(self.__n), 1), dtype=PRECISION)

        for i in range(len(self.__n)):
            if type(self.__n[i]) is Material:
                n[i] = self.__n[i].get_index(photon_energy)
            else:
                n[i] = self.__n[i]

        wavelength = h*c/photon_energy
        # Calculating the angle in each layer, given the angle in layer 0
        
        angles = np.empty((d.shape[0],1),dtype=PRECISION)
        angles[0] = theta
        for i in range(1,d.shape[0]):
            angles[i] = np.arcsin(n[i-1]*np.sin(angles[i-1])/n[i])

        # horizontal wave-vector
        Kx = 2*np.pi*n/wavelength*np.cos(angles)
        L = Kx[1:]/Kx[:-1]
        
        M_TE = np.asarray([[1,0],[0,1]],dtype=PRECISION)
        M_TM = np.asarray([[1,0],[0,1]],dtype=PRECISION)
        
        X_TE = np.ndarray((2,2,len(idx)),dtype=PRECISION)
        X_TM = np.ndarray((2,2,len(idx)),dtype=PRECISION)

        ### calculating the forward t-matrix
        count = 0
        for i in range(d.shape[0]-1):
            T_TE = (np.asarray([[1+L[i],1-L[i]],[1-L[i],1+L[i]]],
                                dtype=PRECISION)).reshape(2,2)*0.5
            
            T_TM = (np.asarray([[L[i]*n[i]/n[i+1]+n[i+1]/n[i],
                                 L[i]*n[i]/n[i+1]-n[i+1]/n[i]],
                                [L[i]*n[i]/n[i+1]-n[i+1]/n[i],
                                 L[i]*n[i]/n[i+1]+n[i+1]/n[i]]],
                                 dtype=PRECISION)).reshape(2,2)*0.5
                
            P = (np.asarray([[np.exp(-1j*Kx[i+1]*d[i+1]),0],
                             [0,np.exp(1j*Kx[i+1]*d[i+1])]],
                             dtype=PRECISION)).reshape(2,2)
            
            B_TE = T_TE @ P
            B_TM = T_TM @ P
            
            if i+1 in idx:
                X_TE[:,:,count] = M_TE @ T_TE
                X_TM[:,:,count] = M_TM @ T_TM
                count += 1
            
            M_TE = M_TE @ B_TE
            M_TM = M_TM @ B_TM

        # NOTE: Need to cap off transmission matrices?
        # Don't need this if d[-1]=0.0

        ## calculating the derivative
        dM_dd_TE = np.ndarray((2,2,len(idx)),dtype=PRECISION)
        dM_dd_TM = np.ndarray((2,2,len(idx)),dtype=PRECISION)   

        count = 0
        for i in idx:
            phase_m = np.exp(-1j*Kx[i]*d[i])
            phase_p = np.exp(1j*Kx[i]*d[i])

            P = (np.asarray([[phase_m,0],
                             [0,phase_p]],
                             dtype=PRECISION)).reshape(2,2)

            Pdash = (np.asarray([[-phase_m,0],
                                 [0,phase_p]],
                                 dtype=PRECISION)*1j*Kx[i]).reshape(2,2)

            Y_TE = np.linalg.pinv(X_TE[:,:,count] @ P) @ M_TE
            dM_dd_TE[:,:,count] = X_TE[:,:,count] @ Pdash @ Y_TE

            Y_TM = np.linalg.pinv(X_TM[:,:,count] @ P) @ M_TM
            dM_dd_TM[:,:,count] = X_TM[:,:,count] @ Pdash @ Y_TM

            count += 1

        return M_TE, M_TM, dM_dd_TE, dM_dd_TM

    def __solve_ALPHA(self, photon_energy, theta):
        assert self.__build

        d = self.__d
        idx = np.array(self.__idx, dtype=np.int32)
        n = np.empty(len(self.__n), dtype=np.complex128)

        len_idx = len(idx)
        len_d = len(d)

        for i in range(len(self.__n)):
            if type(self.__n[i]) is Material:
                n[i] = self.__n[i].get_index(photon_energy)
            else:
                n[i] = self.__n[i]

        M_TE = np.empty(4, dtype=np.complex128)
        M_TM = np.empty(4, dtype=np.complex128)
        dM_dd_TE = np.empty(4*len_idx, dtype=np.complex128)
        dM_dd_TM = np.empty(4*len_idx, dtype=np.complex128)

        lib.solve(photon_energy,
                  theta,
                  len_d,
                  len_idx,
                  d,
                  n,
                  idx,
                  M_TE,
                  M_TM,
                  dM_dd_TE,
                  dM_dd_TM)

        #M_TE = np.asarray(M_TE, dtype=PRECISION)
        #M_TM = np.asarray(M_TM, dtype=PRECISION)
        #dM_dd_TE = np.asarray(dM_dd_TE, dtype=PRECISION)
        #dM_dd_TM = np.asarray(dM_dd_TM, dtype=PRECISION)

        return(M_TE.reshape((2,2)), M_TM.reshape((2,2)),
               dM_dd_TE.reshape((2,2,-1)), dM_dd_TM.reshape((2,2,-1)))

    def _rTE(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return M_TE[:,1,0]/M_TE[:,0,0]

    def _rTM(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return M_TM[:,1,0]/M_TM[:,0,0]

    def _tTE(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return 1.0/M_TE[:,0,0]

    def _tTM(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return 1.0/M_TM[:,0,0]

    def _M_TE(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return M_TE

    def _M_TM(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return M_TM

    def _dM_dp_TE(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TE)

    def _dM_dp_TM(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TM)

    def _dM11_dp_TE(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TE[:,0,0,:])

    def _dM11_dp_TM(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TM[:,0,0,:])

    def _dM21_dp_TE(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TE[:,1,0,:])

    def _dM21_dp_TM(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TM[:,1,0,:])

    def _dM12_dp_TE(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TE[:,0,1,:])

    def _dM12_dp_TM(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TM[:,0,1,:])

    def _dM22_dp_TE(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TE[:,1,1,:])

    def _dM22_dp_TM(self, M_TE, M_TM, dM_dd_TE, dM_dd_TM):
        return self.get_layer_gradients_BETA(dM_dd_TM[:,1,1,:])
