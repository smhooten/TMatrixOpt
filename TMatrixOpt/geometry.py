"""
Module for defining geometry objects that are compatible with the TMatrix
solver. Each geometry object should inherit from the Geometry class. The
class should have at least 5 methods that override the abstract methods
of the Geometry class given below.
"""
from abc import ABCMeta, abstractmethod
import csv
import copy
import numpy as np

from .physical_constants import *

class Material:
    """
    Allows the user to define dispersive materials to feed into
    the solver. The user should provide a csv data_file that gives
    the material (complex) refractive index as a function of photon
    energy. The solver will linearly interpolate the data to find
    the (complex) refractive index at undefined photon energies.

    If refractive index is complex, the data file should include
    3 columns: energy, n, k. Otherwise use two columns.
    """
    def __init__(self, data_file):
        self.data_file = data_file
        photon_energy = []
        n = []
        k = []
        with open(data_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                photon_energy.append(float(row[0]))
                n.append(float(row[1]))
                try:
                    k.append(float(row[2]))
                except:
                    pass

        photon_energy = np.array(photon_energy)
        sorting_inds = np.argsort(photon_energy)
        self.photon_energy = photon_energy[sorting_inds]
        n = np.array(n, dtype=PRECISION)[sorting_inds]
        if k:
            k = np.array(k, dtype=PRECISION)[sorting_inds]
            self.n = n + 1j*k
        else:
            self.n = n

    def get_index(self, PEq):
        return np.interp(PEq, self.photon_energy, self.n)

    def __call__(self, PEq):
        return self.get_index(PEq)

class Geometry:
    """
    All geometry objects should inherit this class!
    Please override each abstractmethod and return the
    necessary parameters, gradients, and etc.
    """
    def __init__(self, name):
        """
        Initialize the name of the layer using super().__init__(name).
        """
        self.name = name

    @abstractmethod
    def print_info(self):
        """
        Prints any desired information from the layer (e.g. thicknesses,
        refractive indices, etc) when using the TMatrixObject.print_info() 
        method.
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Returns layer thicknesses, refractive indexes, designable boolean,
        and number of layer parameters as lists in that order.
        """
        pass

    @abstractmethod
    def update(self, params):
        """
        Updates layer parameters given params from the TMatrix solver.
        """
        pass

    @abstractmethod
    def get_layer_gradient(self, dM_dd):
        """
        Given gradient of the matrix elements with respect to the 
        layer thicknesses (dM/dd), return gradient of the matrix 
        elements with respect to the layer parameters (dM/dp).
        """
        pass

class Layer(Geometry):
    """
    A simple layer with some thickness and refractive index.
    Its only parameter updates the layer thickness from its nominal (initial)
    value.
    """
    def __init__(self, name, d, n, designable):
        super().__init__(name)
        assert type(d) is float or np.float64
        #assert d >= 0.0
        #if d < 10e-9:
        #    d = 10e-9
        self.d0 = d
        self.d = copy.deepcopy(d)

        assert type(n) is float or PRECISION or np.complex128 or Material
        self.n = n
        self.designable = designable
        self.num_params = 1

    def print_info(self):
        print('Name: %s' % str(self.name))
        print('Refractive Index: %s' % str(self.n))
        print('Thickness: %s' % str(self.d))
        print('Designable: %s' % str(self.designable))

    def get_params(self):
        return self.d, self.n, self.designable, self.num_params

    def update(self, param):
        d = self.d0 + float(param)
        #if d < 10e-9:
        #    d = 10e-9
        if d < 0.0:
             d = 0.0
        self.d = d

    def get_layer_gradient(self, dM_dd):
        return dM_dd

class Layer0(Geometry):
    """
    A simple layer with some thickness and refractive index.
    Its only parameter changes the thickness to whatever the parameter value
    is supplied.
    """
    def __init__(self, name, d, n, designable):
        super().__init__(name)
        assert type(d) is float or np.float64
        self.d = d

        assert type(n) is float or PRECISION or np.complex128 or Material
        self.n = n
        self.designable = designable
        self.num_params = 1

    def print_info(self):
        print('Name: %s' % str(self.name))
        print('Refractive Index: %s' % str(self.n))
        print('Thickness: %s' % str(self.d))
        print('Designable: %s' % str(self.designable))

    def get_params(self):
        return self.d, self.n, self.designable, self.num_params

    def update(self, param):
        d = float(param)
        if d < 0.0:
             d = 0.0
        self.d = d

    def get_layer_gradient(self, dM_dd):
        return dM_dd

class Layer_Map(Geometry):
    """
    A simple layer with some thickness and refractive index.
    Its only parameter changes the thickness to whatever the parameter value
    is supplied.
    """
    def __init__(self, name, d, n, designable, lower=0.0, upper=2.0e-6):
        super().__init__(name)
        assert type(d) is float or np.float64
        self.d = d

        assert type(n) is float or PRECISION or np.complex128 or Material
        self.n = n
        self.designable = designable
        self.num_params = 1
        self.lower = lower
        self.upper = upper

    def print_info(self):
        print('Name: %s' % str(self.name))
        print('Refractive Index: %s' % str(self.n))
        print('Thickness: %s' % str(self.d))
        print('Designable: %s' % str(self.designable))

    def get_params(self):
        return self.d, self.n, self.designable, self.num_params

    def update(self, param):
        d = float(self.lower*(1.0-param) + self.upper*param)
        #d = float(param)
        if d < 0.0:
             d = 0.0
        self.d = d

    def get_layer_gradient(self, dM_dd):
        return dM_dd*(self.upper - self.lower)

class DiscreteChirp(Geometry):
    """
    A binarized Bragg Layer with Np pairs. Provide a center energy (in eV),
    refractive indices (n1, n2 -- possibly dispersive), and center refractive 
    indices (cen_n1, cen_n2 -- must be type float). Each design parameter updates 
    the layer thicknesses individually.
    """
    def __init__(self, name, cen_energy, Np, n1, n2, cen_n1, cen_n2, designable):
        super().__init__(name)
        self.cen_energy = cen_energy
        self.Np = Np
        self.designable = designable
        self.n1 = n1
        self.n2 = n2
        self.num_params = 2*Np

        M = h*c/(4.0*cen_energy*q)
        d1 = M/cen_n1
        d2 = M/cen_n2
        
        layers = []
        for i in range(Np):
            layers.append(Layer(self.name+'_Np1_'+str(i), d1, n1, designable))
            layers.append(Layer(self.name+'_Np2_'+str(i), d2, n2, designable))

        self.layers = layers

    def print_info(self):
        print('Discrete Chirp: %s' % self.name)
        for i in range(2*self.Np):
            self.layers[i].print_info()

    def get_params(self):
        d = []
        n = []
        designable = []
        for i in range(2*self.Np):
            d.append(self.layers[i].d)
            n.append(self.layers[i].n)
            designable.append(self.designable)
        return d, n, designable, self.num_params

    def update(self, params):
        for j in range(len(self.layers)):
            layer = self.layers[j]
            layer.update(params[j])

    def get_layer_gradient(self, dM_dd):
        return dM_dd

class DiscreteChirp0(Geometry):
    """
    A binarized Bragg Layer with Np pairs. Provide a center energy (in eV),
    refractive indices (n1, n2 -- possibly dispersive), and center refractive 
    indices (cen_n1, cen_n2 -- must be type float). Each design parameter updates 
    the layer thicknesses individually.
    """
    def __init__(self, name, cen_energy, Np, n1, n2, cen_n1, cen_n2, designable):
        super().__init__(name)
        self.cen_energy = cen_energy
        self.Np = Np
        self.designable = designable
        self.n1 = n1
        self.n2 = n2
        self.num_params = 2*Np

        M = h*c/(4.0*cen_energy*q)
        d1 = M/cen_n1
        d2 = M/cen_n2
        
        layers = []
        for i in range(Np):
            layers.append(Layer0(self.name+'_Np1_'+str(i), d1, n1, designable))
            layers.append(Layer0(self.name+'_Np2_'+str(i), d2, n2, designable))

        self.layers = layers

    def print_info(self):
        print('Discrete Chirp: %s' % self.name)
        for i in range(2*self.Np):
            self.layers[i].print_info()

    def get_params(self):
        d = []
        n = []
        designable = []
        for i in range(2*self.Np):
            d.append(self.layers[i].d)
            n.append(self.layers[i].n)
            designable.append(self.designable)
        return d, n, designable, self.num_params

    def update(self, params):
        for j in range(len(self.layers)):
            layer = self.layers[j]
            layer.update(params[j])

    def get_layer_gradient(self, dM_dd):
        return dM_dd

class LinearChirp(Geometry):
    """
    A binarized linearly chirped Bragg Layer with Np pairs. Provide a starting and
    end energy (in eV), refractive indices (n1, n2 -- possibly dispersive), and center refractive 
    indices (cen_n1, cen_n2 -- must be type float). Each design parameter updates 
    the layer thicknesses individually.
    """
    def __init__(self, name, energy1, energy2, Np, n1, n2, cen_n1, cen_n2, designable):
        super().__init__(name)
        self.energy1 = energy1 # minimum frequency over which the chirp is defined
        self.energy2 = energy2 # maximum frequency over which the chirp is defined
        self.Np = Np # number of high and low index pairs used in the chirp
        self.designable = designable # If 1, the chirp is optimized; if 0, the chirp is not optimized
        self.n1 = n1 # refractive index of the low-index material
        self.n2 = n2 # refractive index of the high-index material
        self.num_params = 2*Np 

        M1 = h*c/(4.0*energy1*q)
        M2 = h*c/(4.0*energy2*q)

        a = M1
        if Np <= 1:
            b = 0.0
        else:
            b = 1.0/(Np-1)*(M1-M2)
        
        layers = []
        for i in range(Np):
            d1 = (a-b*i)/cen_n1
            d2 = (a-b*i)/cen_n2
            layers.append(Layer(name+'_Np1_'+str(i), d1, n1, designable))
            layers.append(Layer(name+'_Np2_'+str(i), d2, n2, designable))

        self.layers = layers

    def print_info(self):
        print('Linear Chirp: %s' % self.name)
        for i in range(2*self.Np):
            self.layers[i].print_info()

    def get_params(self):
        d = []
        n = []
        designable = []
        for i in range(2*self.Np):
            d.append(self.layers[i].d)
            n.append(self.layers[i].n)
            designable.append(self.designable)
        return d, n, designable, self.num_params

    def update(self, params):
        for j in range(len(self.layers)):
            layer = self.layers[j]
            layer.update(params[j])

    def get_layer_gradient(self, dM_dd):
        return dM_dd

class DiscreteChirp_BETA(Geometry):
    """
    A binarized Bragg Layer with Np pairs. Provide a starting and
    end energy (in eV), refractive indices (n1, n2 -- possibly dispersive), and center refractive 
    indices (cen_n1, cen_n2 -- must be type float). In this case there is only one design parameter
    which updates the center energy of the DiscreteChirp. So each layer will change
    collectively.
    """
    def __init__(self, name, cen_energy, Np, n1, n2, cen_n1, cen_n2, designable):
        super().__init__(name)
        self.cen_energy0 = cen_energy
        self.cen_energy = np.copy(cen_energy)
        self.Np = Np
        self.designable = designable
        self.n1 = n1
        self.n2 = n2
        self.cen_n1 = cen_n1
        self.cen_n2 = cen_n2
        self.num_params = 1

        self.build_layers()

    def build_layers(self):
        M = h*c/(4.0*self.cen_energy*q)
        d1 = M/self.cen_n1
        d2 = M/self.cen_n2
        
        layers = []
        for i in range(self.Np):
            layers.append(Layer(self.name+'_Np1_'+str(i), d1, self.n1, self.designable))
            layers.append(Layer(self.name+'_Np2_'+str(i), d2, self.n2, self.designable))

        self.layers = layers

    def print_info(self):
        print('Discrete Chirp: %s' % self.name)
        for i in range(2*self.Np):
            self.layers[i].print_info()

    def get_params(self):
        d = []
        n = []
        designable = []
        for i in range(2*self.Np):
            d.append(self.layers[i].d)
            n.append(self.layers[i].n)
            designable.append(self.designable)
        return d, n, designable, self.num_params

    def update(self, params):
        cen_eng_new = self.cen_energy0 + float(params)
        self.cen_energy = cen_eng_new
        self.build_layers()

    def get_layer_gradient(self, dM_dd):
        M = -h*c/(4.0*q*self.cen_energy**2)
        dd_dp_n1 = M/self.cen_n1
        dd_dp_n2 = M/self.cen_n2
        dM_dp = np.zeros(dM_dd.shape, dtype=PRECISION)
        for i in range(self.Np):
            dM_dp[..., ::2] = dM_dd[..., ::2] * dd_dp_n1
            dM_dp[..., 1::2] = dM_dd[..., 1::2] * dd_dp_n2

        dM_dp = np.sum(dM_dp, axis=-1)

        return dM_dp
