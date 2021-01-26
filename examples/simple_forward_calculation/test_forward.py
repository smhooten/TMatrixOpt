"""
Documentation in development!
"""

import numpy as np
import TMatrixOpt
from TMatrixOpt import geometry, optimizer, solve, fomutils, solve
from TMatrixOpt.solve import TMatrix
from TMatrixOpt.physical_constants import *
from TMatrixOpt.parallel_toolkit import RANK, parallel_integral

class LayerStack(TMatrix):
    def __init__(self, photon_energies, thetas, layers):
        super().__init__(fom_setting = 'Reflectivity_FORWARD_ONLY')
        self.photon_energies = photon_energies*q
        self.thetas = thetas * np.pi/180.0

        wse = self.energy_weighting_func(self.photon_energies)
        wsa = self.angle_weighting_func(self.thetas)

        self.wse = wse[:, np.newaxis, np.newaxis]
        self.wsa = wsa[np.newaxis, :, np.newaxis]
        
        self.denominator1 = np.trapz(wse, x=self.photon_energies)
        self.denominator2 = np.trapz(wsa, x=self.thetas)
        
        self.add_layers(layers)
    
    def input_func(self):
        return self.photon_energies, self.thetas

    def calc_fom(self, RTE, RTM):
        self.RTE = RTE
        self.RTM = RTM

        int1_TE = np.trapz(RTE*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TE = np.trapz(int1_TE, x=self.thetas, axis=0)

        int1_TM = np.trapz(RTM*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TM = np.trapz(int1_TM, x=self.thetas, axis=0)

        tot = 0.5 * (int2_TE + int2_TM) / (self.denominator1 * self.denominator2)

        return tot

    def calc_grads(self):
        pass

    def energy_weighting_func(self, Ep, T=1473.0):
        weight = 2*Ep**3/(c**2*h**3*(np.exp(Ep/(Kb*T))-1))
        return weight

    def angle_weighting_func(self, thetas):
        weight = np.cos(thetas)*np.sin(thetas)
        return weight

    def plot_fom(self):
        import matplotlib.pyplot as plt
        pe = self.photon_energies / q
        thetas = self.thetas * 180.0 / np.pi
        extent = [thetas[0], thetas[-1], pe[0], pe[-1]]

        f = plt.figure()
        ax1 = f.add_subplot(121)
        im1 = ax1.imshow(self.RTE, origin='lower', cmap='plasma', vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
        ax1.set_title('TE Reflectivity')
        ax1.set_xlabel('Incident Angle (Degrees)')
        ax1.set_ylabel('Photon Energy (eV)')
        f.colorbar(im1, ax=ax1)

        ax2 = f.add_subplot(122)
        im2 = ax2.imshow(self.RTM, origin='lower', cmap='plasma', vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
        ax2.set_title('TM Reflectivity')
        ax2.set_xlabel('Incident Angle (Degrees)')
        ax2.set_ylabel('Photon Energy (eV)')
        f.colorbar(im2, ax=ax2)

        plt.show()

if __name__ == '__main__':
    import time

    air_front = geometry.Layer('air_front', 0.0, 1.0, False)
    superstrate = geometry.Layer('superstrate', 0.4668e-6, 3.5, False)
    #layer2 = geometry.LinearChirp('linearchirp', 0.1, 0.74, 6, 1.6, 3.5, 1.6, 3.5, False)
    layer2 = geometry.DiscreteChirp('linearchirp', 0.45, 20, 1.6, 3.5, 1.6, 3.5, False)
    metal_back = geometry.Layer('metal_back', 0.2e-6, 0.5+11*1j, False)
    air_back = geometry.Layer('air_back', 0.0, 1.0, False)
    #layers = [air_front, superstrate, layer2, metal_back, air_back]
    #layers = [air_front, superstrate, layer2, air_back]
    layers = [air_front, layer2, air_back]
   
    photon_energies = np.linspace(0.1, 0.74, num=6400)
    thetas = np.linspace(0,90, num=181)
    stack = LayerStack(photon_energies, thetas, layers)
    stack.build()

    start = time.time()
    fom = stack.fom_forward()
    end = time.time()

    if(RANK==0):
        print(fom)
        print('time:', end-start)
        stack.print_info()
        stack.plot_fom()

