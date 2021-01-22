"""
Note: This is an advanced script, it is advised that you consider the other
examples before this one.

This script compares the speed of Lumerical's stackrt feature to TMatrixOpt.
See lumerical_comparison.lsf for the script needed to run stackrt
of an equivalent structure used here.

Notably, this script is longer, but the philosophy of TMatrixOpt is that it
is incredibly modular and flexible at the cost of some additional user
input. By default, Lumerical always calculates all of the Fresnel coefficients
as well as the power coefficients (reflectivity, transmission, etc). In this 
case, for an accurate comparison with lumerical, we calculate all of the same 
results.

However, it should be noted that TMatrixOpt allows one to choose exactly what 
results are desired and then perform operations over those results only (e.g. 
weighted average of reflectivity), as well as gradients of those results. 
This gives the flexibility to use creative merit functions for (e.g.) bandpass 
filters, and easily optimize devices for that purpose.

Furthermore, TMatrixOpt is parallelized, which of course requires some
additional overhead. Lumerical's stackrt feature is not parallelized, allowing
their calculations to be somewhat more efficient than TMatrixOpt for single
CPU operation. However, when using multiple CPUs, TMatrixOpt is faster.
"""

import numpy as np
import TMatrixOpt
from TMatrixOpt import geometry, solve, fomutils
from TMatrixOpt.solve import TMatrix
from TMatrixOpt.physical_constants import *
from TMatrixOpt.parallel_toolkit import RANK

class LayerStack(TMatrix):
    def __init__(self, photon_energies, thetas, layers, allreal=False):
        self.allreal = allreal # simplifies transmission calculation if true

        # We collect all the Fresnel coefficients, calculate power coefficients
        # locally, since we are not interested in any actual figure of merit
        return_results = ['rTE', 'rTM', 'tTE', 'tTM'] 

        # init TMatrix, we set fom to custom for slight speed-up
        super().__init__(fom_setting = 'Custom', return_results = return_results)

        # convert user provided pe and ia to SI units
        self.photon_energies = photon_energies*q
        self.thetas = thetas * np.pi/180.0

        # quick hack to get first and last refractive index values of layers
        _, n1, _, _ = layers[0].get_params()
        _, n2, _, _ = layers[-1].get_params()

        try:
            n1 = n1[0]
        except:
            pass

        try:
            n2 = n2[-1]
        except:
            pass

        # Simplifies transmission calculation if this is true
        if n1 == n2:
            self.n1EQUALSn2 = True
        else:
            self.n1EQUALSn2 = False
            self.n1 = n1
            self.n2 = n2

        self.add_layers(layers)
    
    def input_func(self):
        # required function for TMatrix
        return self.photon_energies, self.thetas

    def calc_fom(self, rTE, rTM, tTE, tTM):
        # required function for TMatrix
        # We calculate the reflection and transmission coefficients
        RTE = (rTE*np.conj(rTE)).real
        RTM = (rTM*np.conj(rTM)).real

        # If materials are real, then there is no absorption
        if self.allreal:
            TTE = 1.0-RTE
            TTM = 1.0-RTM
        else:
            if self.n1EQUALSn2:
                TTE = (tTE*np.conj(tTE)).real
                TTM = (tTM*np.conj(tTM)).real
            else:
                # A little extra work needs to be done
                costheta = np.cos(self.thetas)
                coeff = np.sqrt( (self.n2/ self.n1)**2 + costheta**2 - 1.0 ) / costheta
                coeff = coeff[np.newaxis, :, np.newaxis]

                TTE = coeff*(tTE*np.conj(tTE)).real
                TTM = coeff*(tTM*np.conj(tTM)).real

        # save all data
        self.rTE = rTE
        self.rTM = rTM
        self.tTE = tTE
        self.tTM = tTM

        self.RTE = RTE
        self.RTM = RTM
        self.TTE = TTE
        self.TTM = TTM

        return 0.0

    def calc_grads(self):
        # Not needed, but still needs to be defined
        pass

    def plot_fom(self):
        # plot reflection, transmission, and absorption
        import matplotlib.pyplot as plt
        pe = self.photon_energies / q
        thetas = self.thetas * 180.0 / np.pi
        extent = [thetas[0], thetas[-1], pe[0], pe[-1]]

        f = plt.figure()
        ax1 = f.add_subplot(121)
        im1 = ax1.imshow(self.RTE, origin='lower', cmap='hot', vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
        ax1.set_title('TE Reflectivity')
        ax1.set_xlabel('Incident Angle (Degrees)')
        ax1.set_ylabel('Photon Energy (eV)')
        f.colorbar(im1, ax=ax1)

        ax2 = f.add_subplot(122)
        im2 = ax2.imshow(self.RTM, origin='lower', cmap='hot', vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
        ax2.set_title('TM Reflectivity')
        ax2.set_xlabel('Incident Angle (Degrees)')
        ax2.set_ylabel('Photon Energy (eV)')
        f.colorbar(im2, ax=ax2)

        ff = plt.figure()
        ax3 = ff.add_subplot(121)
        im3 = ax3.imshow(self.TTE, origin='lower', cmap='hot', vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
        ax3.set_title('TE Transmission')
        ax3.set_xlabel('Incident Angle (Degrees)')
        ax3.set_ylabel('Photon Energy (eV)')
        ff.colorbar(im3, ax=ax3)

        ax4 = ff.add_subplot(122)
        im4 = ax4.imshow(self.TTM, origin='lower', cmap='hot', vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
        ax4.set_title('TM Transmission')
        ax4.set_xlabel('Incident Angle (Degrees)')
        ax4.set_ylabel('Photon Energy (eV)')
        ff.colorbar(im4, ax=ax4)

        fff = plt.figure()
        ax5 = fff.add_subplot(121)
        im5 = ax5.imshow(1.0-(self.RTE+self.TTE), origin='lower', cmap='hot', vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
        ax5.set_title('TE Absorption')
        ax5.set_xlabel('Incident Angle (Degrees)')
        ax5.set_ylabel('Photon Energy (eV)')
        fff.colorbar(im5, ax=ax5)

        ax6 = fff.add_subplot(122)
        im6 = ax6.imshow(1.0-(self.RTM+self.TTM), origin='lower', cmap='hot', vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
        ax6.set_title('TM Absorption')
        ax6.set_xlabel('Incident Angle (Degrees)')
        ax6.set_ylabel('Photon Energy (eV)')
        fff.colorbar(im6, ax=ax6)

        plt.show()

if __name__ == '__main__':
    import time

    # define lossy materials to show absorption (this is fake loss)
    nair = 1.0
    nsio2 = 1.4 + 0.005j
    nsi = 3.5 + 0.005j
    #nsio2 = 1.4
    #nsi = 3.5
    nau = 0.5+11*1j

    # We build a linearly chirped Bragg mirror
    air_front = geometry.Layer('air_front', 0.0, nair, False)
    discretechirp = geometry.LinearChirp('discretechirp',0.2, 0.6, 5, nsi, nsio2, 3.5, 1.4, False)
    air_back = geometry.Layer('air_back', 0.0, nair, False)
    layers = [air_front, discretechirp, air_back]
   
    # Relatively high resolution for calculation, to easily compare total time
    photon_energies = np.linspace(0.1, 0.74, num=6401)
    thetas = np.linspace(0,90, num=361)
    stack = LayerStack(photon_energies, thetas, layers)
    stack.build()

    # Perform a forward calculation, and time it
    start = time.time()
    fom = stack.fom_forward()
    end = time.time()

    if(RANK==0):
        # Plot and print results
        print('time:', end-start)
        stack.print_info()
        stack.plot_fom()
