import numpy as np
import TMatrixOpt
from TMatrixOpt import geometry, optimizer, solve, fomutils, solve
from TMatrixOpt.solve import TMatrix
from TMatrixOpt.physical_constants import *
from TMatrixOpt.parallel_toolkit import RANK, parallel_integral

class LayerStack(TMatrix):
    def __init__(self, photon_energies, thetas, layers, allreal=False):
        self.allreal = allreal
        return_results = ['rTE', 'rTM', 'tTE', 'tTM']

        super().__init__(fom_setting = 'Custom', return_results = return_results)

        self.photon_energies = photon_energies*q
        self.thetas = thetas * np.pi/180.0

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

        self.n1 = n1
        self.n2 = n2

        self.add_layers(layers)
    
    def input_func(self):
        return self.photon_energies, self.thetas

    def calc_fom(self, rTE, rTM, tTE, tTM):
        RTE = (rTE*np.conj(rTE)).real
        RTM = (rTM*np.conj(rTM)).real

        if self.allreal:
            TTE = 1.0-RTE
            TTM = 1.0-RTM
        else:
            costheta = np.cos(self.thetas)
            coeff = np.sqrt( (self.n2/ self.n1)**2 + costheta**2 - 1.0 ) / costheta
            coeff = coeff[np.newaxis, :, np.newaxis]

            TTE = coeff*(tTE*np.conj(tTE)).real
            TTM = coeff*(tTM*np.conj(tTM)).real

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
        pass

    #def energy_weighting_func(self, Ep, T=1473.0):
    #    weight = 2*Ep**3/(c**2*h**3*(np.exp(Ep/(Kb*T))-1))
    #    return weight

    #def angle_weighting_func(self, thetas):
    #    weight = np.cos(thetas)*np.sin(thetas)
    #    return weight

    def plot_fom(self):
        import matplotlib.pyplot as plt
        pe = self.photon_energies / q
        thetas = self.thetas * 180.0 / np.pi
        extent = [thetas[0], thetas[-1], pe[0], pe[-1]]

        f = plt.figure()
        ax1 = f.add_subplot(121)
        im1 = ax1.imshow(self.RTE, origin='lower', cmap='hot', vmin=0.9, vmax=1.0, extent=extent, aspect='auto')
        ax1.set_title('TE Reflectivity')
        ax1.set_xlabel('Incident Angle (Degrees)')
        ax1.set_ylabel('Photon Energy (eV)')
        f.colorbar(im1, ax=ax1)

        ax2 = f.add_subplot(122)
        im2 = ax2.imshow(self.RTM, origin='lower', cmap='hot', vmin=0.9, vmax=1.0, extent=extent, aspect='auto')
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
    nair = 1.0
    #nsio2 = 1.4 + 0.01j
    #nsi = 3.5 + 0.01j
    nsio2 = 1.4
    nsi = 3.5
    nau = 0.5+11*1j

    air_front = geometry.Layer('air_front', 0.0, nair, False)
    #discretechirp = geometry.DiscreteChirp('discretechirp', 0.45, 10, nsi, nsio2, 3.5, 1.4, False)
    discretechirp = geometry.LinearChirp('discretechirp',0.2, 0.6, 5, nsi, nsio2, 3.5, 1.4, False)
    metal_back = geometry.Layer('metal_back', 0.5e-6, 0.5+11*1j, False)
    air_back = geometry.Layer('air_back', 0.0, nair, False)
    #layers = [air_front, superstrate, layer2, metal_back, air_back]
    #layers = [air_front, superstrate, layer2, air_back]
    #layers = [air_front, discretechirp, air_back]
    layers = [air_front, discretechirp, metal_back, air_back]
    #layers = [air_front, layer2, air_back]
   
    photon_energies = np.linspace(0.1, 0.74, num=1281)
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

