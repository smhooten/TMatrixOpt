import numpy as np
import geometry
import optimizer
from solve import TMatrix
import fomutils
from physical_constants import *
from parallel_toolkit import RANK, parallel_integral
#import scipy.optimize
#from custom_optimizer import momentum

class LayerStack(TMatrix):
    def __init__(self, photon_energies, thetas, layer, TM_weight=0.5, save_data = None):
        super().__init__()
        self.photon_energies = photon_energies*q
        self.thetas = thetas * np.pi/180.0
        self.TM_weight = TM_weight

        self.save_data = save_data
        self.fom_counter = 0
        self.ds_tracker = []
        self.fom_tracker = []
        self.best_results = [[], []]

        wse = self.energy_weighting_func(self.photon_energies)
        wsa = self.angle_weighting_func(self.thetas)

        self.wse = wse[:, np.newaxis, np.newaxis]
        self.wsa = wsa[np.newaxis, :, np.newaxis]
        
        #if RANK == 0:
        #    import matplotlib.pyplot as plt
        #    f = plt.figure()
        #    ax = f.add_subplot(111)
        #    #bla = (self.wse*self.wsa).reshape(self.wse.shape[:-1])
        #    #ax.imshow(bla, extent=[1, 90, 0.74, 0.1], aspect='auto')
        #    ax.plot(photon_energies, wse)
        #    plt.show()

        self.denominator1 = np.trapz(wse, x=self.photon_energies)
        self.denominator2 = np.trapz(wsa, x=self.thetas)
        
        self.add_layers(layer)
    
    def input_func(self,*arg):
        if (len(arg)<3):
          delE = 0
          delTheta = 0
        else:
          delE = arg[0]*q
          delTheta = arg[1]*np.pi/180
        self.photon_energies+=delE # introducing frequency hopping
        self.thetas+=delTheta # introducing angle hopping
        return self.photon_energies, self.thetas

    def calc_fom(self, RTE, RTM):
        TM_weight = self.TM_weight

        int1_TE = np.trapz(RTE*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TE = np.trapz(int1_TE, x=self.thetas, axis=0)

        int1_TM = np.trapz(RTM*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TM = np.trapz(int1_TM, x=self.thetas, axis=0)

        if self.save_data is not None:
            if self.fom_counter % self.save_data == 0:
                self.RTE = RTE.squeeze()
                self.RTM = RTM.squeeze()

            fom_val = (0.5*int2_TE+0.5*int2_TM)/(self.denominator1*self.denominator2)
            self.fom_tracker.append(fom_val)
            self.ds_tracker.append(self.d)

            if all(fom_val >= f for f in self.fom_tracker):
                self.best_results[0] = fom_val
                self.best_results[1] = self.d

            self.fom_counter += 1

        #int1_TE = parallel_integral(RTE*self.wse*self.wsa, self.photon_energies, axis=0)
        #int2_TE = parallel_integral(int1_TE, self.thetas, axis=0)

        #int1_TM = parallel_integral(RTM*self.wse*self.wsa, self.photon_energies, axis=0)
        #int2_TM = parallel_integral(int1_TM, self.thetas, axis=0)

        tot = ((1.0-TM_weight) * int2_TE + TM_weight * int2_TM) / (self.denominator1 * self.denominator2)

        return -tot
        

    def calc_grads(self, dRTE_dp, dRTM_dp):
        TM_weight = self.TM_weight

        #int1_TE = parallel_integral(dRTE_dp*self.wse*self.wsa, self.photon_energies, axis=0)
        #int2_TE = parallel_integral(int1_TE, self.thetas, axis=0)

        #int1_TM = parallel_integral(dRTM_dp*self.wse*self.wsa, self.photon_energies, axis=0)
        #int2_TM = parallel_integral(int1_TM, self.thetas, axis=0)

        int1_TE = np.trapz(dRTE_dp*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TE = np.trapz(int1_TE, x=self.thetas, axis=0)

        int1_TM = np.trapz(dRTM_dp*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TM = np.trapz(int1_TM, x=self.thetas, axis=0)

        tot = ((1.0-TM_weight) * int2_TE + TM_weight * int2_TM) / (self.denominator1 * self.denominator2)

        return -tot

    def energy_weighting_func(self, Ep, T=1473.0):
        weight = 2*Ep**3/(c**2*h**3*(np.exp(Ep/(Kb*T))-1))
        return weight

    def angle_weighting_func(self, thetas):
        weight = np.cos(thetas)*np.sin(thetas)
        return weight

    def plot_fom(self):
        ds = self.ds_tracker
        import matplotlib.pyplot as plt
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(ds)
        
        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        ax2.plot(self.fom_tracker)

        f3 = plt.figure()
        ax3 = f3.add_subplot(121)
        ax3.imshow(self.RTE, aspect='auto')
        ax4 = f3.add_subplot(122)
        ax4.imshow(self.RTM, aspect='auto')

        plt.show()

if __name__ == '__main__':
    import time
    import scipy.io

    layer0 = geometry.Layer('0', 0.0, 1.0, False)
    layer1 = geometry.Layer('1', 1e-6, 1.5, True)
    layer2 = geometry.Layer('2', 2e-6, 3.5, True)
    layer3 = geometry.Layer('3', 3e-6, 1.5, True)
    layer4 = geometry.Layer('4', 0.0, 1.0, False)
    layers = [layer0, layer1, layer2, layer3, layer4]
   
    photon_energies = np.array([1.0])
    thetas = np.array([0.4 * 180.0/np.pi])
    stack = LayerStack(photon_energies, thetas, layers, TM_weight=0.5, save_data=5)
    stack.build()

    stack.solve(stack.photon_energies, stack.thetas)

