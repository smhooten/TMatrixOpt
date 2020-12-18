import numpy as np
import geometry
import optimizer
from solve import TMatrix
import fomutils
from physical_constants import *
from parallel_toolkit import RANK, parallel_integral
import scipy.optimize
#from custom_optimizer import momentum

class LayerStack(TMatrix):
    def __init__(self, photon_energies, thetas, layer, TM_weight=0.5, save_data = None, fom_setting='Transmission'):
        super().__init__(fom_setting=fom_setting)
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
    
    def input_func(self):
        return self.photon_energies, self.thetas

    def calc_fom(self, TTE, TTM):
        TM_weight = self.TM_weight

        int1_TE = np.trapz(TTE*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TE = np.trapz(int1_TE, x=self.thetas, axis=0)

        int1_TM = np.trapz(TTM*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TM = np.trapz(int1_TM, x=self.thetas, axis=0)

        if self.save_data is not None:
            if self.fom_counter % self.save_data == 0:
                self.TTE = TTE.squeeze()
                self.TTM = TTM.squeeze()

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
        

    def calc_grads(self, dTTE_dp, dTTM_dp):
        TM_weight = self.TM_weight

        #int1_TE = parallel_integral(dRTE_dp*self.wse*self.wsa, self.photon_energies, axis=0)
        #int2_TE = parallel_integral(int1_TE, self.thetas, axis=0)

        #int1_TM = parallel_integral(dRTM_dp*self.wse*self.wsa, self.photon_energies, axis=0)
        #int2_TM = parallel_integral(int1_TM, self.thetas, axis=0)

        int1_TE = np.trapz(dTTE_dp*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TE = np.trapz(int1_TE, x=self.thetas, axis=0)

        int1_TM = np.trapz(dTTM_dp*self.wse*self.wsa, x=self.photon_energies, axis=0)
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
        ax3.imshow(self.TTE, aspect='auto')
        ax4 = f3.add_subplot(122)
        ax4.imshow(self.TTM, aspect='auto')

        plt.show()

if __name__ == '__main__':
    import time
    import scipy.io

    air_front = geometry.Layer('air_front', 0.0, 1.0, False)
    #superstrate = geometry.Layer('superstrate', 0.4668e-6, 3.5, False)
    #layer2 = geometry.LinearChirp('linearchirp', 0.38, 1.6, 5, 1.6, 3.5, 1.6, 3.5, True)
    #layer2 = geometry.DiscreteChirp_BETA('linearchirp', 0.45, 6, 1.6, 3.5, 1.6, 3.5, True)
    #metal_back = geometry.Layer('metal_back', 0.2e-6, 0.5+11*1j, False)
    air_test = geometry.Layer('air_test', 1e-6, 1.0, True)
    semi_test2 = geometry.Layer('semi_test2', 1e-8, 3.5, True)
    air_back = geometry.Layer('air_back', 0.0, 1.0, False)
    layers = [air_front, air_test, semi_test2, air_back]
    #layers = [air_front, superstrate, layer2, air_back]
   
    photon_energies = np.linspace(0.1, 0.74, num=1000)
    thetas = np.linspace(0,90, num=90)
    stack = LayerStack(photon_energies, thetas, layers, TM_weight=0.5, save_data=5, fom_setting='Transmission')
    stack.build()

    params = stack.param_vec()

    #steps = np.array([1e-5, 5e-10, 5e-10, 5e-10, 5e-10, 5e-10, 5e-10, 5e-10, 5e-10])
    #jstack.check_gradient(params, step=steps)
    start = time.time()
    stack.check_gradient(params, step=5e-10)
    end = time.time()
    print('time')
    print(end-start)

    #lb = -1.0e-2*np.ones((2,1))
    #ub = 2.0e-2*np.ones((2,1))
    #bounds = scipy.optimize.Bounds(lb, ub)
    bounds = None

    callback = lambda p: print(stack.FOM)
    opt = optimizer.Optimizer(stack, params, callback_func=callback, opt_method='BFGS', Nmax=100, tol=1e-10, bounds=bounds, scipy_verbose = None)
    #opt = momentum(stack, params, callback_func=callback, opt_method='BFGS', Nmax=1000, tol=1e-10, bounds=bounds, scipy_verbose = None)
    fom_final, params_final = opt.run()
    if RANK == 0:
        best_results = stack.best_results
        fom_final = best_results[0]
        ds_final = best_results[1]
        to_save = {'fom':fom_final, 'd':ds_final}
        scipy.io.savemat('results_trans_Np5.mat', to_save)
        stack.print_info()
        stack.plot_fom()
