"""
Run a Reflectivity Optimization!
Call this script using the following:
In a terminal, run:
mpirun -n 8 python run_opt.py
Where "8" is the number of cpu nodes (this can be changed)

The layers are the following:

Air (0nm for boundary condition, fixed)
---------------------------------------
Si (variable thickness)
---------------------------------------
SiO2 (variable thickness)
---------------------------------------
Si (variable thickness)
---------------------------------------
SiO2 (variable thickness)
---------------------------------------
Si (variable thickness)
---------------------------------------
SiO2 (variable thickness)
---------------------------------------
Si (variable thickness)
---------------------------------------
SiO2 (variable thickness)
---------------------------------------
Si (variable thickness)
---------------------------------------
SiO2 (variable thickness)
---------------------------------------
Gold (500nm, fixed)
---------------------------------------
Air (0nm for boundary condition, fixed)


10 variable thickness layers in total.

Note that initial thicknesses of each layer must be set
for gradient descent optimization.

For this initial condition, we choose a simple
Bragg reflector centered on 0.45eV.


The figure of merit is the Planck spectrum
weighted reflectivity at T=1473K assuming the source occupies
the full half plane of incident solid angles.

"""

# We must import a bunch of things from TMatrixOpt:
import numpy as np
import TMatrixOpt
from TMatrixOpt import geometry, optimizer, fomutils
from TMatrixOpt.solve import TMatrix
from TMatrixOpt.physical_constants import *
from TMatrixOpt.parallel_toolkit import RANK, parallel_integral

class LayerStack(TMatrix):
    """
    This class is reponsible for defining the overall merit function 
    and gradient of the merit function.

    In this example, our merit function is the Planck spectrum weighted reflectivity
    across all incident angles assuming an infinite planar source at T=1473K. We also
    take the average of the TE (s) and TM (p) polarized light:

    Let "pe" denote incident photon energy (frequency),
    Let "ia" denote incident angle,
    Let "dpe" denote differential photon energy,
    Let "dia" denote differential incident angle,
    Let "PS(pe)" denote the Planck spectrum as a function of photon energy,
    Let "AF(ia)" denote an angle function as a function of incident angle,
    Let "R_TE(pe, ia)" denote the reflectivity as a function of photon energy and incident angle
        of the TE polarization (s-polarization),
    Let "R_TM(pe, ia)" denote the reflectivity as a function of photon energy and incident angle
        of the TM polarization (p-polarization)

    Then the full weighted reflectivity merit function is:
        Rtot = \int{ \int{ dpe * dia* PS(pe) * AF(ia) * 0.5*(R_TE(pe,ia) + R_TM(pe,ia))} } /
                         \int{ \int{ dpe * dia* PS(pe) * AF(ia)} }

        (Note the second line is dividing the first line)
        (Note 0.5 is to equally average TE and TM polarized light, this can be changed)

    TMatrixOpt will automatically calculate R_TE(pe, ia) and R_TM(pe, ia), thus the user need
    only define pe, ia, PS, and AF and perform the above calculation.


    Furthermore, the user must provide the overall gradient of the merit function with respect to the design
    parameters. In this case the design parameters are just the (10) variable layer thicknesses shown above.
    TMatrixOpt will provide the gradients of the reflectivity at each user-defined photon energy and incident angle,
    thus the user need only tell the optimizer how to sum these gradients into the full gradient. In other words:

    Reuse all quantities defined above,
    Let p represent the design parameters (layer thicknesses),
    Let dp represent a differential design parameter
    Let "dR_TE_dp(pe, ia)" denote the gradient of the reflectivity with respect to the design parameters 
        as a function of photon energy and incident angle of the TE polarization (s-polarization)
    Let "dR_TM_dp(pe, ia)" denote the gradient of the reflectivity with respect to the design parameters 
        as a function of photon energy and incident angle of the TM polarization (p-polarization)

    We then take the gradient of Rtot from above with respect to the design parameters. By linearity
    of gradient:
        dRtot_dp = \int{ \int{ dpe * dia* PS(pe) * AF(ia) * 0.5*(dR_TE_dp(pe,ia) + dR_TM_dp(pe,ia))} } /
                            \int{ \int{ dpe * dia* PS(pe) * AF(ia)} }

    So we see that this is just a simple extension of the merit function, except we might have to be careful
    with the shape (dimension) of the gradient terms when we are performing these integrals.


    Note:
        Notice that this class inherits TMatrix from the TMatrixOpt.solve module
        All optimizations must do so for proper functionality
    """
    def __init__(self, photon_energies, thetas, layers, TM_weight=0.5):
        # initialize TMatrix
        super().__init__(fom_setting='Reflectivity_FORWARD_ONLY')

        # Assumes that user provides photon energy in eV and angles in degrees. We must
        # convert to SI units:
        self.photon_energies = photon_energies*q
        self.thetas = thetas * np.pi/180.0

        # Optional TM weight, changing this can sometimes yield better results.
        # However, note that the "true" weighted reflectivity assumes that this value is 0.5
        self.TM_weight = TM_weight

        # We will track the thicknesses and merit function values as the optimization proceeds
        self.ds_tracker = []
        self.fom_tracker = []

        # The energy weighting function (planck spectrum) and angle weighting functions do not
        # change throughout the optimization (in this case), so it is convenient to only evaluate
        # them once and save the values, We also reshape the arrays for convenience.
        wse = self.energy_weighting_func(self.photon_energies)
        wsa = self.angle_weighting_func(self.thetas)

        self.wse = wse[:, np.newaxis, np.newaxis]
        self.wsa = wsa[np.newaxis, :, np.newaxis]
        
        # The denominator of the merit function does not change (just the intregrals of the
        # weighting functions). We pre-calculate them here:
        self.denominator1 = np.trapz(wse, x=self.photon_energies)
        self.denominator2 = np.trapz(wsa, x=self.thetas)
        
        # Critical: we use the TMatrix.add_layers(.) method:
        self.add_layers(layers)
    
    def input_func(self):
        # THIS IS A REQUIRED METHOD
        # It must be named input_func.
        #
        # In this case, it simply returns the user provided photon energies and incident angles
        #
        # More advanced users may want to change the values provided to the optimizer
        # per iteration. That can be done here.
        return self.photon_energies, self.thetas

    def calc_fom(self, RTE, RTM):
        # THIS IS A REQUIRED METHOD
        # For reflectivity calculations it must have function signature calc_fom(self, RTE, RTM)
        # For different types of calculations, different function signature would be needed.
        #
        # Note that RTE has shape (len(pe), len(ia), 1)
        # Note that RTM has shape (len(pe), len(ia), 1)
        #
        # NOTE: This method only runs on the master node

        # We evaluate Eq.(1) from above.
        TM_weight = self.TM_weight
       
        int1_TE = np.trapz(RTE*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TE = np.trapz(int1_TE, x=self.thetas, axis=0)

        int1_TM = np.trapz(RTM*self.wse*self.wsa, x=self.photon_energies, axis=0)
        int2_TM = np.trapz(int1_TM, x=self.thetas, axis=0)

        # The total reflectivity. Note that this is the OPTIMIZATION figure of merit
        Rtot = ((1.0-TM_weight) * int2_TE + TM_weight * int2_TM) / (self.denominator1 * self.denominator2)

        # This is the figure of merit of interest to the user (TE and TM weighted equally)
        user_fom = 0.5 * (int2_TE + int2_TM) / (self.denominator1 * self.denominator2)

        # We save the RTE and RTM arrays for analysis uses later.
        self.RTE = RTE.squeeze()
        self.RTM = RTM.squeeze()

        # We save the user fom:
        self.fom_tracker.append(user_fom)
        
        # We save the layer thicknesses for this iteration (we use the TMatrix attribute TMatrix.d)
        self.ds_tracker.append(self.d)

        return -1*Rtot # Take the negative (optimization assumes a minimization problem)
        

    def calc_grads(self, dRTE_dp, dRTM_dp):
        pass

    def energy_weighting_func(self, Ep, T=1473.0):
        # Planck spectrum with photon energy Ep and temperature T
        weight = 2*Ep**3/(c**2*h**3*(np.exp(Ep/(Kb*T))-1))
        return weight

    def angle_weighting_func(self, thetas):
        # Angle weighting function for a source occupying the full half space of solid angles
        weight = np.cos(thetas)*np.sin(thetas)
        return weight

    def plot_fom(self):
        # Plot the layer thicknesses tracked over the optimization
        #ds = np.array(self.ds_tracker)
        import matplotlib.pyplot as plt
        #f1 = plt.figure()
        #ax1 = f1.add_subplot(111)
        #ax1.plot(ds*1e6)
        #ax1.set_xlabel('Iteration')
        #ax1.set_ylabel('Layer Thicknesses(um)')
        #
        ## Plot the foms tracked over the optimization
        #f2 = plt.figure()
        #ax2 = f2.add_subplot(111)
        #ax2.plot(self.fom_tracker)
        #ax2.set_xlabel('Iteration')
        #ax2.set_ylabel('Average Reflectivity')

        # Plot the final RTE and RTM arrays
        pe, ia = self.input_func()
        pe1 = pe[0]/q
        pe2 = pe[-1]/q
        ia1 = ia[0]*180.0/np.pi
        ia2 = ia[-1]*180.0/np.pi

        f3 = plt.figure(figsize=(12,6))
        ax3 = f3.add_subplot(121)
        im3 = ax3.imshow(100*self.RTE, extent=[ia1, ia2, pe1, pe2], aspect='auto', origin='lower', cmap='gnuplot2', vmin=0.0, vmax=100.0)
        f3.colorbar(im3)
        ax3.set_xlabel('Incident Angle (degrees)')
        ax3.set_ylabel('Photon Energy (eV)')
        ax3.set_title('TE Reflectivity (%)')
        ax3.set_xticks(np.linspace(0.0,90.0,num=10))
        ax3.set_yticks(np.linspace(0.1,0.75,num=14))

        ax4 = f3.add_subplot(122)
        im4 = ax4.imshow(100*self.RTM, extent=[ia1, ia2, pe1, pe2], aspect='auto', origin='lower', cmap='gnuplot2', vmin=0.0, vmax=100.0)
        f3.colorbar(im4)
        ax4.set_xlabel('Incident Angle (degrees)')
        ax4.set_ylabel('Photon Energy (eV)')
        ax4.set_title('TM Reflectivity (%)')
        ax4.set_xticks(np.linspace(0.0,90.0,num=10))
        ax4.set_yticks(np.linspace(0.1,0.75,num=14))

        #plt.show()
        #plt.savefig('10-Layer-Mirror_Final_Reflectivity.pdf')
        #plt.savefig('10-Layer-Mirror_Initial_Reflectivity.png')
        plt.savefig('10-Layer-Mirror_Final_Reflectivity.png')

if __name__ == '__main__':
    """
    Here we actually define the layer geometry and set up the optimization

    mpirun -n 20 python run_opt.py
    """
    # For saving data later:
    import scipy.io

    data = scipy.io.loadmat('results.mat')
    ds = data['ds']
    #ds_init = ds[0,1:-1].squeeze()
    #ds_final = ds[-1,1:-1].squeeze()
    ds_init = ds[-1,1:-1].squeeze()

    # We define the material refractive index values.
    # The solver supports complex materials.
    # If dispersive materials are required, please see TMatrixOpt.geometry
    # for how to define them.
    nair = 1.0 # air
    nsi = 3.5 # silicon
    nsio2 = 1.4 # silicon dioxide

    # See layer stack at top of this script. We define each layer
    # using the TMatrixOpt.geometry module
    layers = []
    air_front = geometry.Layer('air_front', 0.0, nair, False)

    for i in range(10):
        if i%2 == 0:
            layer = geometry.Layer('layer', ds_init[i], nsi, False)
        else:
            layer = geometry.Layer('layer', ds_init[i], nsio2, False)
        layers.append(layer)

    air_back = geometry.Layer('air_back', 0.0, nair, False)

    layers = [air_front] + layers + [air_back]
   
    # We define the desired photon energies and incident angles for calculation
    #photon_energies = np.linspace(0.1, 0.8, num=14001)
    #thetas = np.linspace(0,90, num=901)
    #photon_energies = np.linspace(0.1, 0.75, num=2601)
    #thetas = np.linspace(0,90, num=361)
    photon_energies = np.linspace(0.1, 0.75, num=6501)
    thetas = np.linspace(0,90, num=460)

    # We create the LayerStack
    stack = LayerStack(photon_energies, thetas, layers, TM_weight=0.5)

    # Important, we must run stack.build() from the solve module:
    stack.build()

    # For convenience, we get a vector of parameters with elements equal to the number
    # of designable parameters (in this case, number of variable thickness layers)
    #params = stack.param_vec()

    fom = stack.fom_forward()

    # We save our final results, only the main node needs to save this data
    if RANK == 0:
        stack.plot_fom()
