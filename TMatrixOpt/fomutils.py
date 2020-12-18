from abc import ABCMeta, abstractmethod
import numpy as np

def GaAsAbsorpCoeff(E):
    # inputs
    # E - photon energy
    # alpha0 = inverse of absorption depth in GaAs, in 1/m,
    # E0 - Urbach energy
    # Eprime - above-band edge parameter
    
    # outputs
    # alpha - absorption coefficient in units of 1/m
    q = 1.602e-19; # electron charge
    Eg = 1.42*q # bandgap
    alpha0 = 8e5
    EPrime = 140e-3*q
    E0 = 6.7e-3*q
    
    
    alpha = np.zeros((E.shape))
    alpha[E<=Eg] = alpha0*np.exp((E[E<=Eg]-Eg)/E0)
    alpha[E>Eg] = alpha0*(1+(E[E>Eg]-Eg)/EPrime)
    
    return alpha

def PlanckPhotonCounts(E,V,n,T):
    
    
    #Inputs
    # E - photon energy
    # V - applied bias
    # n - refractive index
    # T - temperature of the object
    
    #Ouputs
    # PhotonFlux - #Photon Nos/(m^2*s*J)
    
    q = 1.602e-19
    c = 3e8
    h = 6.626e-34
    Kb = 1.38e-23
    
    PhotonFlux = 8*np.pi*n**2/(c**2*h**3)*E**2*np.exp((q*V-E)/(Kb*T))
    
    return PhotonFlux

class MeritFunction:
    @abstractmethod
    def calc_fom(**results):
        pass

    @abstractmethod
    def calc_grads(**results):
        pass

class Reflectivity(MeritFunction):
    return_results = ['rTE',
                      'rTM',
                      'tTE',
                      'tTM',
                      'dM11_dp_TE',  
                      'dM11_dp_TM',  
                      'dM21_dp_TE',  
                      'dM21_dp_TM']

    @staticmethod
    def calc_fom(rTE, rTM):
        RTE = (rTE*np.conj(rTE)).real
        RTM = (rTM*np.conj(rTM)).real
        return {'RTE': RTE, 'RTM': RTM}

    # USER FUNCTION SIGNATURE SHOULD BE
    # def calc_fom(RTE, RTM)

    @staticmethod
    def calc_grads(rTE, rTM, tTE, tTM, dM11_dp_TE, dM11_dp_TM,
                   dM21_dp_TE, dM21_dp_TM):

        drTE_dp = tTE * dM21_dp_TE - rTE * tTE * dM11_dp_TE
        drTM_dp = tTM * dM21_dp_TM - rTM * tTM * dM11_dp_TM

        dRTE_dp = 2*(np.conj(rTE)*drTE_dp).real
        dRTM_dp = 2*(np.conj(rTM)*drTM_dp).real

        return {'dRTE_dp': dRTE_dp, 'dRTM_dp': dRTM_dp}

    # USER FUNCTION SIGNATURE SHOULD BE
    # def calc_grads(dRTE_dp, dRTM_dp)

class Reflectivity_FORWARD_ONLY(MeritFunction):
    return_results = ['rTE',
                      'rTM']

    @staticmethod
    def calc_fom(rTE, rTM):
        RTE = (rTE*np.conj(rTE)).real
        RTM = (rTM*np.conj(rTM)).real
        return {'RTE': RTE, 'RTM': RTM}

    # USER FUNCTION SIGNATURE SHOULD BE
    # def calc_fom(RTE, RTM)

    @staticmethod
    def calc_grads():
        pass


### CURRENTLY NOT WORKING BELOW ###
class Transmission(MeritFunction):
    return_results = ['rTE',
                      'rTM',
                      'tTE',
                      'tTM',
                      'dM11_dp_TE',  
                      'dM11_dp_TM',  
                      'dM21_dp_TE',  
                      'dM21_dp_TM']

    @staticmethod
    def calc_fom(rTE, rTM):
        TTE = 1.0-(rTE*np.conj(rTE)).real
        TTM = 1.0-(rTM*np.conj(rTM)).real
        return {'TTE': TTE, 'TTM': TTM}

    # USER FUNCTION SIGNATURE SHOULD BE
    # def calc_fom(TTE, TTM)

    @staticmethod
    def calc_grads(rTE, rTM, tTE, tTM, dM11_dp_TE, dM11_dp_TM,
                   dM21_dp_TE, dM21_dp_TM):

        drTE_dp = tTE * dM21_dp_TE - rTE * tTE * dM11_dp_TE
        drTM_dp = tTM * dM21_dp_TM - rTM * tTM * dM11_dp_TM

        dTTE_dp = -2*(np.conj(rTE)*drTE_dp).real
        dTTM_dp = -2*(np.conj(rTM)*drTM_dp).real

        return {'dTTE_dp': dTTE_dp, 'dTTM_dp': dTTM_dp}

    # USER FUNCTION SIGNATURE SHOULD BE
    # def calc_grads(dTTE_dp, dTTM_dp)

"""
    @staticmethod
    def calc_grads_old(rTE, rTM, tTE, tTM, dM11_dp_TE, dM11_dp_TM,
                   dM21_dp_TE, dM21_dp_TM):

        #drTE_dp = tTE * dM21_dp_TE - rTE * tTE * dM11_dp_TE
        #drTM_dp = tTM * dM21_dp_TM - rTM * tTM * dM11_dp_TM

        #dRTE_dp = 2*(np.conj(rTE)*drTE_dp).real
        #dRTM_dp = 2*(np.conj(rTM)*drTM_dp).real
        len_pe, len_theta, len_par = dM11_dp_TE.shape

        dRTE_dp = np.empty(len_pe*len_theta*len_par, dtype=np.double)
        dRTM_dp = np.empty(len_pe*len_theta*len_par, dtype=np.double)

        lib.reflectivity_grads(rTE.ravel().astype(np.complex128),
                               rTM.ravel().astype(np.complex128),
                               tTE.ravel().astype(np.complex128),
                               tTM.ravel().astype(np.complex128),
                               dM11_dp_TE.ravel().astype(np.complex128),
                               dM11_dp_TM.ravel().astype(np.complex128),
                               dM21_dp_TE.ravel().astype(np.complex128),
                               dM21_dp_TM.ravel().astype(np.complex128),
                               len_pe,
                               len_theta,
                               len_par,
                               dRTE_dp,
                               dRTM_dp)

        return {'dRTE_dp': dRTE_dp.reshape((len_pe, len_theta, len_par)), 'dRTM_dp': dRTM_dp.reshape((len_pe, len_theta, len_par))}
"""
