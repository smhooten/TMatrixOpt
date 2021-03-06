"""
TMatrixOpt: A fast and modular transfer-matrix optimization package 
for 1D optical devices.
Copyright (C) 2021 Sean Hooten & Zunaid Omair

TMatrixOpt/merit_functions.py

TMatrixOpt by default can return all Fresnel coefficients
to the user, as well as derivatives of the matrix elements of the
transfer matrix system. This can be inconvenient to a user who's
overall merit function is only related to, for example, the 
reflection/transmission fractions of power. 

This module provides a convenient intermediate layer between the raw 
output of TMatrixOpt and the user who may only be interested in an
average over structure reflectivity, for example. As mentioned in 
TMatrixOpt/solve.py, the user needs to override the calc_fom and
calc_grads methods of the TMatrix base class. By initializing 
TMatrix specifiying one of these classes, e.g.,
    super().__init__(fom_setting='Reflectivity')
The user then need only define their calc_fom method with the 
signature,
    calc_fom(RTE, RTM)
where RTE and RTM are the power coefficients of reflectivity for TE (s) 
and TM (p) polarized light respectively defined over all input
photon energies and angles of incidence specified by the user from
the input_func method.

Using one of these intermediary is classes is not necessary. If the
user desires to use the Fresnel coefficients or transfer matrix
elements directly, the user may initialize the TMatrix class with
the 'Custom' fom_setting. See examples/lumerical_comparison for
an example of this. Moreover, one may define their own custom
intermediary merit function class here, it need only inherit the
MeritFunction base class below and override the abstract methods.
"""

__author__ = 'Sean Hooten'
__version__ = '1.0'
__license__ = 'GPL 3.0'

from abc import ABCMeta, abstractmethod
import numpy as np

class MeritFunction:
    """
    Define general framework for an intermediate MeritFunction
    class. Any MeritFunction should inherit this base class.
    
    A MeritFunction class simplifies the user's responsibility 
    to handle the TMatrixOpt output (at an insignificant speed
    cost). The meanings of the Fresnel coefficients and the
    program's matrix elements may be opaque to a general end
    user, so these classes sidestep that responsible and only
    provide the quantities of interest to the user. For example,
    the Reflectivity class will automatically calculate the
    Power Reflection Coefficient as a function of user provided
    incident photon energies and angles, as well as its derivative with
    respect to the design parameters. Therefore, the user need
    only define there overall figure of merit and system derivative
    with respect to R.
    """
    @abstractmethod
    def calc_fom(**results):
        pass

    @abstractmethod
    def calc_grads(**results):
        pass

class Reflectivity(MeritFunction):
    """
    Simplifies optimizations over the structure
    power coefficient of reflectivity for TE and TM
    polarized light.

    User function signatures of calc_fom and calc_grads
    are defined below.
    """
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
    """
    If the user only needs to perform a forward calculation
    of reflectivity (i.e. no gradients required), the user
    may use this convenience class.

    User function signatures of calc_fom and calc_grads
    are defined below.
    """
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

    # USER FUNCTION SIGNATURE SHOULD BE
    # def calc_grads():
    #     pass


class Transmission(MeritFunction):
    """
    Simplifies optimizations over the structure
    power coefficient of reflectivity for TE and TM
    polarized light.

    CAUTION: This should only be used with lossless
    (i.e. noncomplex refractive index) structures.
    Lossy structures are a bit more involved, and transmission
    cannot necessarily be calculated without additional
    information provided by the user.
    Providing a general framework for transmission is in
    development. For now, if a demonstration of calculating
    transmission for a lossy structure is needed, please see
    examples/lumerical_comparison/lumerical_comparison.py

    User function signatures of calc_fom and calc_grads
    are defined below.
    """
    return_results = ['rTE',
                      'rTM',
                      'dM11_dp_TE',  
                      'dM11_dp_TM']

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


# Some additional convenience functions below

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
