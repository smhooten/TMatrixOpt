#include <iostream> 
#include <complex>

#ifndef __SOLVE_HPP__
#define __SOLVE_HPP__

typedef struct struct_complex64 {
    double real, imag;

    struct_complex64& operator=(std::complex<double> val) {
        //struct_complex64 output;

        real = val.real();
        imag = val.imag();
        return *this;
    }
} complex64;

extern "C" {

    void solve(double photon_energy, 
               double theta, 
               int len_d, 
               int len_idx,
               double* d_i, 
               std::complex<double>* n_i, 
               int* idx_i, 
               complex64* M_TE_o,
               complex64* M_TM_o,
               complex64* dM_dd_TE_o,
               complex64* dM_dd_TM_o);

    void solve_forward(double photon_energy, 
                       double theta, 
                       int len_d, 
                       int len_idx,
                       double* d_i, 
                       std::complex<double>* n_i, 
                       int* idx_i, 
                       complex64* M_TE_o,
                       complex64* M_TM_o);


/*
    void reflectivity_grads(std::complex<double>* rTE,
                            std::complex<double>* rTM,
                            std::complex<double>* tTE,
                            std::complex<double>* tTM,
                            std::complex<double>* dM11_dp_TE,
                            std::complex<double>* dM11_dp_TM,
                            std::complex<double>* dM21_dp_TE,
                            std::complex<double>* dM21_dp_TM,
                            int len_pe,
                            int len_theta,
                            int len_par,
                            double* dRTE_dp,
                            double* dRTM_dp);
*/

};

#endif
