#include <iostream> 
#include <complex>

#ifndef __SOLVE_HPP__
#define __SOLVE_HPP__

typedef struct struct_complex64 {
    double real, imag;

    struct_complex64& operator=(std::complex<double> val) {
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
};

#endif
