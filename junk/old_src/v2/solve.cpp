#include "solve.hpp"
#include <Eigen/Core> 
#include <Eigen/Dense> 
#include <cmath>
//#include <math.h>

using namespace Eigen;

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
           complex64* dM_dd_TM_o) 
    {

    std::complex<double> j(0.0,1.0);
    double h = 6.626070040e-34;
    double c = 299792458.0;
    double wavelength = h * c / photon_energy;

    Matrix2cd Idash;
    Idash(0,0) = -1.0;
    Idash(0,1) = 0.0;
    Idash(1,0) = 0.0;
    Idash(1,1) = 1.0;

    // copy inputs to arrays for convenience
    //ArrayXd n(len_d);
    //ArrayXd d(len_d);
    //ArrayXi idx(len_idx);

    //for(int i = 0; i < len_d; ++i) {
    //    n(i) = n_i[i];
    //    d(i) = d_i[i];
    //    if(i<len_idx) {
    //        idx(i) = idx_i[i];
    //    };
    //};


    ArrayXcd angles(len_d);
    ArrayXcd Kx;
    angles(0) = theta;
    Kx(0) = n_i[0];
    for(int i = 1; i < len_d; ++i) {
        angles(i) = std::asin(n_i[i-1] * std::sin(angles(i-1)) / n_i[i]);
        Kx(i) = n_i[i];
    };
    Kx = Kx*2*M_PI/wavelength*angles.cos();

    ArrayXcd L = Kx(seq(1,last))/Kx(seq(0,last-1));

    Matrix2cd M_TE;
    Matrix2cd M_TM;
    M_TE << 1.0, 0.0,
            0.0, 1.0;
    M_TM << 1.0, 0.0,
            0.0, 1.0;

    Matrix2cd X_TE[len_idx];
    Matrix2cd X_TM[len_idx];
    Matrix2cd T_TE;
    Matrix2cd T_TM;
    Matrix2cd P;
    Matrix2cd B_TE;
    Matrix2cd B_TM;

    int counter = 0;
    for(int i = 0; i < len_d-1; ++i) {
        T_TE(0,0) = 1.0+L(i); 
        T_TE(0,1) = 1.0-L(i); 
        T_TE(1,0) = 1.0-L(i); 
        T_TE(1,1) = 1.0+L(i); 
        T_TE *= 0.5;

        T_TM(0,0) = L(i)*n_i[i]/n_i[i+1]+n_i[i+1]/n_i[i];
        T_TM(0,1) = L(i)*n_i[i]/n_i[i+1]-n_i[i+1]/n_i[i];
        T_TM(1,0) = L(i)*n_i[i]/n_i[i+1]-n_i[i+1]/n_i[i];
        T_TM(1,1) = L(i)*n_i[i]/n_i[i+1]+n_i[i+1]/n_i[i];
        T_TM *= 0.5;

        P(0,0) = exp(-j*Kx(i+1)*d_i[i+1]);
        P(1,0) = 0.0;
        P(0,1) = 0.0;
        P(1,1) = exp(j*Kx(i+1)*d_i[i+1]);

        B_TE = T_TE*P;
        B_TM = T_TM*P;

        for(int k = 0; k<len_idx; k++) {
            if(i+1==idx_i[k]) {
                X_TE[counter] = M_TE * T_TE;
                X_TM[counter] = M_TM * T_TM;
                ++counter;
                break;
            }
        }
        M_TE = M_TE * B_TE;
        M_TM = M_TM * B_TM;
    }

    Matrix2cd dM_dd_TE[len_idx];
    Matrix2cd dM_dd_TM[len_idx];

    counter = 0;
    for(int i = 0; i < len_idx; ++i) {
        int k = idx_i[i];

        Matrix2cd A_TE = X_TE[counter].colPivHouseholderQr().solve(M_TE);
        Matrix2cd A_TM = X_TM[counter].colPivHouseholderQr().solve(M_TM);

        dM_dd_TE[counter] = j * Kx[k] * X_TE[counter] * Idash * A_TE;
        dM_dd_TM[counter] = j * Kx[k] * X_TM[counter] * Idash * A_TM;

        ++counter;
    }

    for(int m=0; m<2; m++) {
        for(int n=0; n<2; n++) {
            M_TE_o[m*2 + n] = M_TE(m,n);
            M_TM_o[m*2 + n] = M_TM(m,n);
            for(int o=0; o<len_idx; o++) {
                dM_dd_TE_o[len_idx*2*m + len_idx*n + o] = dM_dd_TE[o](m,n);
                dM_dd_TM_o[len_idx*2*m + len_idx*n + o] = dM_dd_TM[o](m,n);
            }
        }
    }

    //for(int k = 0; k<12; k++) {
    //    std::cout << dM_dd_TE_o[k].real << std::endl;
    //    std::cout << dM_dd_TE_o[k].imag << std::endl;
    //}
   
};


/*
int main() {
    double q = 1.60217662e-19;
    //double q = 1.602e-19;
    double photon_energy = 0.5*q;
    double theta = 0.2;

    int len_d = 5;
    int len_idx = 3;

    double d[len_d];
    d[0] = 0.0;
    d[1] = 1e-6;
    d[2] = 2e-6;
    d[3] = 3e-6;
    d[4] = 0.0;

    double n[len_d];
    n[0] = 1.0;
    n[1] = 1.5;
    n[2] = 3.5;
    n[3] = 1.5;
    n[4] = 1.0;

    int idx[len_idx];
    idx[0] = 1; 
    idx[1] = 2;
    idx[2] = 3;

    complex64 M_TE[4];
    complex64 M_TM[4];

    complex64 dM_dd_TE[12];
    complex64 dM_dd_TM[12];

    //Matrix2cd dM_dd_TE[len_idx];
    //Matrix2cd dM_dd_TM[len_idx];

    solve(photon_energy, theta, len_d, len_idx, d, n, idx, M_TE, M_TM, dM_dd_TE, dM_dd_TM);

    std::cout << M_TE[0].real << std::endl;
    std::cout << M_TE[0].imag << std::endl;
    std::cout << M_TE[1].real << std::endl;
    std::cout << M_TE[1].imag << std::endl;
    std::cout << M_TE[2].real << std::endl;
    std::cout << M_TE[2].imag << std::endl;
    std::cout << M_TE[3].real << std::endl;
    std::cout << M_TE[3].imag << std::endl;
    std::cout << M_TM << std::endl;
    for(int k = 0; k<12; k++) {
        std::cout << dM_dd_TE[k].real << std::endl;
        std::cout << dM_dd_TE[k].imag << std::endl;
    }
    //std::cout << dM_dd_TM << std::endl;

    return 0;
}
*/


