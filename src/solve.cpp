/*

TMatrixOpt: A fast and modular transfer-matrix optimization package 
for 1D optical devices.
Copyright (C) 2021 Sean Hooten & Zunaid Omair

TMatrixOpt/solve.cpp

Solves transfer matrix and provies gradients

Authors: Sean Hooten, Zunaid Omair
Version: 1.0
License: GPL 3.0 

*/

#include "solve.hpp"
#include <Eigen/Core> 
#include <Eigen/Dense> 
#include <cmath>

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
    Idash << -1.0, 0.0,
              0.0, 1.0;

    ArrayXcd n(len_d);
    ArrayXd d(len_d);
    ArrayXi idx(len_idx);

    for(int i = 0; i < len_d; ++i) {
        n(i) = n_i[i];
        d(i) = d_i[i];
        if(i<len_idx) {
            idx(i) = idx_i[i];
        };
    };

    ArrayXcd angles(len_d);
    angles(0) = theta;

    for(int i = 1; i < len_d; ++i) {
        angles(i) = std::asin(n(i-1) * std::sin(angles(i-1)) / n(i));
    };

    ArrayXcd Kx = 2*M_PI*n/wavelength*angles.cos();
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

        T_TM(0,0) = L(i)*n(i)/n(i+1)+n(i+1)/n(i);
        T_TM(0,1) = L(i)*n(i)/n(i+1)-n(i+1)/n(i);
        T_TM(1,0) = L(i)*n(i)/n(i+1)-n(i+1)/n(i);
        T_TM(1,1) = L(i)*n(i)/n(i+1)+n(i+1)/n(i);
        T_TM *= 0.5;

        P(0,0) = exp(-j*Kx(i+1)*d(i+1));
        P(1,0) = 0.0;
        P(0,1) = 0.0;
        P(1,1) = exp(j*Kx(i+1)*d(i+1));

        B_TE = T_TE*P;
        B_TM = T_TM*P;

        for(int k = 0; k<len_idx; ++k) {
            if(i+1==idx(k)) {
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
        int k = idx(i);

        Matrix2cd A_TE = X_TE[counter].colPivHouseholderQr().solve(M_TE);
        Matrix2cd A_TM = X_TM[counter].colPivHouseholderQr().solve(M_TM);

        dM_dd_TE[counter] = j * Kx[k] * X_TE[counter] * Idash * A_TE;
        dM_dd_TM[counter] = j * Kx[k] * X_TM[counter] * Idash * A_TM;

        ++counter;
    }

    for(int m=0; m<2; ++m) {
        for(int n=0; n<2; ++n) {
            int ind1 = m*2 + n;
            M_TE_o[ind1] = M_TE(m,n);
            M_TM_o[ind1] = M_TM(m,n);
            for(int o=0; o<len_idx; ++o) {
                int ind2 = len_idx*2*m + len_idx*n + o;
                dM_dd_TE_o[ind2] = dM_dd_TE[o](m,n);
                dM_dd_TM_o[ind2] = dM_dd_TM[o](m,n);
            }
        }
    }
};

void solve_forward(double photon_energy, 
                   double theta, 
                   int len_d, 
                   int len_idx,
                   double* d_i, 
                   std::complex<double>* n_i, 
                   int* idx_i, 
                   complex64* M_TE_o,
                   complex64* M_TM_o)
    {

    std::complex<double> j(0.0,1.0);
    double h = 6.626070040e-34;
    double c = 299792458.0;
    double wavelength = h * c / photon_energy;

    ArrayXcd n(len_d);
    ArrayXd d(len_d);
    ArrayXi idx(len_idx);

    for(int i = 0; i < len_d; ++i) {
        n(i) = n_i[i];
        d(i) = d_i[i];
        if(i<len_idx) {
            idx(i) = idx_i[i];
        };
    };

    ArrayXcd angles(len_d);
    angles(0) = theta;

    for(int i = 1; i < len_d; ++i) {
        angles(i) = std::asin(n(i-1) * std::sin(angles(i-1)) / n(i));
    };

    ArrayXcd Kx = 2*M_PI*n/wavelength*angles.cos();
    ArrayXcd L = Kx(seq(1,last))/Kx(seq(0,last-1));

    Matrix2cd M_TE;
    Matrix2cd M_TM;
    M_TE << 1.0, 0.0,
            0.0, 1.0;
    M_TM << 1.0, 0.0,
            0.0, 1.0;

    Matrix2cd T_TE;
    Matrix2cd T_TM;
    Matrix2cd P;

    for(int i = 0; i < len_d-1; ++i) {
        T_TE(0,0) = 1.0+L(i); 
        T_TE(0,1) = 1.0-L(i); 
        T_TE(1,0) = 1.0-L(i); 
        T_TE(1,1) = 1.0+L(i); 
        T_TE *= 0.5;

        T_TM(0,0) = L(i)*n(i)/n(i+1)+n(i+1)/n(i);
        T_TM(0,1) = L(i)*n(i)/n(i+1)-n(i+1)/n(i);
        T_TM(1,0) = L(i)*n(i)/n(i+1)-n(i+1)/n(i);
        T_TM(1,1) = L(i)*n(i)/n(i+1)+n(i+1)/n(i);
        T_TM *= 0.5;

        P(0,0) = exp(-j*Kx(i+1)*d(i+1));
        P(1,0) = 0.0;
        P(0,1) = 0.0;
        P(1,1) = exp(j*Kx(i+1)*d(i+1));

        M_TE = M_TE * T_TE * P;
        M_TM = M_TM * T_TM * P;
    }

    for(int m=0; m<2; ++m) {
        for(int n=0; n<2; ++n) {
            int ind1 = m*2 + n;
            M_TE_o[ind1] = M_TE(m,n);
            M_TM_o[ind1] = M_TM(m,n);
        }
    }
};
