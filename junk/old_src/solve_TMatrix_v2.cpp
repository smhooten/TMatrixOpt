#include <iostream> 
#include <Eigen/Core> 
#include <Eigen/Dense> 
#include <math.h>
#include <complex>

#ifndef LEN_IDX
#define LEN_IDX
#endif

using namespace Eigen;


void solve(double const& photon_energy, 
          double const& theta, 
          ArrayXd const& d, 
          ArrayXd const& n, 
          ArrayXi const& idx, 
          int const& len_d, 
          int const& len_idx,
          Matrix2cd *M_TE_f,
          Matrix2cd *M_TM_f,
          Matrix2cd dM_dd_TE_f[],
          Matrix2cd dM_dd_TM_f[]) 
    {

    std::complex<double> j(0.0,1.0);
    double h = 6.626070040e-34;
    double wavelength = h * 299792458.0 / photon_energy;
    //std::cout << wavelength << std::endl;
    //std::cout << photon_energy << std::endl;
    ArrayXd angles(len_d);
    angles(0) = theta;

    for(int i = 1; i < len_d; ++i) {
        angles(i) = asin(n(i-1) * sin(angles(i-1)) / n(i));
    };

    ArrayXd Kx = 2*M_PI*n/wavelength*angles.cos();
    ArrayXd L = Kx(seq(1,last))/Kx(seq(0,last-1));

    Matrix2cd M_TE;
    Matrix2cd M_TM;
    M_TE << 1.0, 0.0,
            0.0, 1.0;
    M_TM << 1.0, 0.0,
            0.0, 1.0;

    Matrix2cd X_TE[len_idx];
    Matrix2cd X_TM[len_idx];

    int counter = 0;
    Matrix2cd T_TE;
    Matrix2cd T_TM;
    Matrix2cd P;
    Matrix2cd B_TE;
    Matrix2cd B_TM;
    for(int i = 0; i < len_d-1; ++i) {
        T_TE(0,0) = 1+L(i); 
        T_TE(0,1) = 1-L(i); 
        T_TE(1,0) = 1-L(i); 
        T_TE(1,1) = 1+L(i); 
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

        Matrix2cd B_TE = T_TE*P;
        Matrix2cd B_TM = T_TM*P;

        for(int k = 0; k<len_idx; k++) {
            if(i+1==idx(k)) {
                X_TE[counter] = M_TE * T_TE;
                X_TM[counter] = M_TM * T_TM;
                counter++;
            }
        }
        M_TE = M_TE * B_TE;
        M_TM = M_TM * B_TM;
    }

    Matrix2cd dM_dd_TE[len_idx];
    Matrix2cd dM_dd_TM[len_idx];

    Matrix2cd Idash;
    Idash(0,0) = -1.0;
    Idash(0,1) = 0.0;
    Idash(1,0) = 0.0;
    Idash(1,1) = 1.0;

    counter = 0;
    for(int i = 0; i < len_idx; ++i) {
        int k = idx(i);

        //Matrix2cd A_TE = X_TE[counter].colPivHouseholderQr().solve(M_TE);
        //Matrix2cd A_TM = X_TM[counter].colPivHouseholderQr().solve(M_TM);
        Matrix2cd A_TE = X_TE[counter].householderQr().solve(M_TE);
        Matrix2cd A_TM = X_TM[counter].householderQr().solve(M_TM);

        dM_dd_TE[counter] = j * Kx[k] * X_TE[counter] * Idash * A_TE;
        dM_dd_TM[counter] = j * Kx[k] * X_TM[counter] * Idash * A_TM;

        ++counter;
    }

    *M_TE_f = M_TE;
    *M_TM_f = M_TM;

    for(int i=0; i<len_idx; ++i) {
        dM_dd_TE_f[i] = dM_dd_TE[i];
        dM_dd_TM_f[i] = dM_dd_TM[i];
    }
};

int main() {
    double q = 1.60217662e-19;
    //double q = 1.602e-19;
    double photon_energy = 0.5*q;
    double theta = 0.2;

    ArrayXd d(5);
    d(0) = 0.0;
    d(1) = 1e-6;
    d(2) = 2e-6;
    d(3) = 3e-6;
    d(4) = 0.0;

    ArrayXd n(5);
    n(0) = 1.0;
    n(1) = 1.5;
    n(2) = 3.5;
    n(3) = 1.5;
    n(4) = 1.0;

    ArrayXi idx(3);
    idx(0) = 1; 
    idx(1) = 2;
    idx(2) = 3;

    const int len_d = 5;
    const int len_idx = 3;

    Matrix2cd M_TE;
    Matrix2cd M_TM;

    Matrix2cd dM_dd_TE[len_idx];
    Matrix2cd dM_dd_TM[len_idx];

    solve(photon_energy, theta, d, n, idx, len_d, len_idx, &M_TE, &M_TM, dM_dd_TE, dM_dd_TM);

    std::cout << M_TE << std::endl;
    std::cout << M_TM << std::endl;
    std::cout << dM_dd_TE[0] << std::endl;
    std::cout << dM_dd_TE[1] << std::endl;
    std::cout << dM_dd_TE[2] << std::endl;
    std::cout << dM_dd_TM[0] << std::endl;
    std::cout << dM_dd_TM[1] << std::endl;
    std::cout << dM_dd_TM[2] << std::endl;


    return 0;
}

