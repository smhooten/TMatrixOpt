# TMatrixOpt
TMatrixOpt (transfer matrix optimizer) is an electromagnetic solver capable of simulating and optimizing 1D (thin-layer) structures via the semi-analytical transfer matrix method. For example, one can simulate and optimize broadband distributed Bragg reflectors, anti-reflection coatings, optical bandbass filters, and photovoltaic devices -- any 1D device that assumes an incident wave with arbitrary frequency and angle of incidence.

The utility of TMatrixOpt is best illustrated with an example. Consider the following distributed Bragg reflector with 10 (total) alternating silicon (n=3.5) and silica (n=1.4) layers, floating in vacuum:

   |                     ^

   |                     |

   | Incident Light      | Reflected Light

   |                     |

   v                     |

Vacuum (0nm for boundary condition, fixed)

\---------------------------------------

Si (variable thickness)

\---------------------------------------

SiO2 (variable thickness)

\---------------------------------------

Si (variable thickness)

\---------------------------------------

SiO2 (variable thickness)

\---------------------------------------

Si (variable thickness)

\---------------------------------------

SiO2 (variable thickness)

\---------------------------------------

Si (variable thickness)

\---------------------------------------

SiO2 (variable thickness)

\---------------------------------------

Si (variable thickness)

\---------------------------------------


SiO2 (variable thickness)

\---------------------------------------


Vacuum (0nm for boundary condition, fixed)

Using TMatrixOpt, we can easily optimize the thicknesses of each of these layers to maximize the average reflectivity over a desired bandwidth. In this case, we choose both a large frequency bandwidth and incident angle bandwidth of [0.1eV, 0.75eV] and [0.0 degrees, 90 degrees] respectively. The initial thicknesses are chosen to reflect normally incident light with energy of 0.45eV. The initial reflectivity spectra (both TE and TM incident light) are given here:

![initial](https://github.com/smhooten/TMatrixOpt/blob/master/readme_images/10-Layer-Mirror_Initial_Reflectivity.png?raw=true)

which has an average reflectivity of approximately 70%. After optimization with TMatrixOpt, the reflectivity profile improves dramatically:

![final](https://github.com/smhooten/TMatrixOpt/blob/master/readme_images/10-Layer-Mirror_Final_Reflectivity.png?raw=true)

The optimized structure greatly extends the reflectivity bandwidth, with overall average reflectivity increasing to approximately 90%.

This was a relatively simple example, showing that TMatrixOpt is capable of greatly improving a user-defined merit function. More complex merit functions are possible, enabling optimization of any of the devices mentioned above.

## Why Should I Use TMatrixOpt?
TMatrixOpt is __modular__ and __fast__. 

The modularity of TMatrixOpt allows one to simulate any desired 1D structure that is a function of the Fresnel coefficients at any input frequency and angle of incidence. In principle, defining a structure is as simple as instantiating a few classes from the TMatrixOpt.geometry module, and the user is free to choose exactly which layers can and cannot be optimized. For example, a photovoltaic device with a fixed active absorbing layer sandwiched between passive layers is possible. Moreover, the materials defining the structure may have loss, gain, or be dispersive. Lastly, the user can easily choose between various conventional scipy optimization algorithms, or define a custom optimization routine.

The speed of TMatrixOpt is enabled by MPI (message passing interface) which allows ease of invoking parallelized operations. Consider the example given in examples/lumerical\_comparison. As the name implies, the speed of TMatrixOpt is compared to the speed of the built-in transfer-matrix solver of Lumerical, a commercial photonic software. Using MPI, we can quickly simulate (for example) a 1D lossy, linearly-chirped distributed Bragg reflector with very fine incident angle and photon energy resolution. The angular and photon energy resolution are [0 degrees, 90 degrees, 0.125 degree spacing] and [0.1 eV, 0.74 eV, 0.0001 eV spacing] respectively. This constitutes over 4.2 million transfer matrix evaluations in total. The time required to solve this system versus number of MPI nodes is shown below (compared to the speed of Lumerical):

![mpi](https://github.com/smhooten/TMatrixOpt/blob/master/readme_images/10-Layer-Mirror_Final_Reflectivity.png?raw=true)

Notably, Lumerical is faster for single-CPU operation, but TMatrixOpt outperforms Lumerical with MPI (with solve times of 76s and ~18s for Lumerical and TMatrixOpt with 28 MPI nodes respectively). The saturation of simulation time is a result of the increased computation and memory overhead for using MPI. High-performance servers (such as HPC) would likely offer even further improvement for large problems.

TMatrixOpt also offers the functionality of providing parameter gradients to the user, with the ease of data manipulation provided by Python programming (as opposed to the bulky Lumerical GUI). Fast gradient computation is enabled by the novel method described in Ref. [1] below, whereby gradients can be computed much faster than a simple finite-difference scheme.

## Installation
Installation will require common Python dependencies (numpy, scipy, matplotlib) and OpenMP (for MPI). See INSTALL\_INSTRUCTIONS for a step-by-step installation. Further instructions to install MPI for systems other than CentOS will be provided later on.

## Examples
Please see the /examples directory. Documentation coming soon!

## Authors
Sean Hooten

Zunaid Omair

e-mails: {shooten, zomair} (at) eecs.berkeley.edu

Acknowledgement: Andrew Michaels

## Citation
If you use this software in your research, please cite:

[1] Z. Omair, S. Hooten & E. Yablonovitch. Optimized Optics for Highly Efficient Photovoltaic Devices. in 2020 47th IEEE Photovoltaic Specialists Conference (PVSC) 18131815 (2020). doi:10.1109/PVSC45281.2020.9300579.

## License
TMatrixOpt: A modular transfer-matrix optimization package for 1D
optical devices.
Copyright (C) 2021 Sean Hooten & Zunaid Omair

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

==========================================================================

This software redistributes and modifies code (under the GNU GPL 3.0 License)
from EMopt, copyright Andrew Michaels. It may be found at:
    https://github.com/anstmichaels/emopt.git
