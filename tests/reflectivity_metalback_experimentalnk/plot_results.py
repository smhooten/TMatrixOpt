import scipy.io
import matplotlib.pyplot as plt
import numpy as np

f = plt.figure()
ax = f.add_subplot(111)

Nps = [2,3,4,5,6,7,8,9,10]
Npsa = np.array(Nps).squeeze()

suffixes = ['_variablesuperstrate_dispersiveSi_fixedSio2.mat',
            '_2.mat',
            '_variablesuperstrate.mat',
            '_variablesuperstrate_fixedindex.mat']

folders = ['./optimized_superstrate_dispersiveSi_fixedSio2/',
           './unoptimized_superstrate/',
           './optimized_superstrate/',
           './optimized_superstrate_fixedindex/']
            
foms = dict.fromkeys(folders, None)
for folder, suffix in zip(folders,suffixes):
    foms[folder] = []
    for Np in Nps:
        filename = folder+'results_Np'+str(Np)+suffix
        a = scipy.io.loadmat(filename)
        foms[folder].append(a['fom'])
    foms[folder] = np.array(foms[folder]).squeeze()
    ax.plot(Nps, 100*foms[folder])

ax.set_xlabel('Bragg Pairs')
ax.set_ylabel('Averaged Reflectivity (%)')
ax.set_ylim([95, 100])
#ax.legend(folders)
ax.legend(['variable superstrate, dispersive Si + fixed SiO2 (1.4)', 'fixed superstrate, dispersive Si + SiO2', 'variable superstrate, dispersive Si + SiO2', 'variable superstrate, fixed Si (3.05) + fixed SiO2 (1.4)',])
plt.show()
