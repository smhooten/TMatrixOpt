import scipy.io
import matplotlib.pyplot as plt
import numpy as np

Nps = [2,3,4,5,6,7,8,9,10]
foms = []

for Np in Nps:
    filename = 'results_Np'+str(Np)+'_2.mat'
    a = scipy.io.loadmat(filename)
    foms.append(a['fom'])
    #best_d = a['d']

foms = np.array(foms).flatten()
Nps = np.array(Nps)
f = plt.figure()
ax = f.add_subplot(111)
ax.plot(Nps, foms*100)
ax.set_xlabel('Bragg Pairs')
ax.set_ylabel('Averaged Reflectivity (%)')
plt.show()
