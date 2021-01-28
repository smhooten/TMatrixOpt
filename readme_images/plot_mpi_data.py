import numpy as np
import matplotlib.pyplot as plt

number_cpus = []
times = []

lumerical_time = 76.0 # use lumerical_comparison.lsf

with open('mpi_data.txt') as f:
    data = f.read().split('\n')
    for i in range(len(data)-1):
        if i%2 == 0:
            number_cpus.append(float(data[i]))
        else:
            times.append(float(data[i]))

f = plt.figure()
ax = f.add_subplot(111)
ax.plot(number_cpus, times, '-ob')
ax.plot(number_cpus, lumerical_time*np.ones(len(number_cpus)), '--k')
ax.set_xlabel('Number of MPI Nodes')
ax.set_ylabel('Solve Time (seconds)')
ax.set_xlim([0, 41])
ax.set_ylim([0, 250])
ax.legend(['TMatrixOpt', 'Lumerical\'s stackrt'])
ax.set_title('Problem Size = 4.62 million transfer matrix calculations')
#plt.show()
plt.savefig('mpi_data.pdf')
