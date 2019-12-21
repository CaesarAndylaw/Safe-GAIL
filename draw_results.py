# drawing the results
from scipy.io import loadmat
import matplotlib
import numpy as np

backend = 'TkAgg'
matplotlib.use(backend)
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

# data_dir = pjoin(dirname(sio.__file__), 'result')
# mat_fname = pjoin(data_dir, 'veh8_nossa_seg3_trial1.mat')
mat_fname = "result/nossa/veh8_nossa_seg3_trial1.mat"
mat_contents = loadmat(mat_fname)
phi_data = mat_contents['phi_data']
phi_data = np.squeeze(phi_data)

# draw the figure 
# print(phi_data)
plt.plot(phi_data)
plt.ylabel('Safety index value')
plt.xlabel('Frame number')
plt.show()
