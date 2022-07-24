import numpy as np
import matplotlib.pyplot as plt

a = np.load('debris_site_1.npy')
plt.subplot(3,1,1)
plt.plot(a[:,0])

plt.subplot(3,1,2)
plt.plot(a[:,1])

plt.subplot(3,1,3)
plt.plot(a[:,2])
plt.show()