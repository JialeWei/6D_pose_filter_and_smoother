import numpy as np
import matplotlib.pyplot as plt

df = np.loadtxt('../result/out.txt', delimiter=',')

t = df[:, 0]
gt = df[:, 1]
filtered = df[:, 2]
smoothed = df[:, 3]

plt.plot(t, gt, label='gt')
plt.plot(t, filtered, label='filtered')
plt.plot(t, smoothed, label='smoothed')
plt.legend()

plt.show()
