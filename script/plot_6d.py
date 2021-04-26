import numpy as np
import matplotlib.pyplot as plt

df = np.loadtxt('../result/out.txt', delimiter=',', dtype=np.float64)

t = df[:, 0]
gt_x = df[:, 1]
gt_y = df[:, 2]
gt_z = df[:, 3]
gt_R = df[:, 4]
gt_P = df[:, 5]
gt_Y = df[:, 6]
filtered_x = df[:, 7]
filtered_y = df[:, 8]
filtered_z = df[:, 9]
filtered_R= df[:, 10]
filtered_P = df[:, 11]
filtered_Y = df[:, 12]
smoothed_x = df[:, 13]
smoothed_y = df[:, 14]
smoothed_z = df[:, 15]
smoothed_R = df[:, 16]
smoothed_P = df[:, 17]
smoothed_Y = df[:, 18]

plt.figure(1)
plt.subplot(311)
plt.plot(t, gt_x, label='gt')
plt.plot(t, filtered_x, label='filtered_x')
plt.plot(t, smoothed_x, label='smoothed_x')
plt.legend() 

plt.subplot(312)
plt.plot(t, gt_y, label='gt')
plt.plot(t, filtered_y, label='filtered_y')
plt.plot(t, smoothed_y, label='smoothed_y')
plt.legend()

plt.subplot(313)
plt.plot(t, gt_z, label='gt')
plt.plot(t, filtered_z, label='filtered_z')
plt.plot(t, smoothed_z, label='smoothed_z')
plt.legend()

plt.figure(2)
plt.subplot(311)
plt.plot(t, gt_R, label='gt')
plt.plot(t, filtered_R, label='filtered_R')
plt.plot(t, smoothed_R, label='smoothed_R')
plt.legend()

plt.subplot(312)
plt.plot(t, gt_P, label='gt')
plt.plot(t, filtered_P, label='filtered_P')
plt.plot(t, smoothed_P, label='smoothed_P')
plt.legend()

plt.subplot(313)
plt.plot(t, gt_Y, label='gt')
plt.plot(t, filtered_Y, label='filtered_Y')
plt.plot(t, smoothed_Y, label='smoothed_Y')
plt.legend()

plt.show()
