from random import choice,randint
import matplotlib.pyplot as plt
import numpy as np

x_values = np.load('src/trajectory_result/xs.npy', allow_pickle=True)
y_values = np.load('src/trajectory_result/ys.npy', allow_pickle=True)
all_obs_xvalues = np.load('src/trajectory_result/obs_xs.npy', allow_pickle=True)
all_obs_yvalues = np.load('src/trajectory_result/obs_ys.npy', allow_pickle=True)

safe_obs_xvalues = np.load('src/trajectory_result/safe_obs_xs.npy', allow_pickle=True)
safe_obs_yvalues = np.load('src/trajectory_result/safe_obs_ys.npy', allow_pickle=True)
for i in range(len(all_obs_xvalues)):
	if (all_obs_xvalues[i] > 1.0):
		all_obs_xvalues[i] = -1 + (all_obs_xvalues[i] - 1)
	elif (all_obs_xvalues[i] < -1.0):
		all_obs_xvalues[i] = 1 + (all_obs_xvalues[i] + 1)

plt.figure(0)
#绘制运动的轨迹图，且颜色由浅入深
point_numbers = np.array(range(len(x_values)))
obs_point_numbers = np.array(range(len(all_obs_xvalues)))
plt.scatter(x_values, y_values, c=point_numbers, cmap=plt.cm.Greens, edgecolors='none', s=15)
plt.scatter(all_obs_xvalues, all_obs_yvalues, c=obs_point_numbers, cmap=plt.cm.Reds, s=40)
plt.scatter(safe_obs_xvalues[-50:], safe_obs_yvalues[-50:], c='red', s=1)
#将起点和终点高亮显示，s=100代表绘制的点的大小
plt.scatter(x_values[0], y_values[0], c='green', s=100)
plt.scatter(x_values[-1], y_values[-1], c='Black', s=100, marker = 's')

plt.axhline(y=1, xmin=-1, xmax=1, color='g',linewidth=4., linestyle='-')
plt.axvline(x=0.995, ymin=-1, ymax=1, color='g',linewidth=4., linestyle='-')
plt.axvline(x=-0.995, ymin=-1, ymax=1, color='g',linewidth=4., linestyle='-')

plt.show()
#plt.savefig("/Users/zheng/Desktop/Research/Week1/success.png", dpi=600, format='png')

#plot for adamBA valid action choices
# plt.figure(1)
#plot the out, yes, valid from AdamBA
#out_s = np.load('src/out_s.npy', allow_pickle=True)
# yes_s = np.load('src/trajectory_result/yes_s.npy', allow_pickle=True)
# valid_s = np.load('src/trajectory_result/valid_s.npy', allow_pickle=True)
# x = np.linspace(-1,1,len(valid_s))
# #plt.scatter(x,out_s)
# plt.plot(yes_s)
# plt.plot(valid_s)
