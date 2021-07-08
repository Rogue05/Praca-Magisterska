# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:46:06 2021

@author: Wojtek
"""

import numpy as np
import pandas as pd
import zipfile

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

print('START',flush=True)
filename = '66574_759325_M-33-35-C-a-3-1.xyz'
zipfilename = filename + '.zip'
# zf = zipfile.ZipFile(filename)
# data = pd.read_csv(zf.open(zf.namelist()[0]),header=None,delimiter='   ')
data = np.array(pd.read_csv(filename,header=None,delimiter='\s+')).T


# lim = (data[0].max() - data[0].min())*0.01 + data[0].min()
# inds = data[0]<lim

# x,y,z = data[0][inds],data[1][inds],data[2][inds]


x,y,z = data[0].astype(np.int64),data[1].astype(np.int64),data[2]
x -= x.min()
y -= y.min()

# print('plotting...',flush=True)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(x,y,z, color='white', edgecolors='grey', alpha=0.5)
# # ax.scatter(x,y,z)
# plt.show()

# x,y,z = data[0].astype(np.int64),data[1].astype(np.int64),data[2]
# x -= x.min()
# y -= y.min()

# print(x.shape, y.shape, z.shape)

grid = np.zeros((x.max()+1,y.max()+1))#+np.average(z)
grid[x,y] = z - z.min()
# nx,ny = np.meshgrid(range(grid.shape[0]),range(grid.shape[1]))
# print(nx.shape, ny.shape, grid.shape)
# print('plotting...',flush=True)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# lim = -1
# beg = 1800
# ax.plot_surface(nx[beg:lim,beg:lim], 
# 				ny[beg:lim,beg:lim],
# 				grid.T[beg:lim,beg:lim],
# 				cmap=cm.coolwarm, rstride=1, cstride=1,
# 				linewidth=0)
# plt.show()


# with open('map.npy','wb') as file:
# 	np.save(file, grid)
# # grid = np.exp(grid)
# # N=1000
# # plt.imshow(grid[:N,:N])
tmp = grid.T[1000:,1000:]
avg = np.average(tmp)
tmp[tmp<avg] = avg
plt.imshow(tmp-avg)
# plt.xlim([0,x.max()+1])
# plt.ylim([0,y.max()+1])
plt.show()
