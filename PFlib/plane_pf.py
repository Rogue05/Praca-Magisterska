# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:40:19 2021

@author: Wojtek
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm

from matplotlib import cm
import pandas as pd
import PFlib as pf
import numpy as np

print('START',flush=True)
filename = '66574_759325_M-33-35-C-a-3-1.xyz'
zipfilename = filename + '.zip'
data = np.array(pd.read_csv(filename,header=None,delimiter='\s+')).T
x,y,z = data[0].astype(np.int64),data[1].astype(np.int64),data[2]
x -= x.min()
y -= y.min()

print('loaded data',flush=True)
print(x.shape, y.shape, z.shape)

grid = np.zeros((x.max()+1,y.max()+1))#+np.average(z)
grid[x,y] = z - z.min()

mapa = pf.HeightMap(grid)

fir = pf.ParticleFilter()

pos = np.array(grid.shape)/100
ori = 0
vel = 10
model = pf.Model(*pos,ori,vel,30, .1,.03)

# dori = -np.pi/30

# pop_size = 1000
model.set_map(mapa)
# fir.set_model(model)
# fir.setup(pop_size)

# errs = []
# errs2 = []

# pop = None

# ori1 = []
# ori2 = []
# oris = []


# def print_err():
#     global pest,ws
#     estpos = np.array(fir.get_est())
#     print('\t\t\t\t\t\t\t\t========',fir.get_est_meas())

#     pop = fir.get_pop()
#     ws = fir.get_weights()
#     # print(np.var(ws),ws)

#     # plt.figure(2)
#     # plt.cla()
#     # plt.hist(ws,bins=100)

#     estpos[:2] = np.average(pop,axis=0,weights=ws)[:2]

#     pest = estpos
#     posori = np.array([*pos,ori%(2*np.pi),vel])
#     diffori = (np.cos(estpos[2])-np.cos(posori[2]))**2+(np.sin(estpos[2])-np.sin(posori[2]))**2

#     diff = estpos-posori
#     diff[2] = map_size*diffori/2/np.pi

#     print(estpos)
#     errs.append(np.array([diff[0],diff[1],diff[2]]))

# poss = []
# ests = []

# plt.ion()
# fig = plt.figure(figsize=(8,8))
# plt.axis('equal')
# plt.xlim([0,map_size])
# plt.ylim([0,map_size])

# grid = np.array(mapa.get_grid()).T
# plt.imshow(grid,cmap='gray')
# points, = plt.plot([],[],'.b',alpha=0.01)
# # from matplotlib import cm

# # points = plt.scatter([],[])
# # # print(points.__dict__)

# posline, = plt.plot([],[],'.r',ms=15)
# estline, = plt.plot([],[],'.y',ms=15)
# plt.show()

# oris = []
# step=np.pi/5/2


# # @profile
# def iter(i):
#     global ori
#     if not plt.fignum_exists(fig.number):
#         # break;
#         return False

#     dori = np.random.uniform(-step,step)

#     m = model.get_meas()
#     m.randomize(.05)
#     print(i,m,pos,ori,flush=True)

#     fir.update_weights(m)

#     print_err()

#     # Neff = fir.get_effective_N()
#     # if Neff < pop_size*0.8:
#     #     print('resample',Neff,flush=True)
#     #     fir.resample(pf.RESAMPLE_TYPE.SUS)
#     fir.resample(pf.RESAMPLE_TYPE.SUS)
#     # fir.resample(pf.RESAMPLE_TYPE.ROULETTE_WHEEL)

#     prev = pos.copy()
    
#     fir.drift(dori)
#     model.update(dori,0)
#     pos[0], pos[1], ori, tmp = model.get()

#     poss.append(prev)
#     ests.append(pest[:2])


#     pop = fir.get_pop()
#     points.set_data(pop[:,0],pop[:,1])
#     posline.set_data(*prev)
#     estline.set_data(*pest[:2])

#     # if i%10:
#     #     return points,posline,estline
#     # #     return True
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     # return points,posline,estline
#     return True

# def init():
#     points.set_data([],[])
#     posline.set_data([],[])
#     estline.set_data([],[])
#     return points,posline,estline

# for i in range(1000):
#     if not iter(i):
#         break
#     # if i == 10:
#     #     import time
#     #     time.sleep(5)
#     #     break

# # anim = FuncAnimation(fig, iter, init_func=init,
# #                       # frames=200, interval=40)
# #                       frames=1000, interval=40)
# # # anim.save('pf_test.gif')
# # writergif = PillowWriter(fps=25)
# # anim.save("pf_test.gif",writer=writergif)

