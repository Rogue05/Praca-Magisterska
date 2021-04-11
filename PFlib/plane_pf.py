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

# from scipy.ndimage.filters import gaussian_filter

from noise import pnoise
grid = pnoise(2**10,2**10,2**7)

print('minmax =',grid.max(),grid.min())

mapa = pf.HeightMap(grid)
# # print(grid.shape)
# plt.subplot(121)
# plt.imshow(grid)
# plt.plot([368,],[368,],'.r')
# plt.subplot(122)

# meas = grid[368,368]
# print('real = ', meas)

# for x in range(1024):
#     for y in range(1024):
#         grid[x,y] = mapa.get_meas_prob(meas,x,y,0.0)

# plt.imshow(grid)
# plt.plot([368,],[368,],'.r')
# plt.show()



fir = pf.PlaneParticleFilter()

pos = np.array(grid.shape)/3
ori = np.pi/4
vel = 2
model = pf.PlaneModel(*pos,ori,vel,1.5*vel+1, .01,0.03)
model.set_map(mapa)

dori = -np.pi/30

pop_size = 10000
fir.set_model(model)
fir.setup(pop_size)

poss = []
ests = []

erri = []
errs = []

plt.ion()
fig = plt.figure(figsize=(16,8))
plt.subplot(121)
plt.axis('equal')
plt.xlim([0,grid.shape[0]])
plt.ylim([0,grid.shape[1]])
plt.imshow(grid)
posline, = plt.plot([],[],'.r',ms=15)
estline, = plt.plot([],[],'.y',ms=15)
points, = plt.plot([],[],'.w',ms=1)
plt.show()

oris = []
step=np.pi/5/2
# plt.subplot(122)
# errp, = plt.plot([],[])
# plt.xlim([0,1000])
# plt.ylim([0,np.abs(grid.max()-grid.min())])

# @profile
def iter(i):
    global ori
    if not plt.fignum_exists(fig.number):
        # break;
        return False

    dori = 0.0

    model.update_meas(0.0, 0.0)
    print(i,pos,ori,flush=True)
    print(model.get_real())

    fir.update_weights()

    pest = np.array(fir.get_est())
    erri.append(i)
    errs.append(i)
    
    Neff = fir.get_effective_N()
    if Neff < pop_size*0.8:
        print('resample',Neff,flush=True)
        fir.resample(pf.RESAMPLE_TYPE.SUS)
    # fir.resample(pf.RESAMPLE_TYPE.SUS)
    # fir.resample(pf.RESAMPLE_TYPE.ROULETTE_WHEEL)

    prev = pos.copy()
    # print('PRE',dori,model.get_real())
    model.update(dori,0.0)
    # print('POS',dori,model.get_real())
    fir.drift()
    pos[0], pos[1], ori, tmp = model.get_real()

    poss.append(prev)
    ests.append(pest[:2])


    pop = fir.get_pop()
    points.set_data(pop[:,0],pop[:,1])
    posline.set_data(*prev)
    estline.set_data(*pest[:2])
    # errp.set_data(erri, errs)

    plt.subplot(122)
    plt.cla()
    # plt.hist(pop[:,3],bins=100)
    plt.hist(pop[:,2],bins=100)

    # if i%10:
    # #     return points,posline,estline
    #     return True
    fig.canvas.draw()
    fig.canvas.flush_events()
    # return points,posline,estline
    return True

def init():
    points.set_data([],[])
    posline.set_data([],[])
    estline.set_data([],[])
    return points,posline,estline

for i in range(1000):
    if not iter(i):
        break

# anim = FuncAnimation(fig, iter, init_func=init,
#                       # frames=200, interval=40)
#                       frames=1000, interval=40)
# # anim.save('pf_test.gif')
# writergif = PillowWriter(fps=25)
# anim.save("pf_test.gif",writer=writergif)

