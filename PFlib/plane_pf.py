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


fir = pf.ParticleFilter()

pos = np.array(grid.shape)/3
ori = np.pi/4
vel = 1
model = pf.Model(*pos,ori,vel,1.5*vel+0.01, .01,.03)
model.set_map(mapa)

dori = -np.pi/30

pop_size = 10000
fir.set_model(model)
fir.setup(pop_size)

errs = []
erri = []

pop = None

def print_err():
    global pest,ws
    estpos = np.array(fir.get_est())
    # print('\t\t\t\t\t\t\t\t========',fir.get_est_meas())
    # errs.append(fir.get_est_meas())

    pop = fir.get_pop()
    ws = fir.get_weights()
    # print(np.var(ws),ws)

    # plt.figure(2)
    # plt.cla()
    # plt.hist(ws,bins=100)

    estpos[:2] = np.average(pop,axis=0,weights=ws)[:2]

    pest = estpos
    posori = np.array([*pos,ori%(2*np.pi),vel])
    diffori = (np.cos(estpos[2])-np.cos(posori[2]))**2+(np.sin(estpos[2])-np.sin(posori[2]))**2

    diff = estpos-posori
    diff[2] = grid.shape[0]*diffori/2/np.pi

    # print(estpos)
    # errs.append(np.array([diff[0],diff[1],diff[2]]))

poss = []
ests = []

plt.ion()
fig = plt.figure(figsize=(16,8))
plt.subplot(121)
plt.axis('equal')
plt.xlim([0,grid.shape[0]])
plt.ylim([0,grid.shape[1]])

# grid = np.array(mapa.get_grid()).T
# print(grid.shape)
plt.imshow(grid)
# plt.imshow(grid,cmap='gray')

# points, = plt.plot([],[],'.w',alpha=0.01)
points, = plt.plot([],[],'.w',ms=1)
posline, = plt.plot([],[],'.r',ms=15)
estline, = plt.plot([],[],'.y',ms=15)
plt.show()

# import time
# time.sleep(5)

oris = []
step=np.pi/5/2
plt.subplot(122)
errp, = plt.plot([],[])
plt.xlim([0,1000])
plt.ylim([0,np.abs(grid.max()-grid.min())])

# @profile
def iter(i):
    global ori
    if not plt.fignum_exists(fig.number):
        # break;
        return False

    dori = 0

    m = model.get_meas()
    m.randomize(.0)
    print(i,m.get_real(),pos,ori,flush=True)

    fir.update_weights(m)

    # print('updated weights',flush=True)

    print_err()
    erri.append(i)
    
    Neff = fir.get_effective_N()
    if Neff < pop_size*0.8:
        print('resample',Neff,flush=True)
        fir.resample(pf.RESAMPLE_TYPE.SUS)
    # fir.resample(pf.RESAMPLE_TYPE.SUS)
    # fir.resample(pf.RESAMPLE_TYPE.ROULETTE_WHEEL)

    prev = pos.copy()
    
    fir.drift(dori, False)
    model.update(dori,0, False)
    pos[0], pos[1], ori, tmp = model.get_real()

    poss.append(prev)
    ests.append(pest[:2])


    pop = fir.get_pop()
    points.set_data(pop[:,0],pop[:,1])
    posline.set_data(*prev)
    estline.set_data(*pest[:2])
    errp.set_data(erri, errs)

    if i%10:
    #     return points,posline,estline
        return True
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

