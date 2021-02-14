# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:40:19 2021

@author: Wojtek
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm

from matplotlib import cm
import PFlib as pf
import numpy as np

map_size = 1000

mapa = pf.FastMap(map_size)

mapa.add_line(-1,1,800)

mapa.add_circle(1000,1000,300)
mapa.add_circle(200,0,300)
mapa.add_circle(200,700,100)
mapa.add_circle(800,400,60)

fir = pf.ParticleFilter()

pos = np.array([map_size/2,map_size/2])
ori = 0
vel = 10
model = pf.Model(*pos,ori,vel)

# dori = np.pi/70
dori = -np.pi/30

pop_size = 10000
model.set_map(mapa)
fir.set_model(model)
fir.setup(pop_size)

errs = []
errs2 = []

pop = None

ori1 = []
ori2 = []
oris = []


def print_err():
    global pest
    estpos = np.array(fir.get_est())

    pop = fir.get_pop()
    ws = fir.get_weights()

    estpos[:2] = np.average(pop,axis=0,weights=ws)[:2]

    pest = estpos
    posori = np.array([*pos,ori%(2*np.pi)])
    diffori = (np.cos(estpos[2])-np.cos(posori[2]))**2+(np.sin(estpos[2])-np.sin(posori[2]))**2

    diff = estpos-posori
    diff[2] = map_size*diffori/2/np.pi

    errs.append(np.array([diff[0],diff[1],diff[2]]))

poss = []
ests = []

plt.ion()
fig = plt.figure(figsize=(8,8))
plt.axis('equal')
plt.xlim([0,map_size])
plt.ylim([0,map_size])

grid = np.array(mapa.get_grid()).T
plt.imshow(grid,cmap='gray')
points, = plt.plot([],[],'.b',alpha=0.01)

posline, = plt.plot([],[],'.r',ms=15)
estline, = plt.plot([],[],'.y',ms=15)
plt.show()

oris = []
step=np.pi/5/2
# alpha_mask = grid.astype(np.float32)
# for i in range(100):
# for i in range(300):
# def animate(i):
    # global dori,ori

# @profile
def iter(i):
    global ori
    if not plt.fignum_exists(fig.number):
        # break;
        return False

    dori = np.random.uniform(-step,step)
    # dori = -np.pi/30

    # model.set(*pos,ori,vel)
    m = model.get_meas()
    print(i,m,pos,ori,flush=True)

    fir.update_weights(m)

    # print('UNO',flush=True)
    print_err() #===========================
    # print('EFF',flush=True)

    Neff = fir.get_effective_N()
    # print('ERR',flush=True)
    # if Neff < pop_size*0.8:
    #     print('resample',Neff,flush=True)
    #     fir.resample(pf.RESAMPLE_TYPE.SUS)
    fir.resample(pf.RESAMPLE_TYPE.SUS)
    # fir.resample(pf.RESAMPLE_TYPE.ROULETTE_WHEEL)

    # print('COPYING',flush=True)
    prev = pos.copy()
    # print('DOS',flush=True)

    fir.drift(dori,.01,.03)
    model.update(dori,0)
    pos[0], pos[1], ori, tmp = model.get()
    # # print('DRIFTED',flush=True)
    # ori += dori
    # # if grid.T[np.int64(pos[0]+np.cos(ori)*vel)][np.int64(pos[1]+np.sin(ori)*vel)] == False:
    # if m<vel*2:
    #     print('Bump')
    #     # break
    #     ori+=np.pi
    # pos[0] = pos[0]+np.cos(ori)*vel
    # pos[1] = pos[1]+np.sin(ori)*vel


    # pf.diffuse(1,.01)
    # pf.diffuse(.08,.03) # 100
    # print('TRES',flush=True)
    # fir.diffuse(.9,.03)
    # print('QUATRO',flush=True)

    poss.append(prev)
    ests.append(pest[:2])


    pop = fir.get_pop()
    points.set_data(pop[:,0],pop[:,1])
    posline.set_data(*prev)
    estline.set_data(*pest[:2])

    # return points,posline,estline

    fig.canvas.draw()
    fig.canvas.flush_events()

    # # if True:
    # # # if i%10==9:
    # pop = fir.get_pop()
    # plt.axis('equal')
    # # plt.title(str(i)+' '+str(errs[-1]))
    # plt.imshow(grid,cmap='gray')
    # plt.scatter(pop[:,0],pop[:,1],s=1,alpha=1000/pop_size)
    # plt.imshow(grid,cmap='gray')
    # plt.plot(*prev,'.r')
    # plt.plot(*pest[:2],'.k')

    # plt.xlim([0,map_size])
    # plt.ylim([0,map_size])
    # plt.show()
    return True

def init():
    points.set_data([],[])
    posline.set_data([],[])
    estline.set_data([],[])
    return points,posline,estline

for i in range(1000):
    if not iter(i):
        break

# anim = FuncAnimation(fig, animate, init_func=init,
#                       frames=1000, interval=40)
# # anim.save('pf_test.gif')
# writergif = PillowWriter(fps=25)
# anim.save("pf_test.gif",writer=writergif)




# plt.figure()
# poss = np.array(poss)
# ests = np.array(ests)

# # INIT = 100
# INIT = 0
# plt.plot(errs[INIT:])
# plt.legend(['x','y','ori'])
# plt.xlabel('nr. iteracji')
# plt.ylabel('err')
# plt.show()

# plt.imshow(grid,cmap='gray')
# plt.axis('equal')
# plt.plot(poss[INIT:,0],poss[INIT:,1],label='faktyczne położenie')
# plt.plot(ests[INIT:,0],ests[INIT:,1],label='wyznaczone położenie')
# plt.plot(*prev,'.r',label='końcowa pozycja')
# plt.legend()
# plt.show()