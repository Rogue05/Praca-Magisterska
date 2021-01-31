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

map_size = 100

mapa = pf.Map()
mapa.setup(map_size,map_size,map_size//50)
mapa.add_circle((map_size/2,3*map_size/4),map_size/10)
mapa.add_circle((0,0),map_size/4)

mapa.add_circle((map_size*3/5,0),map_size/8)
mapa.add_circle((map_size/10,map_size*3/5),map_size/11)
pos = (2*map_size//3,5*map_size//8)
size = (map_size//2,map_size//20)
mapa.add_box(pos,size)

pos = (3*map_size//4,map_size//4)
size = (map_size//10,map_size//10)
mapa.add_box(pos,size)
# pf.tmp_set_map(mapa)

# ##============================SCAN
# pos = (map_size/2,)*2
# pos = (50,50)
# plt.axis('equal')
# plt.plot(*pos,'r.')

# for ori in np.linspace(0,2*np.pi,1000):
#     m = pf._get_meas(*pos,ori)
#     plt.plot(m*np.cos(ori)+pos[0],m*np.sin(ori)+pos[1],'.r',ms=1)
# # plt.show()
# plt.imshow(np.asarray(mapa.get()).T,cmap='gray')
# plt.xlim([0,map_size])
# plt.ylim([0,map_size])

##============================SYMULACJA

fir = pf.ParticleFilter()

grid = np.logical_not(mapa.get().T)

pos = np.array([map_size/2,map_size/2])
ori = 0
# vel = 10
# vel = 10/2
vel = 1

# dori = np.pi/70
dori = -np.pi/30

pop_size = 10000
fir.set_map(mapa)
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

plt.axis('equal')
plt.xlim([0,map_size])
plt.ylim([0,map_size])

plt.imshow(grid,cmap='gray')
points, = plt.plot([],[],'.b',alpha=0.01)

posline, = plt.plot([],[],'.r',ms=15)
estline, = plt.plot([],[],'.y',ms=15)

oris = []
step=np.pi/5/2
alpha_mask = grid.astype(np.float32)
for i in range(1000):
# for i in range(300):
# def animate(i):
    # global dori,ori
    print(i,pos,ori,flush=True)

    dori = np.random.uniform(-step,step)
    # dori = -np.pi/30

    fir.set_model(*pos,ori,vel)
    m = fir._get_meas(*pos,ori)

    fir.update_weights(m)

    # print('UNO',flush=True)
    print_err()
    # print('EFF',flush=True)

    Neff = fir.get_effective_N()
    # print('ERR',flush=True)
    # if Neff < pop_size*0.5:
    #     print('resample',Neff,flush=True)
    #     pf.resample(pf.RESAMPLE_TYPE.SUS)
    fir.resample(pf.RESAMPLE_TYPE.SUS)

    # print('COPYING',flush=True)
    prev = pos.copy()
    # print('DOS',flush=True)

    fir.drift(dori)
    # print('DRIFTED',flush=True)
    ori += dori
    if grid.T[np.int64(pos[0]+np.cos(ori)*vel)][np.int64(pos[1]+np.sin(ori)*vel)] == False:
        print('Bump')
        # break
        ori+=np.pi
    pos[0] = pos[0]+np.cos(ori)*vel
    pos[1] = pos[1]+np.sin(ori)*vel


    # pf.diffuse(1,.01)
    # pf.diffuse(.08,.03) # 100
    # print('TRES',flush=True)
    fir.diffuse(.9,.03)
    # print('QUATRO',flush=True)

    poss.append(prev)
    ests.append(pest[:2])


    # pop = pf.get_pop()
    # points.set_data(pop[:,0],pop[:,1])
    # posline.set_data(*prev)
    # estline.set_data(*pest[:2])

    # return points,posline,estline



    # if True:
    # # if i%10==9:
    pop = fir.get_pop()
    plt.axis('equal')
    # plt.title(str(i)+' '+str(errs[-1]))
    plt.imshow(grid,cmap='gray')
    plt.scatter(pop[:,0],pop[:,1],s=1,alpha=1000/pop_size)
    plt.imshow(grid,cmap='gray')
    plt.plot(*prev,'.r')
    plt.plot(*pest[:2],'.k')

    plt.xlim([0,map_size])
    plt.ylim([0,map_size])
    plt.show()

def init():
    points.set_data([],[])
    posline.set_data([],[])
    estline.set_data([],[])
    return points,posline,estline

# anim = FuncAnimation(fig, animate, init_func=init,
#                       frames=1000, interval=40)
# # anim.save('pf_test.gif')
# writergif = PillowWriter(fps=25)
# anim.save("pf_test.gif",writer=writergif)

plt.figure()
poss = np.array(poss)
ests = np.array(ests)

# INIT = 100
INIT = 0
plt.plot(errs[INIT:])
plt.legend(['x','y','ori'])
plt.xlabel('nr. iteracji')
plt.ylabel('err')

plt.show()
plt.imshow(grid,cmap='gray')
plt.axis('equal')
plt.plot(poss[INIT:,0],poss[INIT:,1],label='faktyczne położenie')
plt.plot(ests[INIT:,0],ests[INIT:,1],label='wyznaczone położenie')
plt.plot(*prev,'.r',label='końcowa pozycja')
plt.legend()
plt.show()