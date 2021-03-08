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
# mapa.add_line(1,-1,700)
mapa.add_circle(1000,1000,300)
mapa.add_circle(200,0,300)
mapa.add_circle(200,700,100)
mapa.add_circle(800,400,60)

fir = pf.ParticleFilter()

pos = np.array([map_size/2,map_size/2])
ori = 0
vel = 10
model = pf.Model(*pos,ori,vel,30, .1,.03)

dori = -np.pi/30

pop_size = 10000
model.set_map(mapa)
fir.set_model(model)
fir.setup(pop_size)

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
# from matplotlib import cm

# points = plt.scatter([],[])
# # print(points.__dict__)

posline, = plt.plot([],[],'.r',ms=15)
estline, = plt.plot([],[],'.y',ms=15)
plt.show()

step=np.pi/5/2


# @profile
def iter(i):
    global ori
    if not plt.fignum_exists(fig.number):
        # break;
        return False

    dori = np.random.uniform(-step,step)
    model.update_meas(np.random.normal(0, 10))
    
    print(i,pos,ori,flush=True)

    fir.update_weights()
    pest = np.array(fir.get_est())

    Neff = fir.get_effective_N()
    if Neff < pop_size*0.8:
        print('resample',Neff,flush=True)
        fir.resample(pf.RESAMPLE_TYPE.SUS)
    # fir.resample(pf.RESAMPLE_TYPE.SUS)
    # fir.resample(pf.RESAMPLE_TYPE.ROULETTE_WHEEL)

    prev = pos.copy()
    
    fir.drift(dori)
    model.update(dori,0)
    pos[0], pos[1], ori, tmp = model.get_real()

    poss.append(prev)
    ests.append(pest[:2])


    pop = fir.get_pop()
    points.set_data(pop[:,0],pop[:,1])
    
    posline.set_data(*prev)
    estline.set_data(*pest[:2])

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
    # if i == 10:
    #     import time
    #     time.sleep(5)
    #     break

# anim = FuncAnimation(fig, iter, init_func=init,
#                       # frames=200, interval=40)
#                       frames=1000, interval=40)
# # anim.save('pf_test.gif')
# writergif = PillowWriter(fps=25)
# anim.save("pf_test.gif",writer=writergif)
