# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:40:19 2021

@author: Wojtek
"""

# print('asknfbglkjadfnglkdfblkafdjg',flush=True)

import matplotlib.pyplot as plt
# print('1',flush=True)
from matplotlib.animation import FuncAnimation, PillowWriter
# print('11')
from scipy.stats import norm

# print('12',flush=True)
from matplotlib import cm
# print('2',flush=True)
import PFlib as pf
# print('3',flush=True)
import numpy as np

map_size = 100

# print('=======',flush=True)
pf.setup_map(map_size,map_size,map_size//50)
pf.add_circle((map_size/2,3*map_size/4),map_size/10)
pf.add_circle((0,0),map_size/4)

# print('=======1',flush=True)
pf.add_circle((map_size*3/5,0),map_size/8)
pf.add_circle((map_size/10,map_size*3/5),map_size/11)
pos = (2*map_size//3,5*map_size//8)
size = (map_size//2,map_size//20)
pf.add_box(pos,size)

# print('=======1',flush=True)
pos = (3*map_size//4,map_size//4)
size = (map_size//10,map_size//10)
pf.add_box(pos,size)

# print('=======5',flush=True)
###============================
# pos = (map_size/2,)*2
# pos = (50,50)
# plt.axis('equal')
# plt.plot(*pos,'r.')

# for ori in np.linspace(0,2*np.pi,1000):
#     m = pf._get_meas(*pos,ori)
#     plt.plot(m*np.cos(ori)+pos[0],m*np.sin(ori)+pos[1],'.r',ms=1)
# # plt.show()
# plt.imshow(np.asarray(pf.get_grid()).T,cmap='gray')
# plt.xlim([0,map_size])
# plt.ylim([0,map_size])

grid = np.logical_not(pf.get_grid().T)

# print('=======got',flush=True)

pos = np.array([map_size/2,map_size/2])
ori = 0
# vel = 10
vel = 1

# dori = np.pi/70
dori = -np.pi/30

pop_size = 100000
# print('=======initing pop',flush=True)
pf.init_pop(pop_size)
# print('=======inited',flush=True)

errs = []
errs2 = []

pop = None

ori1 = []
ori2 = []
oris = []


def print_err():
    global pest
    estpos = np.array(pf.get_est())

    pop = pf.get_pop()
    H, xe, ye = np.histogram2d(pop[:,0],pop[:,1],bins=np.linspace(0,map_size,map_size))
    # print((-H).argsort()[:10])
    # inds = (-H).argsort()
    # print(inds.shape)
    ind = np.unravel_index(np.argsort(H, axis=None), H.shape)
    # print(ind)
    n = pop_size//1000
    estpos[0] = np.average(ind[0][-n:])
    estpos[1] = np.average(ind[1][-n:])

    pest = estpos
    posori = np.array([*pos,ori%(2*np.pi)])
    diffori = (np.cos(estpos[2])-np.cos(posori[2]))**2+(np.sin(estpos[2])-np.sin(posori[2]))**2

    diff = estpos-posori
    # diff[2] = diffori/2/np.pi
    # diff[:2]/=map_size
    diff[2] = map_size*diffori/2/np.pi

    errs.append(np.array([diff[0],diff[1],diff[2]]))
    # errs.append(np.sqrt(np.array([diff[0]**2,diff[1]**2,diff[2]**2])))

poss = []
ests = []

# fig = plt.figure(figsize=(15,15))
# # ax = plt.axes(xlim=(0, 100), ylim=(0, 100))

print('=======start',flush=True)
plt.axis('equal')
plt.xlim([0,map_size])
plt.ylim([0,map_size])
# plt.title(str(i)+' '+str(errs[-1]))
plt.imshow(grid,cmap='gray')
points, = plt.plot([],[],'.b',alpha=0.01)

posline, = plt.plot([],[],'.r',ms=15)
estline, = plt.plot([],[],'.y',ms=15)

# sca = 2
oris = []
step=np.pi/5
# # ori -= np.pi
# # from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
print('=======initing',flush=True)
alpha_mask = grid.astype(np.float32)
for i in range(1000):
# for i in range(300):
# def animate(i):
    # global dori,ori
    print(i,pos,ori,flush=True)
    # dori -= np.pi/3000/sca
    # dori *= 1.003
    # dori = np.pi-2*np.arccos(1/(3-ori))
    # dori -= np.pi/3000
    # dori -= ori/10000
    # vel -= ori/100

    # print(i,ori,pos)
    # if i>0:
    # print_err()

    # m=.1
    # while m < 2:
    #     print('Ściana',m)
    #     # dori = np.random.normal(0.,np.pi/30)
    #     m = pf._get_meas(*pos,ori+dori)

    # dori = np.random.uniform(-step,step)
    dori = -np.pi/30
    # vel = 10

    # while pf._get_meas(*pos,ori+dori) <5:
    #     dori+=np.pi/10

    # oris.append(ori)

    # print('Set model',flush=True)
    pf.set_model(*pos,ori,vel)
    # print('Set model 1',flush=True)
    m = pf._get_meas(*pos,ori)
    # print('Set model 2',flush=True)

    # m *= 1+np.random.normal(0,1)
    pf.update_weights(m)
    # print('Set model 3',flush=True)

    print_err()
    # print('printed',flush=True)
    # pf.resample(pf.RESAMPLE_TYPE.ROULETTE_WHEEL)
    # print('effective N',pf.get_effective_N())

    Neff = pf.get_effective_N()
    # # print('Neff',flush=True)

    if Neff < pop_size*0.5:
    # if Neff < pop_size*0.9:
        print('resample',Neff,flush=True)
        pf.resample(pf.RESAMPLE_TYPE.SUS)
    # pf.resample(pf.RESAMPLE_TYPE.SUS)

    # print('copy',flush=True)
    prev = pos.copy()

    # print('driffing',flush=True)
    # print(dori)
    pf.drift(dori)
    # print('driffed',flush=True)
    ori += dori
    if grid.T[np.int64(pos[0]+np.cos(ori)*vel)][np.int64(pos[1]+np.sin(ori)*vel)] == False:
        print('Bump')
        # break
        ori+=np.pi
    pos[0] = pos[0]+np.cos(ori)*vel
    pos[1] = pos[1]+np.sin(ori)*vel


    # pf.diffuse(1,.01)
    # pf.diffuse(.08,.03) # 100
    pf.diffuse(.9,.03)

    poss.append(prev)
    ests.append(pest[:2])


    # pop = pf.get_pop()
    # points.set_data(pop[:,0],pop[:,1])
    # posline.set_data(*prev)
    # estline.set_data(*pest[:2])

    # return points,posline,estline



    # if True:
    # # if i%10==9:
    #     pop = pf.get_pop()
    #     plt.axis('equal')
    #     # plt.title(str(i)+' '+str(errs[-1]))
    #     plt.imshow(grid,cmap='gray')
    #     plt.scatter(pop[:,0],pop[:,1],s=1,alpha=1000/pop_size)
    #     # plt.plot(pop[:,0],pop[:,1],'.b',alpha=1000/pop_size,ms=1)
    #     # blue_cm = cm.Blues
    #     # blue_cm.set_under('w',1)
    #     # plt.hist2d(pop[:,0],pop[:,1],
    #     #            bins=np.linspace(0,map_size,map_size*2),
    #     #            cmap=blue_cm,
    #     #             # cmin=.1,
    #     #            density=True)
    #     # plt.imshow(grid,cmap='gray')

    #     plt.hist2d(pop[:,0],pop[:,1],bins=np.linspace(0,map_size,100))
    #     plt.imshow(grid,cmap='gray')
    #     plt.plot(*prev,'.r')
    #     plt.plot(*pest[:2],'.k')
    #     # print(np.argmax(H))
    #     # ind = np.unravel_index(np.argmax(H, axis=None), H.shape)
    #     # print(ind)
    #     # bgm = BayesianGaussianMixture(n_components=2).fit(pop[:,:2])
    #     # plt.plot(bgm.means_[:,0],bgm.means_[:,1],'k.')

    #     plt.xlim([0,map_size])
    #     plt.ylim([0,map_size])
    #     plt.show()

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
# plt.plot(oris)
poss = np.array(poss)
ests = np.array(ests)

# INIT = 100
INIT = 0
plt.plot(errs[INIT:])
# plt.plot(errs)
plt.legend(['x','y','ori'])
plt.xlabel('nr. iteracji')
plt.ylabel('err')

plt.show()
# plt.plot(ori1,'.')
# plt.plot(ori2,'.')
# plt.ylim([0,10])

plt.imshow(grid,cmap='gray')
plt.axis('equal')
plt.plot(poss[INIT:,0],poss[INIT:,1],label='faktyczne położenie')
plt.plot(ests[INIT:,0],ests[INIT:,1],label='wyznaczone położenie')
plt.plot(*prev,'.r',label='końcowa pozycja')
plt.legend()
plt.show()