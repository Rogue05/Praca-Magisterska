# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 12:14:47 2021

@author: Wojtek
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# N = 100000
N = 10
L = 11
dx = 1
real_state = 2


pop = np.random.uniform(0,L,N)
w = np.ones(N)/N

NB = 11
ret = [plt.hist(pop,bins=np.linspace(0,L,NB))[0],]
# ret = []
ind = 0
xs = [np.sum(pop*w),]
xs2 = [real_state,]
ys = [ind,]

while real_state < NB-1:
    ind+=1
    real_state += dx
    pop += dx

    meas = L-pop
    real_meas = L - real_state
    prob = norm.pdf(meas-real_meas,0,1)
    prob[pop<0], prob[pop>L] = 0, 0
    w = w*prob
    w /= w.sum()

    est = np.sum(pop*w)
    ys.append(ind)
    xs.append(est)
    xs2.append(real_state)

    inds = np.arange(N)
    pop = pop[np.random.choice(inds, size = N, replace = True, p = w)]
    w = np.ones(N)/N
    pop += np.random.uniform(-0.01,0.01,N)

    print(ind, real_state, est)
    hs = plt.hist(pop,bins=np.linspace(0,L,NB))[0]
    ret.append(hs)

xs,ys,xs2 = np.array(xs),np.array(ys),np.array(xs2)

plt.gcf().clear()
plt.imshow(ret,cmap='gray_r')
plt.plot(xs2-1,ys,'.b',ms=15)
plt.plot(xs-1,ys,'.r')

plt.colorbar()
plt.xlabel('d')
plt.ylabel('t')
plt.arrow(1,0,1,0,color='r',width=.1)
# plt.axis('equal')