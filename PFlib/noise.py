# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:15:08 2021

@author: Wojtek
"""

import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def pnoise(width, height, lim):
    grid = np.random.randint(0,2,(width,height))

    buff = grid.copy().astype(np.float32)
    sca = 1
    while sca<lim:
        sca*=2
        grid = zoom(grid[:grid.shape[0]//2,:grid.shape[1]//2],2,order=1)*2
        buff+=grid
    return buff/buff.max()

if __name__ == '__main__':
    for _ in range(10):
        res = pnoise(2**10,2**10,2**7)
        plt.imshow(res)
        # res2 = pnoise(2**10,2**10,2**7)
        # plt.imshow(np.max(res,res2,axis=0))
        # print(np.max([res,res2],axis=0))
        # tmp = np.max([res,res2],axis=0)
        # plt.imshow(tmp/tmp.max())
        plt.show()