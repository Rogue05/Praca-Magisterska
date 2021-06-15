import numpy as np
import matplotlib.pyplot as plt

import PFlib as pf

pop_size = 10000
d_vel = 0.1
d_ori = 0.1

#================================

# real_state = pf.robot_2d(500, 500, np.pi/4, 10)
real_state = pf.robot_2d(500, 100, np.pi/4, 10)

# from noise import pnoise
# grid = pnoise(2**10,2**10,2**7)
# print('minmax =',grid.max(),grid.min())

grid = np.load('map.npy')
grid /= grid.max()
print('minmax =',grid.max(),grid.min())

mapa = pf.HeightMap(grid)

# model = pf.Model(mapa)
pop = pf.get_random_pop(mapa, pop_size)
weights = pf.get_uniform_weights(pop_size)

pop['ori'] = np.pi/4 # hehe

#================================



plt.ion()
fig = plt.figure(figsize=(7,7))
plt.imshow(np.array(mapa.get_grid()).T, cmap='gray')
real_line, = plt.plot([],[],'.r',ms=15)
pop_line, = plt.plot(*pf.as_array(pop),'.b',alpha=0.01)
est_line, = plt.plot([],[],'.y',ms=15)

plt.xlim([-1,1001])
plt.ylim([-1,1001])
plt.axis('equal')
plt.show()
fig.canvas.draw()
fig.canvas.flush_events()

for i in range(300):
	# if not plt.fignum_exists(fig.number):
	# 	break

	pf.drift_state(mapa, real_state, 0.0, 0.0)
	pf.drift_pop(mapa, pop, 0.0, 0.0, d_ori, d_vel)

	print('      r',i,real_state.x,real_state.y,real_state.ori,real_state.vel)

	# # meas = model.get_meas(real_state)
	# meas = mapa.get_meas(real_state.x,real_state.y,real_state.ori)
	meas = np.random.uniform(0,1)
	weights = pf.update_weights(mapa, meas, pop, weights)

	print(i,'square sum',(weights**2).sum())
	if np.isnan((weights**2).sum()):
		print(weights)
		break

	est_state = pf.get_est(pop,weights)
	print('      e' ,i,est_state.x,est_state.y,est_state.ori,est_state.vel)	

	effN = 1/(weights**2).sum()
	
	print(i,'N_eff',effN,(weights**2).sum())
	
	if effN < 0.8*pop_size:
		print('resample',effN,pop_size)

		# alpha = 100
		# pop_size = pf.get_new_N(mapa, pop, weights, meas, alpha)
		# print('Done-',flush=True)
		
		pop = pf.roulette_wheel_resample(pop, weights, pop_size)
		# pop = pf.sus_resample(pop, weights, pop_size)
		weights = pf.get_uniform_weights(pop_size)
		if len(pop) != len(weights):
			print('FIX SIZES')
			break
		# print('Done',flush=True)

	pop_line.set_data(*pf.as_array(pop))
	real_line.set_data(real_state.x,real_state.y)
	est_line.set_data(est_state.x,est_state.y)

	if i % 10 != 1:
		continue
	fig.canvas.draw()
	fig.canvas.flush_events()
