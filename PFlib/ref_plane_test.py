import numpy as np
import matplotlib.pyplot as plt

import PFlib as pf

pop_size = 100000
d_vel = 0.1
d_ori = 0.1

#================================

# real_state = pf.robot_2d(500, 500, np.pi/4, 10)
real_state = pf.robot_2d(100, 100, np.pi/4, 1)

# from noise import pnoise
# grid = pnoise(2**10,2**10,2**7)
# print('minmax =',grid.max(),grid.min())

# from scipy.ndimage.filters import gaussian_filter
grid = np.load('map.npy')
grid /= grid.max()
# grid = gaussian_filter(grid, sigma=100)
print('minmax =',grid.max(),grid.min())

p = 1000
grid = grid[p:p+300,p:p+300]

mapa = pf.HeightMap(grid)

# model = pf.Plane_Model(mapa,d_ori,d_vel)
model = pf.Model(mapa,d_ori,d_vel)
pop = model.get_random_pop(pop_size)
weights = model.get_weights(pop_size)

#================================



plt.ion()
fig = plt.figure(figsize=(7,7))
plt.imshow(np.array(mapa.get_grid()).T, cmap='gray')
real_line, = plt.plot([],[],'.r',ms=15)
pop_line, = plt.plot(*model.as_array(pop),'.b',alpha=0.01)
est_line, = plt.plot([],[],'.y',ms=15)

plt.xlim([-1,1001])
plt.ylim([-1,1001])
plt.axis('equal')
plt.show()
fig.canvas.draw()
fig.canvas.flush_events()

for i in range(1000):
	if not plt.fignum_exists(fig.number):
		break

	# model.drift_state(real_state, 0.1, 0.0)
	model.drift_state(real_state, 0.0, 0.0)
	# model.drift(pop, 0.1, 0.0)
	model.drift(pop, 0.0, 0.0)

	print('      r ',i,real_state.x,real_state.y,real_state.ori,real_state.vel)

	meas = model.get_meas(real_state)
	weights = model.update_weights(meas, pop, weights)

	est_state = model.get_est(pop,weights)
	print('      e' ,i,est_state.x,est_state.y,est_state.ori,est_state.vel)	

	effN = 1/(weights**2).sum()
	
	print('N_eff',effN)
	
	if effN < 0.8*pop_size:
		print('resample',effN)
		pop = pf.roulette_wheel_resample(pop, weights)
		# pop = pf.sus_resample(pop, weights)
		weights = model.get_weights(pop_size)

	pop_line.set_data(*model.as_array(pop))
	real_line.set_data(real_state.x,real_state.y)
	est_line.set_data(est_state.x,est_state.y)

	if i % 10 != 0:
		continue
	fig.canvas.draw()
	fig.canvas.flush_events()




# grid = np.load('map.npy')
# grid /= grid.max()

# x,y = np.meshgrid(np.linspace(0,1,1000),np.linspace(0,1,1000))
# grid = y

# mapa = pf.HeightMap(grid)
# model = pf.Model(mapa,d_ori,d_vel)
# pop = model.get_linear_pop(grid.shape[0])

# real_state = pf.robot_2d(500, grid.shape[1]/2, np.pi/4, 10)

# meas = meas = model.get_meas(real_state)
# weights = model.update_weights(meas, pop, np.ones((len(pop),)))

# # xs = [e.x for e in pop]

# plt.plot(weights)
# plt.show()