import numpy as np
import matplotlib.pyplot as plt

import PFlib as pf


map_size = 1000

pop_size = 10000
d_vel = 0.2
d_ori = 0.3

real_state = pf.robot_2d(500, 500, 0, 11)

mapa = pf.PrimitiveMap(map_size)
mapa.add_line(-1,1,800)
mapa.add_circle(1000,1000,300)
mapa.add_circle(200,0,300)
mapa.add_circle(200,700,100)
mapa.add_circle(800,400,60)

pop = pf.get_random_pop(mapa,pop_size)
weights = pf.get_uniform_weights(pop_size)


def plot_state(pos, pop, real_state, est_state, title):
	print
	plt.subplot(pos)
	plt.imshow(np.array(mapa.get_grid()).T, cmap='gray')
	plt.plot(real_state.x,real_state.y,'.r',ms=15)
	plt.plot(pop[0],pop[1],'.b',alpha=0.01)
	plt.plot(est_state.x,est_state.y,'.y',ms=15)

	plt.xlim([-1,1001])
	plt.ylim([-1,1001])
	plt.axis('equal')
	plt.title(title)
	# plt.show()

plt.figure(figsize=(15,5))


isss = [0,100,200]
for i in range(max(isss)+1):

	pf.drift_state(mapa, real_state, 0.1, 0.0)
	pf.drift_pop(mapa, pop, 0.1, 0.0, d_ori, d_vel)

	# meas = mapa.get_meas(real_state.x, real_state.y, real_state.ori)
	meas = np.random.uniform(0,1000)
	weights = pf.update_weights(mapa,meas, pop, weights)

	est_state = pf.get_est(pop,weights)

	effN = 1/(weights**2).sum()

	if effN < 0.8*pop_size:
		print(i,'resample',pop_size,effN,flush=True)
		# pop = pf.roulette_wheel_resample(pop, weights, pop_size)
		pop = pf.sus_resample(pop, weights, pop_size)
		weights = pf.get_uniform_weights(pop_size)


	if i in isss:
		pos = {0:131,100:132,200:133}
		plot_state(pos[i],pf.as_array(pop),
			real_state,est_state,'k='+str(i+1))
plt.show()
