import numpy as np
import matplotlib.pyplot as plt

import PFlib as pf


map_size = 1000

# pop_size = 10000

d_vel = 0.1
d_ori = 0.3
vel = 11

#================================


mapa = pf.PrimitiveMap(map_size)

mapa.add_line(-1,1,800)
mapa.add_circle(1000,1000,300)
mapa.add_circle(200,0,300)
mapa.add_circle(200,700,100)
mapa.add_circle(800,400,60)


def get_err(sus=True, pop_size = 1000, limit = 300,
			random_walker=False, est_type = 0):
	pop = pf.get_random_pop(mapa,pop_size)
	weights = pf.get_uniform_weights(pop_size)

	real_state = pf.robot_2d(500, 500, 0, vel)

	#================================

	diffx = []
	diffy = []

	popss = [pop_size,]
	oriss = [real_state.ori,]

	for i in range(limit):
		# if not plt.fignum_exists(fig.number):
		# 	break

		nori = 0.1
		if random_walker:
			nori = np.random.uniform(-.1,.1)

		pf.drift_state(mapa, real_state, nori, 0.0)
		pf.drift_pop(mapa, pop, nori, 0.0, d_ori, d_vel)


		meas = mapa.get_meas(real_state.x, real_state.y, real_state.ori)
		weights = pf.update_weights(mapa,meas, pop, weights)

		est_state = pf.get_est(pop,weights)
		# print('      e' ,i,est_state.x,est_state.y,est_state.ori,est_state.vel)	
		
		if est_type == 0:
			ex = np.sum(pop['x']*weights)/weights.sum()
			ey = np.sum(pop['y']*weights)/weights.sum()

		elif est_type == 1:
			# inds = weights.argsort()[:pop_size//10]
			inds = (-weights).argsort()[:pop_size//10]
			ex = np.sum(pop['x'][inds]*weights[inds])/weights[inds].sum()
			ey = np.sum(pop['y'][inds]*weights[inds])/weights[inds].sum()
		else:
			ex = pop['x'][weights.argmax()]
			ey = pop['y'][weights.argmax()]
		# print('      e' ,i,est_state.x,est_state.y,ex)

		# if i > 100:
		# diffx.append(real_state.x-est_state.x)
		# diffy.append(real_state.y-est_state.y)
		diffx.append(real_state.x-ex)
		diffy.append(real_state.y-ey)

		effN = 1/(weights**2).sum()

		if effN < 0.8*pop_size:
			# print(i,'resample',pop_size,effN,flush=True)
			alpha = 100
			# pop_size = pf.get_new_N(mapa, pop, weights, meas, alpha)
			popss.append(pop_size)
			oriss.append(real_state.ori)
			if sus:
				pop = pf.sus_resample(pop, weights, pop_size)
			else:
				pop = pf.roulette_wheel_resample(pop, weights, pop_size)
			weights = pf.get_uniform_weights(pop_size)
	# plt.plot(pop)
	return np.sqrt(np.array(diffx)**2+np.array(diffy)**2)/vel

# plt.plot(diffx,label='$x_{{err}}$')
# plt.plot(diffy,label='$y_{{err}}$')

# for N in [100,300,1000]:
names = ['średnia ważona wszystkich',
'średnia ważona najlepszych 10%',
'najlepszy osobnik']
for est_type in range(3):
# for est_type in [1,]:
	avgs = 100
	sus = np.zeros(300)
	for a in range(avgs):
		print(est_type,a)
		sus += get_err(True, est_type=est_type)
	sus/=avgs
	plt.plot(sus,label=names[est_type])

plt.legend()
plt.xlabel('t')
plt.ylabel('$\\frac{{\\sqrt{{x^2_{{err}}+y^2_{{err}}}}}}{{v}}$')
# plt.title('Wpływ metody próbkowania na jakość estymacji.')
plt.show()


