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


def get_err(sus=True, pop_size = 1000, limit = 300, random_walker=False):
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

		# if i > 100:
		diffx.append(real_state.x-est_state.x)
		diffy.append(real_state.y-est_state.y)

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

for N in [100,300,1000]:
	avgs = 100
	sus = np.zeros(300)
	for a in range(avgs):
		print('sus',a,N)
		sus += get_err(True, pop_size=N)
	sus/=avgs
	plt.plot(sus,label='SUS N='+str(N))

	rou = np.zeros(300)
	for a in range(avgs):
		print('rou',a,N)
		rou += get_err(False, pop_size=N)
	rou/=avgs
	plt.plot(rou,label='Roulette N='+str(N))

plt.legend()
plt.xlabel('t')
plt.ylabel('$\\frac{{\\sqrt{{x^2_{{err}}+y^2_{{err}}}}}}{{v}}$')
# plt.title('Wpływ metody próbkowania na jakość estymacji.')
plt.show()
# plt.figure()


# plt.plot(get_err(pop_size=10),label='N=10')


# # ----------------------

# L = 1000
# # L = 10
# a100 = np.zeros(L)
# T = 100
# # T = 10
# for t in range(T):
# 	print(t)
# 	tmp = get_err(pop_size=100,limit=L,random_walker=False)
# 	a100 += tmp
# a100/=T

# plt.plot(a100,label='N=100')

# a300 = np.zeros(L)
# # T = 10
# for t in range(T):
# 	print(t)
# 	tmp = get_err(pop_size=300,limit=L,random_walker=False)
# 	a300 += tmp
# a300/=T

# plt.plot(a300,label='N=300')

# a500 = np.zeros(L)
# # T = 10
# for t in range(T):
# 	print(t)
# 	tmp = get_err(pop_size=500,limit=L,random_walker=False)
# 	a500 += tmp
# a500/=T

# plt.plot(a500,label='N=500')


# a1000 = np.zeros(L)
# # T = 10
# for t in range(T):
# 	print(t)
# 	tmp = get_err(pop_size=1000,limit=L)
# 	# plt.plot(tmp,label=str(t))
# 	a1000 += tmp
# a1000/=T
# plt.plot(a1000,label='N=1000')

# a10000 = np.zeros(L)
# # T = 10
# for t in range(T):
# 	print(t)
# 	tmp = get_err(pop_size=10000,limit=L)
# 	# plt.plot(tmp,label=str(t))
# 	a10000 += tmp
# a10000/=T
# plt.plot(a10000,label='N=10000')

# plt.legend()
# plt.xlabel('t')
# # plt.ylabel('$\\sqrt{{x^2_{{err}}+y^2_{{err}}}}/v$')
# plt.ylabel('$\\frac{{\\sqrt{{x^2_{{err}}+y^2_{{err}}}}}}{{v}}$')
# # plt.title('Wpływ liczby cząstek jakość estymacji.')
# plt.show()

