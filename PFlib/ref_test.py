import numpy as np
import matplotlib.pyplot as plt

import PFlib as pf


map_size = 1000

pop_size = 10000
d_vel = 0.2
d_ori = 0.3

#================================

real_state = pf.robot_2d(500, 500, 0, 11)

mapa = pf.PrimitiveMap(map_size)

mapa.add_line(-1,1,800)
mapa.add_circle(1000,1000,300)
mapa.add_circle(200,0,300)
mapa.add_circle(200,700,100)
mapa.add_circle(800,400,60)


# plt.imshow(mapa.get_grid())
# plt.show()
# import sys
# sys.exit()


# model = pf.Model(mapa)
pop = pf.get_random_pop(mapa,pop_size)
weights = pf.get_uniform_weights(pop_size)

#================================


# # model.drift_state(real_state)
# # model.drift(pop)
# # meas = model.get_meas(real_state)
# # weights = model.update_weights(meas, pop, weights)

# # print(pop)
# # pop = pf.sus_resample(pop, weights)
# # print(weights)
# # print(pop)

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

# Gen92
# Hau11

diffx = []
diffy = []

popss = [pop_size,]
oriss = [real_state.ori,]
neffs = []

for i in range(1000):
	if not plt.fignum_exists(fig.number):
		break

	pf.drift_state(mapa, real_state, 0.1, 0.0)
	pf.drift_pop(mapa, pop, 0.1, 0.0, d_ori, d_vel)

	# pf.regularize(pop, 0.1, 0.0, 0.0)

	# print('      r ',i,real_state.x,real_state.y,real_state.ori,real_state.vel)

	meas = mapa.get_meas(real_state.x, real_state.y, real_state.ori)
	weights = pf.update_weights(mapa,meas, pop, weights)

	est_state = pf.get_est(pop,weights)
	# print('      e' ,i,est_state.x,est_state.y,est_state.ori,est_state.vel)	

	if i > 100:
		diffx.append(real_state.x-est_state.x)
		diffy.append(real_state.y-est_state.y)

	effN = 1/(weights**2).sum()

	if effN < 0.8*pop_size:
		print(i,'resample',pop_size,effN,flush=True)
		# exit()
		# pop = pf.roulette_wheel_resample(pop, weights)
		# print(len(pop))
		alpha = 100000
		# # print("get N",flush=True)
		# pop_size = pf.get_new_N(mapa, pop, weights, meas, alpha)
		# print(pop_size)
		neffs.append(pop_size)
		# print('got new',flush=True)
		popss.append(pop_size)
		oriss.append(real_state.ori)
		# # print("got N",flush=True)
		# # pop = pf.sus_resample(pop, weights)
		# pop = pf.roulette_wheel_resample(pop, weights, pop_size)
		pop = pf.sus_resample(pop, weights, pop_size)
		# print('resampled',flush=True)
		weights = pf.get_uniform_weights(pop_size)

	pop_line.set_data(*pf.as_array(pop))
	real_line.set_data(real_state.x,real_state.y)
	est_line.set_data(est_state.x,est_state.y)

	# if i % 10 != 1:
	# 	continue
	fig.canvas.draw()
	fig.canvas.flush_events()

# diff = np.sqrt(np.array(diffx)**2 + np.array(diffy)**2)/11
# # plt.plot(diffx,diffy,'.')
# # plt.axis('equal')

# plt.subplot(211)
# plt.plot(diff)
# plt.xlabel('t')
# plt.ylabel('err/v')
# # plt.figure()
# plt.subplot(212)
# plt.plot(neffs)
# plt.xlabel('t')
# plt.ylabel('N')
# # plt.figure()
# # plt.polar(oriss,popss,'.')
# plt.tight_layout()
# plt.show()


# pop = model.get_linear_pop(map_size) # tak, ma byc map_size
# print(len(pop))
# meas = model.get_meas(real_state)
# weights = model.update_weights(meas, pop, np.ones((len(pop),)))

# # plt.hist(weights,bins=50)
# plt.plot(weights,'.')
# # plt.pie(weights)

# # pop = pf.roulette_wheel_resample(weights, weights)
# print(weights.sum()/weights.shape[0])
# print(np.sum(weights[:360]))
# ret1 = weights[:559].sum()
# ret2 = weights[:648].sum()

# pop = pf.sus_resample(pop.copy(), weights)
# # print(ret)
# # print('=-=-=-=-=-=', weights[533], ret1, ret2)

# weights = model.update_weights(meas, pop, np.ones((len(pop),)))
# # plt.hist(weights,bins=50)
# plt.plot(weights,'.')
# plt.show()
