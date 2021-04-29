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

model = pf.Model(mapa,d_ori,d_vel)
pop = model.get_random_pop(pop_size)
weights = model.get_weights(pop_size)

#================================


# model.drift_state(real_state)
# model.drift(pop)
# meas = model.get_meas(real_state)
# weights = model.update_weights(meas, pop, weights)

# print(pop)
# pop = pf.sus_resample(pop, weights)
# print(weights)
# print(pop)

plt.ion()
fig = plt.figure(figsize=(7,7))
plt.imshow(np.array(mapa.get_grid()).T, cmap='gray')
real_line, = plt.plot([],[],'.r',ms=15)
pop_line, = plt.plot(*model.as_array(pop),'.b',alpha=0.01)
# estline, = plt.plot([],[],'.y',ms=15)

plt.xlim([-1,1001])
plt.ylim([-1,1001])
plt.axis('equal')
plt.show()
fig.canvas.draw()
fig.canvas.flush_events()

for i in range(1000):
	if not plt.fignum_exists(fig.number):
		break

	model.drift_state(real_state, 0.1, 0.0)
	model.drift(pop, 0.1, 0.0)
	print(i,real_state.x,real_state.y,real_state.ori,real_state.vel)

	meas = model.get_meas(real_state)
	weights = model.update_weights(meas, pop, weights)
	print(weights.max())

	effN = 1/(weights**2).sum()

	if effN < 0.8*pop_size:
		print('resample',effN)
		# pop = pf.roulette_wheel_resample(pop, weights)
		pop = pf.sus_resample(pop, weights)
		# pop = np.random.choice(pop,size=pop_size,p=weights,replace=True)
		weights = model.get_weights(pop_size)

	pop_line.set_data(*model.as_array(pop))
	real_line.set_data(real_state.x,real_state.y)

	# if i % 10 != 0:
	# 	continue
	fig.canvas.draw()
	fig.canvas.flush_events()





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
