import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

import PFlib as pf






def get_err(sus=True, pop_size = 1000, limit = 300, random_walker=False):
	map_size = 1000
	vel = 10
	d_vel = 0.1
	d_ori = 0.3

	mapa = pf.PrimitiveMap(map_size)

	mapa.add_line(-1,1,800)
	mapa.add_circle(1000,1000,300)
	mapa.add_circle(200,0,300)
	mapa.add_circle(200,700,100)
	mapa.add_circle(800,400,60)

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
			print(i,'resample',pop_size,effN,flush=True)
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




pf_err = get_err(pop_size=10000)



# pop_sqrt = np.int64(np.sqrt(64))
pop_sqrt = np.int64(np.sqrt(1000))
pop_size = pop_sqrt**2
d_vel = 0.1
d_ori = 0.01

patches = list([pat.Rectangle((0,0),0,0,fill=False,color='b') for _ in range(pop_size)])

for rct in patches:
	plt.gca().add_patch(rct)
#================================
# def plot_pop(pop):
# 	for p in pop:
# 		rct = pat.Rectangle((p[0],p[2]),
# 			p[1]-p[0],p[3]-p[2],fill=False,color='r')
# 		plt.gca().add_patch(rct)

map_size = 1000

# real_state = pf.robot_2d(500, 500, np.pi/4, 10)
# real_state = pf.robot_2d(100, 500, np.pi/4, 10)
real_state = pf.robot_2d(map_size//2, map_size//2, np.pi/4, 10)

# mapa = pf.HeightMap(grid)
mapa = pf.PrimitiveMap(map_size)
mapa.add_line(-1,1,800)
mapa.add_circle(1000,1000,300)
mapa.add_circle(200,0,300)
mapa.add_circle(200,700,100)
mapa.add_circle(800,400,60)
grid = np.array(mapa.get_grid())

bpf = pf.BoxParticleFilter(mapa)
bpf.init_pop(pop_sqrt)

print('============================')

mapg = np.array(mapa.get_grid()).T
plt.imshow(mapg, cmap='gray')
plt.xlim([-1,grid.shape[0]+1])
plt.ylim([-1,grid.shape[1]+1])
plt.xlabel('x')
plt.ylabel('y')
realpos, = plt.plot(real_state.x,real_state.y,'r')
estpos, = plt.plot(real_state.x,real_state.y,'y')
plt.ion()
plt.show()

rx, ry = [], []
ex, ey = [], []

errx, erry = [], []

cnt = 0
tmp = 0

dori = 0.1

ending = False
for ind in range(300):
	# dori+=0.01
	# dori *= 1.005
	real_state.ori += dori
	# real_state.ori += np.random.uniform(-.1,.1)
	real_state.x = real_state.x + np.cos(real_state.ori)*real_state.vel
	real_state.y = real_state.y + np.sin(real_state.ori)*real_state.vel
	# bpf.drift(dori,0.0)
	bpf.drift(dori,0.0)
	meas = mapa.get_meas(real_state.x,real_state.y,real_state.ori)
	effN = bpf.update_weights(meas, 0.01)

	coeff = bpf.get_coeff()
	print(ind,'    paving', coeff)

	if coeff > 0.04:
		bpf.reinit_pop()
		continue
		ending = True

	if real_state.x >= grid.shape[0] or\
		real_state.y >= grid.shape[1]:
		break

	if np.isnan(effN):
		# print('reinit')
		bpf.init_pop(pop_sqrt)
		# print('done')
		continue

	est = bpf.get_est()
	# print(ind, 'effN',effN, est, real_state.x, real_state.y)
	# est = bpf.get_est()
	# if effN < 0.5*pop_size or ind%10==9:
	tmp+=1
	# if effN < 0.5*pop_size or tmp > 10:
	if effN < 0.5*pop_size:
		tmp=0
		cnt+=1
		print('           resample',cnt%4)
		bpf.resample()



	if not plt.get_fignums():
		break

	rx.append(real_state.x);ry.append(real_state.y);
	realpos.set_data(rx, ry)
	if ind >= 0:
		ex.append(est[0]);ey.append(est[1]);
		estpos.set_data(ex, ey)

		errx.append(real_state.x-est[0])
		erry.append(real_state.y-est[1])
	# plt.plot(real_state.x,real_state.y,'.g',ms=1)
	# plt.plot(est[0],est[1],'.r',ms=1)
	
	if not ind%10==0:
		continue
	pop = bpf.get_pop()
	for i, p in enumerate(pop):
		# print(p[0],p[2])
		patches[i].set_xy((p[0],p[2]))
		patches[i].set_width(p[1]-p[0])
		patches[i].set_height(p[3]-p[2])
		pass
	# realpos.set_data([real_state.x,],[real_state.y,])
	plt.gcf().canvas.draw()
	plt.gcf().canvas.flush_events()
	# import time
	# time.sleep(1)




plt.ioff()
# plt.show()





plt.figure(2)
errx = np.array(errx)
erry = np.array(erry)
err = np.sqrt(errx**2+erry**2)/10 #/vel
plt.plot(pf_err,label='podstawowy algorytm')
plt.plot(err,label='Box Particle Filter')
plt.legend()
plt.ylim([0,5])
plt.ylabel('$\\frac{{\\sqrt{{x^2_{{err}}+y^2_{{err}}}}}}{{v}}$')

# plt.plot(errx)
# plt.plot(erry)
# plt.ylim([-50,50])

plt.xlabel('t')
plt.show()
