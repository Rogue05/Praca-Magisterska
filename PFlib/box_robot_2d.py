import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

# w = np.random.uniform(0,1,10)
# w /= w.sum()
# print(w)

# step = w.sum()/len(w)
# init = np.random.uniform(0,step)
# print(step,init)

# inds = init+np.arange(0,len(w))*step
# print(inds)

# ret = []
# i = 0
# oldi = i

# suma = w[0]
# cnt = 0

# fin = []
# for j in range(len(w)):
# 	p = init+j*step

# 	while suma < p:
# 		i+=1
# 		suma+=w[i]

# 	cnt+=1
# 	ret.append(i)

# fin.append((i,cnt))
# print(ret)
# print(fin)


# import sys
# sys.exit()

import PFlib as pf

# pop_sqrt = np.int64(np.sqrt(64))
pop_sqrt = np.int64(np.sqrt(1000))
pop_size = pop_sqrt**2
d_vel = 0.1
d_ori = 0.01

patches = list([pat.Rectangle((0,0),0,0,fill=False,color='y') for _ in range(pop_size)])

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
realpos, = plt.plot(real_state.x,real_state.y,'g')
estpos, = plt.plot(real_state.x,real_state.y,'r')
plt.ion()
plt.show()

rx, ry = [], []
ex, ey = [], []

errx, erry = [], []

cnt = 0
tmp = 0


ending = False
for ind in range(300):
	real_state.ori += 0.1
	real_state.x = real_state.x + np.cos(real_state.ori)*real_state.vel
	real_state.y = real_state.y + np.sin(real_state.ori)*real_state.vel
	bpf.drift(0.1,0.0)
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
		# print('           resample',cnt%4)
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

# # ,p[1]-p[0],p[3]-p[2]


# # 	if ind% 10 ==0:
# # 		plt.imshow(mapg)
# # 		pop = bpf.get_pop()
# # 		plot_pop(pop)
# # 		plt.plot(real_state.x,real_state.y,'.w')
# # 		plt.show()

# # pop = bpf.get_pop()
# # plot_pop(pop)
# # plt.plot(real_state.x,real_state.y,'.w')
# # plt.show()

plt.ioff()
# plt.show()
plt.figure(2)
errx = np.array(errx)
erry = np.array(erry)
err = np.sqrt(errx**2+erry**2)/10 #/vel
plt.plot(err)
plt.ylim([0,5])
plt.ylabel('$d_{{err}}/v$')

# plt.plot(errx)
# plt.plot(erry)
# plt.ylim([-50,50])

plt.xlabel('t')
plt.show()
