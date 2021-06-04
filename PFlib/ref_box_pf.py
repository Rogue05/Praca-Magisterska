import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

import PFlib as pf

pop_sqrt = 10
pop_size = pop_sqrt**2
d_vel = 0.1
d_ori = 0.01

patches = list([pat.Rectangle((0,0),0,0,fill=False,color='r') for _ in range(pop_size)])

for rct in patches:
	plt.gca().add_patch(rct)
#================================
# def plot_pop(pop):
# 	for p in pop:
# 		rct = pat.Rectangle((p[0],p[2]),
# 			p[1]-p[0],p[3]-p[2],fill=False,color='r')
# 		plt.gca().add_patch(rct)


real_state = pf.robot_2d(500, 500, np.pi/4, 10)
# real_state = pf.robot_2d(100, 100, np.pi/4, 10)

# from noise import pnoise
# grid = pnoise(2**10,2**10,2**7)
# print('minmax =',grid.max(),grid.min())

grid = np.load('map.npy')
grid /= grid.max()
print('minmax =',grid.max(),grid.min())

mapa = pf.HeightMap(grid)

bpf = pf.BoxParticleFilter(mapa)
bpf.init_pop(pop_sqrt)

print('============================')

mapg = np.array(mapa.get_grid()).T
plt.imshow(mapg)
realpos, = plt.plot(real_state.x,real_state.y,'.w')
plt.ion()
plt.show()

cnt = 0
tmp = 0
for ind in range(200):
	real_state.x = real_state.x + np.cos(real_state.ori)*real_state.vel
	real_state.y = real_state.y + np.sin(real_state.ori)*real_state.vel
	bpf.drift(0.0,0.0)
	meas = mapa.get_meas(real_state.x,real_state.y,real_state.ori)
	effN = bpf.update_weights(meas, 0.01)
	# print('effN',effN,meas)
	print(ind, 'effN',effN, bpf.get_est(), real_state.x, real_state.y)
	# est = bpf.get_est()
	# if effN < 0.5*pop_size or ind%10==9:
	tmp+=1
	if effN < 0.5*pop_size or tmp > 10:
		tmp=0
		cnt+=1
		print('           resample',cnt%4)
		bpf.resample()

	if not plt.get_fignums():
		break
	if not ind%10==0:
		continue
	pop = bpf.get_pop()
	for i, p in enumerate(pop):
		# print(p[0],p[2])
		patches[i].set_xy((p[0],p[2]))
		patches[i].set_width(p[1]-p[0])
		patches[i].set_height(p[3]-p[2])
		pass
	realpos.set_data([real_state.x,],[real_state.y,])
	# plt.plot(real_state.x,real_state.y,'.w')
	plt.gcf().canvas.draw()
	plt.gcf().canvas.flush_events()

# ,p[1]-p[0],p[3]-p[2]


# 	if ind% 10 ==0:
# 		plt.imshow(mapg)
# 		pop = bpf.get_pop()
# 		plot_pop(pop)
# 		plt.plot(real_state.x,real_state.y,'.w')
# 		plt.show()

# pop = bpf.get_pop()
# plot_pop(pop)
# plt.plot(real_state.x,real_state.y,'.w')
# plt.show()


