import numpy as np
import matplotlib.pyplot as plt

import PFlib as pf


# map_size = 1000

pop_size = 10
d_vel = 0.1
d_ori = 0.1


#================================

mapa = pf.PrimitiveMap(1000)

mapa.add_line(-1,1,800)
mapa.add_circle(1000,1000,300)
mapa.add_circle(200,0,300)
mapa.add_circle(200,700,100)
mapa.add_circle(800,400,60)


model = pf.Model(mapa,d_ori,d_vel)
pop = model.get_random_pop(pop_size)
weights = model.get_weights(pop_size)
model.drift(pop)

print(pop)
pop = pf.roulette_wheel_resample(pop, weights)
pop = pf.roulette_wheel_resample(pop, weights)
pop = pf.roulette_wheel_resample(pop, weights)
model.drift(pop)
model.drift(pop)
model.drift(pop)
print(pop)

# plt.figure(figsize=(7,7))
# plt.imshow(np.array(mapa.get_grid()).T)
# plt.plot(*model.as_array(pop),'.')
# plt.xlim([0,1000])
# plt.ylim([0,1000])
# plt.axis('equal')
# plt.show()

