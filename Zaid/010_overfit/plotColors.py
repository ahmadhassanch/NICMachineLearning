import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

# make values from -5 to 5, for this example
zvals = np.random.rand(100,100)+1

# make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['blue','black'])
bounds=[-0.1,0.5,1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = plt.imshow(zvals,interpolation='nearest',
                    cmap = cmap,norm=norm)

# make a color bar
plt.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[0,0.5,1])

plt.show()
