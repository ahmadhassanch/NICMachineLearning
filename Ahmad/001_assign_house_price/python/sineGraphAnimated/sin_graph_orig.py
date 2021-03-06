import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig0 = plt.figure(1)
ax0 = fig0.add_subplot(111, autoscale_on=False, xlim=(0, 2), ylim=(-2, 2))
#plt.show(block=False)

fig1 = plt.figure(0)
ax1 = fig1.add_subplot(222, autoscale_on=False, xlim=(0, 2), ylim=(-2, 2))
#plt.show(block=False)

ax0.grid()
ax1.grid()

#ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line0, = ax0.plot([], [], 'ro-', lw=2)
line1, = ax1.plot([], [], 'bo-', lw=2)

# initialization function: plot the background of each frame
def init0():
    line0.set_data([], [])
    return line0,

# initialization function: plot the background of each frame
def init1():
    line0.set_data([], [])
    return line1,

# animation function.  This is called sequentially
def animate0(i):
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line0.set_data((x, y))
    return line0

def animate1(i):
    y = np.cos(2 * np.pi * (x - 0.01 * i))
    line1.set_data((x, y))
    return line1

# call the animator.  blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=False)
x = np.linspace(0, 2, 25)
    
print 'start anim'
anim0 = animation.FuncAnimation(fig0, animate0, init_func=init0, frames=200, interval=20, blit=False, repeat=False)
anim1 = animation.FuncAnimation(fig1, animate1, init_func=init1, frames=200, interval=20, blit=False, repeat=False)
print 'end_anim'

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show(block=False)
