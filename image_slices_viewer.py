"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from drag_lines import draggable_lines
from vars import IndexTracker
import cv2
# class IndexTracker(object):
#     def __init__(self, ax, ax2, ax3, X):
#         self.ax = ax
#         self.ax2 = ax2
#         self.ax3 = ax3
#         ax.set_title('use scroll wheel to navigate images')

#         self.X = X
#         rows, cols, self.slices = X.shape
#         self.ind = self.slices//2



#         self.im = ax.imshow(self.X[:, :, self.ind])
#         self.update()

#     def onscroll(self, event):
#         print("%s %s" % (event.button, event.step))
#         if event.button == 'up':
#             self.ind = (self.ind + 1) % self.slices
#         else:
#             self.ind = (self.ind - 1) % self.slices
#         self.update()

#     def update(self):
#         self.im.set_data(self.X[:, :, self.ind])
#         self.ax.set_ylabel('slice %s' % self.ind)
#         self.im.axes.figure.canvas.draw()

#         self.im2.set_data(self.x1, X[:, 8, 1])
#         self.im3.set_data(self.y1, X[:, 8, 1])
#         self.im2.axes.figure.canvas.draw()
#         self.im3.axes.figure.canvas.draw()


        
        
     
gs = gridspec.GridSpec(2, 2)
fig = plt.figure()
fig.set_size_inches(10.5, 6)
plt.subplots_adjust(hspace = 0.4)

ax3 = fig.add_subplot(gs[1, 0]) # row 0, col 0
ax2 = fig.add_subplot(gs[1, 1]) # row 0, col 1
ax = fig.add_subplot(gs[0,:]) # row 0, span all columns

#Get data
X = cv2.imread("test_resized.jpg")
#X = np.random.rand(20, 20, 40)

Vline = draggable_lines(ax, ax2, ax3, "h", 0.5,  X)
Tline = draggable_lines(ax, ax2, ax3, "v", 0.5, X)



axcolor = 'lightgoldenrodyellow'
f0 = 3
delta_f = 1
ax.margins(0)
ax2.margins(0)
# tracker = IndexTracker(ax, ax2, ax3, X)

# Reset lines
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    Vline.reset()
    Tline.reset()
button.on_clicked(reset)

# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()
