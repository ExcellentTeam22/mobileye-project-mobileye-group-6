import cv2
import glob
from PIL import Image
import glob
import matplotlib.pyplot as plt




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

freqs = np.arange(2, 20, 3)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l, = plt.plot(t, s, lw=2)


class Index:
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

if __name__ == '__main__':

    image_list = []
    for filename in glob.glob('test2/*.png'): #assuming gif
        print(filename)
        im=Image.open(filename)
        image_list.append(im)

        f, axarr = plt.subplots(1,1, sharex=True, sharey=True)
        axarr.title.set_text('image')
        axarr.imshow(im)

        callback = Index()
        ax_ign = plt.axes([0.48, 0.05, 0.1, 0.075])
        ax_not = plt.axes([0.59, 0.05, 0.1, 0.075])
        ax_red = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_green = plt.axes([0.81, 0.05, 0.1, 0.075])
        b_not = Button(ax_not, 'not TL',color='b')
        b_ign = Button(ax_ign, 'not TL',color='w')

        b_red = Button(ax_red, 'Red',color='r')
        b_green = Button(ax_green, 'Green',color='g')
        # bnext.on_clicked(callback.next)
        # bprev.on_clicked(callback.prev)

        plt.show()
