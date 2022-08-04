import cv2
import glob
from PIL import Image
import glob
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    Red = '\033[91m'
    Green = '\033[92m'
    Blue = '\033[94m'
    Cyan = '\033[96m'
    White = '\033[97m'
    Yellow = '\033[93m'
    Magenta = '\033[95m'
    Grey = '\033[90m'
    Black = '\033[90m'
    Default = '\033[99m'


class Index:
    ind = 0

    def b_green(self, event, num):
        print(num)
        print(f"{bcolors.OKGREEN}Green{bcolors.ENDC}")

    def b_not(self, event):

        print(f"{bcolors.White}Not{bcolors.ENDC}")


    def b_red(self, event):
        print(f"{bcolors.Red}Red{bcolors.ENDC}")


    def b_ign(self, event):
        print(f"{bcolors.Black}Ignore{bcolors.ENDC}")



if __name__ == '__main__':

    image_list = []
    for filename in glob.glob('test2/*.png'):  # assuming gif
        print(filename)
        im = Image.open(filename)
        image_list.append(im)

        f, axarr = plt.subplots(1, 1, sharex=True, sharey=True)
        axarr.title.set_text('image')
        axarr.imshow(im)

        callback = Index()
        ax_ign = plt.axes([0.12, 0.05, 0.1, 0.075])
        ax_not = plt.axes([0.59, 0.05, 0.1, 0.075])
        ax_red = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_green = plt.axes([0.81, 0.05, 0.1, 0.075])

        b_not = Button(ax_not, 'Not TL', color='b')
        b_not.on_clicked(callback.b_not)

        b_ign = Button(ax_ign, 'Ignore', color='w')
        b_ign.on_clicked(callback.b_ign)
        # b_ign.on_clicked(im.close())
        b_red = Button(ax_red, 'Red', color='r')
        b_red.on_clicked(callback.b_red)
        # b_red.on_clicked(im.close())
        b_green = Button(ax_green, 'Green', color='g')
        num=2
        b_green.on_clicked(callback.b_green(num))
        # b_green.on_clicked(im.close())

        plt.show()
