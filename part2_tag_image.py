import cv2
import glob
from PIL import Image
import glob
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

src_img_dir = 'test2'
des_imag_dir = 'test2_tag_image'
img = ''
img_name = ''
new_name = ''

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

    def b_green(self, event):
        img_name2 = img_name.replace(src_img_dir, des_imag_dir)
        new_name = img_name2.replace(".png", "_G.png")
        plt.imsave(new_name, img)
        print(f"{bcolors.OKGREEN}{new_name}{bcolors.ENDC}")
        plt.close()


    def b_red(self, event):
        img_name2 = img_name.replace(src_img_dir,des_imag_dir)
        new_name = img_name2.replace(".png", "_R.png")
        plt.imsave(new_name,img)
        print(f"{bcolors.Red}{new_name}{bcolors.ENDC}")
        plt.close()


    def b_not(self, event):
        img_name2 = img_name.replace(src_img_dir, des_imag_dir)
        new_name = img_name2.replace(".png", "_N.png")
        plt.imsave(new_name, img)
        print(f"{bcolors.Blue}{new_name}{bcolors.ENDC}")
        plt.close()


    def b_ign(self, event):
        img_name2 = img_name.replace(src_img_dir, des_imag_dir)
        new_name = img_name2.replace(".png", "_I.png")
        plt.imsave(new_name, img)
        print(f"{bcolors.Black}{new_name}{bcolors.ENDC}")
        plt.close()


if __name__ == '__main__':

    image_list = []
    for filename in glob.glob(f'{src_img_dir}/*.png'):  # assuming gif
        print(filename)
        img = Image.open(filename)
        img_name = filename
        f, axarr = plt.subplots(1, 1, sharex=True, sharey=True)
        axarr.title.set_text('image')
        axarr.imshow(img)

        callback = Index()
        ax_ign = plt.axes([0.12, 0.05, 0.1, 0.075])
        ax_not = plt.axes([0.59, 0.05, 0.1, 0.075])
        ax_red = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_green = plt.axes([0.81, 0.05, 0.1, 0.075])

        b_not = Button(ax_not, 'Not TL', color='b')
        b_not.on_clicked(callback.b_not)

        b_ign = Button(ax_ign, 'Ignore', color='w')
        b_ign.on_clicked(callback.b_ign)

        b_red = Button(ax_red, 'Red', color='r')
        b_red.on_clicked(callback.b_red)

        b_green = Button(ax_green, 'Green', color='g')
        b_green.on_clicked(callback.b_green)

        plt.show()
        img_name = ''
        new_name = ''
