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


class Index(object):
    def __init__(self, source, dest):
        self.source= source
        self.dest =dest
        self.img = ''
        self.img_name = ''
        self.new_name = ''


    def classify(self):

        for self.filename in glob.glob(f'{self.source}/*.png'):  # assuming gif
            print(self.filename)
            self.img = Image.open(self.filename)
            self.img_name = self.filename
            f, axarr = plt.subplots(1, 1, sharex=True, sharey=True)
            axarr.title.set_text('image')
            axarr.imshow(self.img)


            ax_ign = plt.axes([0.12, 0.05, 0.1, 0.075])
            ax_not = plt.axes([0.59, 0.05, 0.1, 0.075])
            ax_red = plt.axes([0.7, 0.05, 0.1, 0.075])
            ax_green = plt.axes([0.81, 0.05, 0.1, 0.075])

            b_not = Button(ax_not, 'Not TL', color='b')
            b_not.on_clicked(self.b_not)

            b_ign = Button(ax_ign, 'Ignore', color='w')
            b_ign.on_clicked(self.b_ign)

            b_red = Button(ax_red, 'Red', color='r')
            b_red.on_clicked(self.b_red)

            b_green = Button(ax_green, 'Green', color='g')
            b_green.on_clicked(self.b_green)

            plt.show()
            self.img_name = ''
            self.new_name = ''

    def b_green(self, event):
        img_name2 = self.img_name.replace(src_img_dir, des_imag_dir)
        self.new_name = img_name2.replace(".png", "_G.png")
        self.save_image(self.new_name)
        print(f"{bcolors.Green}{self.new_name}{bcolors.ENDC}")


    def b_red(self, event):
        img_name2 = self.img_name.replace(self.source,self.dest)
        self.new_name = img_name2.replace(".png", "_R.png")
        self.save_image(self.new_name)
        print(f"{bcolors.Red}{self.new_name}{bcolors.ENDC}")


    def b_not(self, event):
        img_name2 = self.img_name.replace(self.source,self.dest)
        self.new_name = img_name2.replace(".png", "_N.png")
        self.save_image(self.new_name)
        print(f"{bcolors.Blue}{self.new_name}{bcolors.ENDC}")


    def b_ign(self, event):
        # img_name2 = self.img_name.replace(self.source,self.dest)
        # self.new_name = img_name2.replace(".png", "_I.png")
        # self.save_image(self.new_name)
        plt.close()
        print(f"{bcolors.Black}{self.new_name}{bcolors.ENDC}")


    def save_image(self,name):
        plt.imsave(self.new_name, np.array(self.img))
        plt.close()

if __name__ == '__main__':
    i = Index(src_img_dir, des_imag_dir)
    i.classify()
