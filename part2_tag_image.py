import cv2
import glob
from PIL import Image
import pandas as pd
import glob
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import macros
import matplotlib.patches as patches


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
    def __init__(self, source_path, file_path, x, y, pd):
        self.pd = pd
        self.source_path = source_path
        self.source = Image.open(source_path)
        self.cropped_image_path = file_path
        self.cropped_image =Image.open(file_path)
        self.img_name = ''
        self.new_name = ''
        self.x = x
        self.y = y


    def classify(self):
        print(self.cropped_image_path)
        f, axarr = plt.subplots(1, 2)
        axarr[0].title.set_text('image')
        axarr[1].title.set_text('crop')
        axarr[0].imshow(self.source)
        axarr[1].imshow(self.cropped_image)

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

        rect = patches.Rectangle((self.y - 50, self.x - 50), 100, 100, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        axarr[0].add_patch(rect)


        plt.show()
        self.img_name = ''
        self.new_name = ''

    def inset_crop_to_panda_table(self, color):
        # print(self.x, self.y, self.source_path)
        new_data_frame = pd.DataFrame({"full_path": self.source_path, "crop_path": self.cropped_image_path, "x": [self.x], "y": [self.x], "color": color, "zoom": 1.00})
        # print(new_data_frame)
        self.pd[0] = pd.concat([self.pd[0], new_data_frame], ignore_index=True)

    def b_green(self, event):
        self.inset_crop_to_panda_table("G")
        self.new_name = self.cropped_image_path
        self.new_name = self.new_name.replace("cropped images\data", macros.GREEN_LIGHTS_DIR_PATH)

        # self.new_name = img_name2.replace(".png", "_G.png")
        self.save_image(self.new_name)
        print(f"{bcolors.Green}{self.new_name}{bcolors.ENDC}")



    def b_red(self, event):
        self.inset_crop_to_panda_table("R")
        self.new_name = self.cropped_image_path
        self.new_name = self.new_name.replace("cropped images\data", macros.RED_LIGHTS_DIR_PATH)


        # self.new_name = img_name2.replace(".png", "_R.png")
        self.save_image(self.new_name)
        print(f"{bcolors.Red}{self.new_name}{bcolors.ENDC}")


    def b_not(self, event):
        self.inset_crop_to_panda_table("N")
        self.new_name = self.cropped_image_path
        self.new_name = self.new_name.replace("cropped images\data", macros.NOT_TRAFFIC_LIGHTS_DIR_PATH)

        # self.new_name = img_name2.replace(".png", "_N.png")
        self.save_image(self.new_name)
        print(f"{bcolors.Blue}{self.new_name}{bcolors.ENDC}")


    def b_ign(self, event):
        # img_name2 = self.img_name.replace(self.source,self.dest)
        # self.new_name = img_name2.replace(".png", "_I.png")
        # self.save_image(self.new_name)
        plt.close()
        print(f"{bcolors.Black}{self.cropped_image}{bcolors.ENDC}")


    def save_image(self,name):
        plt.imsave(self.new_name, np.array(self.cropped_image))
        plt.close()


if __name__ == '__main__':
    pd_table = [pd.DataFrame()]
    for i,row in pd.read_csv(macros.CROP_INFO_CSV_PATH).iterrows():
        print("i: ",i,"\n", row["Source"])
        source_path = row["Source"]
        file_path = row["File name"]
        x = row["X"]
        y = row["Y"]
        if i == 4:
            break
        index = Index(source_path, file_path, x, y, pd_table)
        index.classify()

    print("end")
    print("end")
    pd_table[0].to_csv(macros.LABELS_INFO_PATH, mode="a", index=False, header=False)

