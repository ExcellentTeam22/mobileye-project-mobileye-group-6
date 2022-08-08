try:
    # ------------------ IMPORT ------------------ #
    import os
    import json
    import glob
    import argparse
    import cv2
    import numpy as np
    import scipy.misc
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from typing import List
    from os import walk
    from pathlib import Path
    import pandas as pd
    from scipy import signal as sg, ndimage
    from scipy.ndimage import maximum_filter, convolve
    from PIL import Image
    from skimage.feature import peak_local_max

    import macros
except ImportError:
    print("Need to fix the installation")
    raise

def filter_color(num: int, image: list, list_of_options: list) -> List:
    """
    The function receives a list of coordinates and an image and returns
    only the coordinates that are within the range of the color
    (red = 0, green = 1)
    """
    filter_color_list = []
    for index in list_of_options:
        if (num == 0 and image[index[0]][index[1]][0] > 127) and (
                image[index[0]][index[1]][1] < image[index[0]][index[1]][0]) and (
                image[index[0]][index[1]][2] < image[index[0]][index[1]][0]):
            filter_color_list.append([index[0], index[1]])
        if (num == 1 and image[index[0]][index[1]][1] > 166) and (
                image[index[0]][index[1]][0] < image[index[0]][index[1]][1]) and (
                image[index[0]][index[1]][2] < image[index[0]][index[1]][1]):
            filter_color_list.append([index[0], index[1]])
    return filter_color_list


def design_figure(figure, axarr, image, red_kernel, green_kernel):
    """

    :param axarr:
    :param image:
    :param red_kernel:
    :param green_kernel:
    :return:
    """
    axarr[0][1].title.set_text('Original')
    axarr[0][1].imshow(image)

    axarr[0][0].title.set_text('Red Kernel')
    axarr[0][0].imshow(red_kernel, cmap="gray")

    axarr[0][2].imshow(green_kernel, cmap="gray")
    axarr[0][2].title.set_text('Green Kernel')

    axarr[1][2].imshow(image, cmap="gray")
    axarr[1][2].title.set_text('Green Dots')
    axarr[1][0].imshow(image, cmap="gray")
    axarr[1][0].title.set_text('Red Dots')

    axarr[2][0].imshow(image)
    axarr[2][0].title.set_text('Red Crops')
    axarr[2][2].imshow(image, cmap="gray")
    axarr[2][2].title.set_text('Image')
    axarr[2][1].imshow(image)
    axarr[2][2].title.set_text('Green Crops')
    figure.delaxes(axarr[1][1])


def add_image_info(filter_array, dot_color, rect_color, dots_ax, crops_ax):
    if filter_array:
        t = np.array(filter_array)
        x, y = t.T
        dots_ax.plot(y, x, dot_color)
        for dot_x, dot_y in zip(x, y):
            rec = patches.Rectangle((dot_y - 50, dot_x - 50), 2 * 50, 2 * 50, linewidth=1,
                                    edgecolor=rect_color, facecolor='none')
            crops_ax.add_patch(rec)


def add_crops(filter_array, image, image_path, up, down, sides):
    data = {"File name": [], "X": [], "Y": [], "Source": []}
    if filter_array:
        t = np.array(filter_array)
        dots_x, dots_y = t.T
        index = len(next(walk(macros.CROPPED_IMAGES_DIR_PATH), (None, None, []))[2])
        for x, y in zip(dots_x, dots_y):
            cropped = image[max(0, x - up):min(np.size(image, 0), x + down), max(0, y - sides):min(np.size(image, 1),
                                                                                                   y + sides)]
            image_before_convert = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

            image_after_convert = Image.fromarray(image_before_convert)

            image_after_convert = image_after_convert.resize((30, 40))

            image_after_convert = np.array(image_after_convert)

            cv2.imwrite(f"{macros.CROPPED_IMAGES_DIR_PATH}\\{index}.png", image_after_convert)
            data["File name"].append(f"{macros.CROPPED_IMAGES_DIR_PATH}\\{index}.png")
            data["X"].append(f"{x}")
            data["Y"].append(f"{y}")
            data["Source"].append(image_path)
            index += 1
    if data:
        df = pd.DataFrame(data)
        df.to_csv(macros.CROP_INFO_CSV_PATH, mode="a", index=False, header=False)


def show_3d_filter(image, red_kernel, green_kernel, filter_array_r, filter_array_g, image_path):
    """
    Display function of the images and filters in one window for debugging and zooming
    [0] -> image
    [1] -> result after Kernel
    [2] -> image whit landmark (red/ green)
    """
    f, axarr = plt.subplots(3, 3, sharex=True, sharey=True)
    design_figure(f, axarr, image, red_kernel, green_kernel)
    add_image_info(filter_array_r, "r.", "red", axarr[1][0], axarr[2][0])
    add_image_info(filter_array_g, "g.", "green", axarr[1][2], axarr[2][2])
    plt.show()


def show_image_and_gt(image_path, image, show=False):
    """
        1) Create kernels for red and green colors.
    2) Make convolution between the kernels to the image.
    3) Filter array of red suspicion points of traffic light.
    4) Filter array of green suspicion points of traffic light.
    4) Show 3 image: first- Origin photo, 2) Photo after Kernel 3) Maximum filter Photo with red and green points.
    :param image_path:
    :param image: matrix of image RGB
    :param show:
    :return:
    """
    data = np.array(image)

    kernel_37 = get_ker(macros.KERNEL_PATH_37, macros.GRAY_COLOR_CODE)
    kernel_11 = get_ker(macros.KERNEL_PATH_11, macros.GRAY_COLOR_CODE)

    # convert to color:
    image_after_convert = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    # 11
    kernel_img_11 = (convolve(image_after_convert.astype(float), kernel_11[::-1, ::-1]))
    kernel_img_11 = normalize_arr(kernel_img_11)
    list_of_options_11 = peak_local_max(kernel_img_11, min_distance=100, threshold_abs=150)
    filter_11_r = filter_color(macros.RED_COLOR_CODE, image, list_of_options_11)
    filter_11_g = filter_color(macros.GREEN_COLOR_CODE, image, list_of_options_11)

    # 37
    kernel_img_37 = (convolve(image_after_convert.astype(float), kernel_37[::-1, ::-1]))
    kernel_img_37 = normalize_arr(kernel_img_37)
    list_of_options_37 = peak_local_max(kernel_img_37, min_distance=80, threshold_abs=150)
    filter_37_g = filter_color(macros.GREEN_COLOR_CODE, image, list_of_options_37)
    filter_37_r = filter_color(macros.RED_COLOR_CODE, image, list_of_options_37)

    filter_green = filter_11_g + filter_37_g
    filter_red = filter_11_r + filter_37_r
    add_crops(filter_11_r, image, image_path, sides=4, down=20, up=4)
    add_crops(filter_11_g, image, image_path, sides=4, down=4, up=20)
    add_crops(filter_37_r, image, image_path,sides=15 ,down=81, up=15)
    add_crops(filter_37_g, image, image_path, sides=15, down=15, up=81)

    if show:
        show_3d_filter(image, kernel_img_37, kernel_img_11, filter_red, filter_green, image_path)


def get_ker(url, num):
    """
    :param url: path to the image
    :param num: What color to convert to 0->red
                                         1->green
                                         3->gray
    :return: kernels
    """
    image = Image.open(url)
    data = np.array(image)

    # convert to color:
    if num == macros.GRAY_COLOR_CODE:
        image_after_convert = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    else:
        image_after_convert = extract_color(num, data)
    image_after_convert = image_after_convert - (np.sum(image_after_convert) / image_after_convert.size)
    return image_after_convert


def extract_color(num: int, numpy_array: np.array):
    """
    Extract matrix with the wanted color from RGB metrix
    :param num: num of color red, green or blue
    :param numpy_array: data of image
    :return: matrix of the wanted color
    """
    return numpy_array[:, :, num].copy()


def normalize_arr(arr):
    """
    Normalize array to 0 - 255 val
    :param arr: given array
    :return: normalize array
    """
    return (255 * (arr - np.min(arr)) / np.ptp(arr)).astype(int)


def test_find_tfl_lights(image_path, fig_num=None, show=False):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    return show_image_and_gt(image_path, image, show)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this
    """
    show = True
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    if args.dir is None:
        args.dir = macros.TESTS_PICS_DIR_PATH
        show = False
    flist = [x for x in Path(macros.TESTS_PICS_DIR_PATH).rglob('*.png')]
    for i, image in enumerate(flist):
        if i % macros.NUM_OF_USERS == macros.USER_ID:
            test_find_tfl_lights(str(image), show=show)


if __name__ == '__main__':
    main()
