
try:
    # ------------------ IMPORT ------------------ #
    import os
    import json
    import glob
    import argparse
    import cv2
    import numpy as np
    import scipy.misc
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import signal as sg, ndimage
    from scipy.ndimage import maximum_filter, convolve
    from PIL import Image
    from skimage.feature import peak_local_max


except ImportError:
    print("Need to fix the installation")
    raise

# ------------------ CONSTANTS ------------------ #
RED_COLOR = 0
GREEN_COLOR = 1
GRAY_COLOR = 3

PD_TABLE = pd.DataFrame()

def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    ### WRITE YOUR CODE HERE ###
    ### USE HELPER FUNCTIONS ###

    return [500, 800, 520], [500, 500, 500], [700, 710], [500, 500]


def filter_color(num: int, image: list, list_of_options: list) -> []:
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


def show_3d_filter(image1, image2, filter_array_r, filter_array_g):
    """
    Display function of the images and filters in one window for debugging and zooming
    [0] -> image
    [1] -> result after Kernel
    [2] -> image whit landmark (red/ green)
    """
    f, axarr = plt.subplots(3, 1, sharex=True, sharey=True)
    axarr[0].title.set_text('Before Kernel')
    axarr[1].title.set_text('After Kernel')
    axarr[2].title.set_text('Maximum filter')
    axarr[0].imshow(image1)
    axarr[1].imshow(image2, cmap="gray")
    axarr[2].imshow(image1, cmap="gray")


    if filter_array_r:
        t_red = np.array(filter_array_r)
        x, y = t_red.T
        axarr[2].plot(y, x, 'r.')
    if filter_array_g:
        t_green = np.array(filter_array_g)
        x, y = t_green.T
        axarr[2].plot(y, x, 'g.')


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image_path ,image, objs, fig_num=None):
    """
    1) Create kernel for red and green colors.
    2) Make convolution between the kernel to the image.
    3) Filter array of red suspicion points of traffic light.
    4) Filter array of green suspicion points of traffic light.
    4) Show 3 image: first- Origin photo, 2) Photo after Kernel 3) Maximum filter Photo with red and green points.
    :param image: matrix of image RGB
    :param objs:
    :param fig_num:
    """

    data = np.array(image)

    # convert to color:
    # image_after_convert = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    # RED
    image_after_convert_r = extract_color(RED_COLOR, data)
    url = "kernel trainer/8x8/l2.png"
    kernel = get_ker(url, RED_COLOR)
    image2 = (convolve(image_after_convert_r.astype(float), kernel[::-1, ::-1]))
    image2 = normalize_arr(image2)
    list_of_options = peak_local_max(image2, min_distance=80,threshold_abs= 126)
    # list_of_options = list(filter(lambda x: (image2[x[0]][x[1]] > 126), list_of_options))
    filter_red = filter_color(RED_COLOR, image, list_of_options)

    # GREEN
    image_after_convert_g = extract_color(GREEN_COLOR, data)
    url = "kernel trainer/8x8/l4.png"
    kernel = get_ker(url, GREEN_COLOR)
    image2 = (convolve(image_after_convert_g.astype(float), kernel[::-1, ::-1]))
    image2 = normalize_arr(image2)
    list_of_options = peak_local_max(image2, min_distance=80,threshold_abs= 126)
    # list_of_options = list(filter(lambda x: (image2[x[0]][x[1]] > 126), list_of_options))
    filter_green = filter_color(GREEN_COLOR, image, list_of_options)

    db1 = pd.DataFrame(
        {"path": image_path, "x": [x[0] for x in filter_red], "y": [y[1] for y in filter_red], "color": "r",
         "zoom": 1.00})
    db2 = pd.DataFrame(
        {"path": image_path, "x": [x[0] for x in filter_green], "y": [y[1] for y in filter_green], "color": "g",
         "zoom": 1.00})

    show_3d_filter(image, image2, filter_red, filter_green)
    plt.show()
    return pd.concat([db1, db2], ignore_index=True)
    # labels = set()
    # if objs is not None:
    #     for o in objs:
    #         poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
    #         plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
    #         labels.add(o['label'])
    #     if len(labels) > 1:
    #         plt.legend()


def get_ker(url, num):
    """
    :param url: path to the image
    :param num: What color to convert to 0->red
                                         1->green
                                         3->gray
    :return: kernel
    """
    image = Image.open(url)
    data = np.array(image)

    # convert to color:
    if num == GREEN_COLOR:
        image_after_convert = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

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


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    return show_image_and_gt(image_path, image, objects, fig_num)

    # red_x, red_y, green_x, green_y = find_tfl_lights(image)
    # plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this
    """

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "test"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    pd_table = pd.DataFrame()  # panda table arg!!!!!

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None

        # Add trafic lights of new image
        pd_table = pd.concat([pd_table, test_find_tfl_lights(image, json_fn)],ignore_index=True)
        #print("main func", pd_table)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
