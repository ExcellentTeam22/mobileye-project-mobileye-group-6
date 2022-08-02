
try:
    import os
    import json
    import glob
    import argparse
    import cv2

    import numpy as np
    from scipy import signal as sg, ndimage
    from scipy.ndimage import maximum_filter, convolve
    import scipy.misc
    from skimage.feature import peak_local_max
    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


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


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):

    data = np.array(image)

    #convert to color:
    grayImage = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    url = "kernel trainer/8x8/l2.png"
    kernel = get_ker(url)

    image2 = (convolve(grayImage.astype(float), kernel[::-1, ::-1]))
    image2 = normalize_arr(image2)
    print("-" * 20 , " \n image2 : \n" ,"-" * 20)

    list_of_options = peak_local_max(image2, min_distance=100)
    print("-" * 20, " \n peak_local_max:\n ", "-" * 20)
    print(list_of_options, len(list_of_options))


    f, axarr = plt.subplots(3, 1, sharex=True, sharey=True)
    axarr[0].title.set_text('Before Kernel')
    axarr[1].title.set_text('After Kernel')
    axarr[2].title.set_text('Maximum filter')
    axarr[0].imshow(image)
    axarr[1].imshow(image2, cmap="gray")
    axarr[2].imshow(image, cmap="gray")
    axarr[2].plot(list_of_options[:, 1], list_of_options[:, 0], 'r.')

    # # to get all the optional pixels
    # temp_to_zip = np.where(np.logical_and(result > (np.max(result) - (np.average(result) * 0.05)), result <= 255))
    # list_of_options = list(zip(temp_to_zip[0],temp_to_zip[1]))
    # print("-"*20," \n list of point:\n ", "-"*20)
    # print(len(list_of_options),list_of_options)



    plt.show()


    # labels = set()
    # if objs is not None:
    #     for o in objs:
    #         poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
    #         plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
    #         labels.add(o['label'])
    #     if len(labels) > 1:
    #         plt.legend()


def get_ker(url):

    image = Image.open(url)
    data = np.array(image)

    #converv to color:
    grayImage = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    grayImage = grayImage - (np.sum(grayImage) / grayImage.size)
    return grayImage


def extract_color(num: int, numpy_array: np.array):
    return numpy_array[:, :, num].copy()


def normalize_arr(arr):
    return (255*(arr - np.min(arr))/np.ptp(arr)).astype(int)

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

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "test"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()