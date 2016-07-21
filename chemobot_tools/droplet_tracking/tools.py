import numpy as np
from scipy.ndimage import label

import cv2

WAITKEY_TIME = 1


DEFAULT_DISH_CONFIG = {
    'minDist': np.inf,
    'hough_config': {}
}


def find_petri_dish(frame, config=DEFAULT_DISH_CONFIG, debug=False):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray scale required for HoughCircles

    # using hough searches for a big circle -> the dish
    circles = None
    dp = 0  # we increase dp till we find something
    while circles is None:
        dp += 1
        circles = cv2.HoughCircles(gray_frame, cv2.cv.CV_HOUGH_GRADIENT, dp, config['minDist'], **config['hough_config'])

    dish_circle = circles[0][0]  # out in order of accumulator, first is best
    # [x, y, radius]

    # we define a mask for the dish
    dish_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    cv2.circle(dish_mask, (dish_circle[0], dish_circle[1]), int(dish_circle[2]), 255, -1)  # white, filled circle

    if debug:
        # draw the dish circle
        plot_frame = frame.copy()
        cv2.circle(plot_frame, (dish_circle[0], dish_circle[1]), int(dish_circle[2]), (0, 0, 255), 3)
        cv2.imshow("dish_circle", plot_frame)

        # draw the mask
        cv2.imshow("dish_mask", dish_mask)

        # wait to display
        cv2.waitKey(WAITKEY_TIME)

    return dish_circle, dish_mask


def get_median_dish_from_video(video_filename, config=DEFAULT_DISH_CONFIG, frame_spacing=100):
    # open video to play with frames
    video_capture = cv2.VideoCapture(video_filename)

    dish_circles = []

    ret, frame = video_capture.read()

    # we define an empty mask for the dish
    median_dish_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

    while ret:
        dish_circle, _ = find_petri_dish(frame, config=config)
        dish_circles.append(dish_circle)
        for _ in range(frame_spacing):
            if ret:
                ret, frame = video_capture.read()
    video_capture.release()

    median_dish_circle = np.median(dish_circles, axis=0)

    # populated the mask
    cv2.circle(median_dish_mask, (median_dish_circle[0], median_dish_circle[1]), int(median_dish_circle[2]), 255, -1)  # white, filled circle

    return median_dish_circle, median_dish_mask


def create_dish_arena(dish_circle, dish_mask, arena_dish_ratio=0.9, debug=False):
    # we define a mask for the dish
    arena_circle = list(dish_circle)
    arena_circle[2] = dish_circle[2] * arena_dish_ratio

    arena_mask = np.zeros(dish_mask.shape, np.uint8)
    cv2.circle(arena_mask, (arena_circle[0], arena_circle[1]), int(arena_circle[2]), 255, -1)  # white, filled circle

    if debug:
        # draw the mask
        cv2.imshow("arena_mask", arena_mask)

        # wait to display
        cv2.waitKey(WAITKEY_TIME)

    return arena_circle, arena_mask


DEFAULT_CANNY_CONFIG = {
    'blur_kernel_wh': (5, 5),
    'blur_kernel_sigma': 0,
    'dilate_kernel_wh': (2, 2),
    'canny_lower': 30,
    'canny_upper': 60,
    'noise_kernel_wh': (3, 3)
}


def canny_droplet_detector(frame, mask=None, config=DEFAULT_CANNY_CONFIG, debug=False):
    """
    First use canny to find edges.
    Then fill the image from 0,0.
    Everything that is not filled is a droplet, because the droplets should be the only fully closed objects.
    """

    # First use canny to find edges
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray scale
    blur_frame = cv2.GaussianBlur(gray_frame, config['blur_kernel_wh'], config['blur_kernel_sigma'])  # blur it
    canny_frame = cv2.Canny(blur_frame, config['canny_lower'], config['canny_upper'])

    # dilate edges for circles that may be broken after canny
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['dilate_kernel_wh'])
    canny_dilate = cv2.dilate(canny_frame, kernel)

    # keep only what is in the mask
    if mask is not None:
        canny_dilate[mask == 0] = 0  # what has been masked becomes black

    # segmentation based on inner part
    flood_fill = canny_dilate.copy()
    height, width = flood_fill.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood_fill, flood_mask, (0, 0), 255)  # starts at (0, 0)

    # remove noise
    ret, thresh = cv2.threshold(flood_fill, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones(config['noise_kernel_wh'], np.uint8)
    denoised_flood_fill = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    if debug:
        cv2.imshow("canny_frame", canny_frame)
        cv2.imshow("canny_dilate", canny_dilate)
        cv2.imshow("flood_fill", flood_fill)
        cv2.imshow("denoised_flood_fill", denoised_flood_fill)

        # wait to display
        cv2.waitKey(WAITKEY_TIME)

    return denoised_flood_fill


def watershed(frame, debug=False):
    """
    http://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    """

    dilate_kernel = (5, 5)
    sure_background = cv2.dilate(frame, dilate_kernel)

    dist_transform = cv2.distanceTransform(frame, cv2.cv.CV_DIST_L2, maskSize=3)

    dist_transform -= dist_transform.min()
    if dist_transform.max() > 0:
        dist_transform /= float(dist_transform.max())
    dist_transform = np.uint8(255 * dist_transform)

    threshold_lower = 150
    threshold_upper = 255
    _, sure_foreground = cv2.threshold(dist_transform, threshold_lower, threshold_upper, cv2.THRESH_BINARY)
    sure_foreground = np.uint8(sure_foreground)

    unknown = cv2.subtract(sure_background, sure_foreground)

    markers, num_markers = label(sure_foreground)
    markers += 1
    markers[unknown == 255] = 0

    markers = markers.astype(np.int32)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    cv2.watershed(frame_rgb, markers)

    contour_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    contour_mask[markers == -1] = 255

    dilate_kernel_wh = (2, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_kernel_wh)
    contour_mask = cv2.dilate(contour_mask, kernel)

    # segmentation based on inner part
    height, width = contour_mask.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(contour_mask, flood_mask, dilate_kernel_wh, 255)  # starts at dilate_kernel_wh because the watershed + dilate should have find a boundary all around the screen
    # revert it now
    watershed_island = 255 - contour_mask

    if debug:
        cv2.imshow("watershed_input", frame)
        cv2.imshow("sure_background", sure_background)
        cv2.imshow("dist_transform", dist_transform)
        cv2.imshow("sure_foreground", sure_foreground)
        cv2.imshow("unknown", unknown)
        cv2.imshow("contour_mask", contour_mask)
        cv2.imshow("watershed_island", watershed_island)

        # wait to display
        cv2.waitKey(WAITKEY_TIME)

    return watershed_island


def contours_to_list(contours):
    countours_list = []
    for contour in contours:
        countours_list.append(contour.tolist())
    return countours_list


def list_to_contours(countours_list):
    contours = []
    for countour_list in countours_list:
        contours.append(np.array(countour_list, dtype=np.int32))
    return countours_list
