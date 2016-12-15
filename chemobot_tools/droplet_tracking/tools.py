import numpy as np
from scipy.stats import norm
from scipy.ndimage import label

import cv2

WAITKEY_TIME = 1


def draw_frame_detection(frame, dish_circle, arena_circle, droplet_contours):
    # draw the dish circle
    plot_frame = frame.copy()
    cv2.circle(plot_frame, (dish_circle[0], dish_circle[1]), int(dish_circle[2]), (0, 0, 255), 3)
    cv2.circle(plot_frame, (arena_circle[0], arena_circle[1]), int(arena_circle[2]), (255, 0, 0), 3)
    cv2.drawContours(plot_frame, droplet_contours, -1, (0, 255, 0))
    return plot_frame


DEFAULT_DISH_CONFIG = {
    'minDist': np.inf,
    'hough_config': {},
    'dish_center': None,
    'dish_radius': None
}

def find_petri_dish(frame, config=DEFAULT_DISH_CONFIG, debug=False):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray scale required for HoughCircles

    # using hough searches for a big circle -> the dish
    circles = None
    dp = 0  # we increase dp till we find something
    while circles is None:
        dp += 1
        circles = cv2.HoughCircles(gray_frame, cv2.cv.CV_HOUGH_GRADIENT, dp, config['minDist'], **config['hough_config'])
        if dp > 10:
            return None, None

    dish_circle = circles[0][0]  # out in order of accumulator, first is best
    # [x, y, radius]

    # apply constraints
    if config['dish_center'] is not None:
        dish_circle[0] = config['dish_center'][0]
        dish_circle[1] = config['dish_center'][1]

    if config['dish_radius'] is not None:
        dish_circle[2] = config['dish_radius']

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
    ret, frame = video_capture.read()

    mask_shape = (frame.shape[0], frame.shape[1])

    dish_circles = []
    while ret:
        dish_circle, _ = find_petri_dish(frame, config=config)
        if dish_circle is not None:
            dish_circles.append(dish_circle)
        for _ in range(frame_spacing):
            if ret:
                ret, frame = video_capture.read()
    video_capture.release()

    median_dish_circle = np.median(dish_circles, axis=0)

    # populate the mask
    median_dish_mask = circle_to_mask(median_dish_circle, mask_shape)

    return list(median_dish_circle), median_dish_mask


def create_dish_arena(dish_circle, dish_mask, arena_dish_ratio=0.9, debug=False):
    # we define a mask for the dish
    arena_circle = list(dish_circle)
    arena_circle[2] = dish_circle[2] * arena_dish_ratio

    arena_mask = circle_to_mask(arena_circle, dish_mask.shape)

    if debug:
        # draw the mask
        cv2.imshow("arena_mask", arena_mask)

        # wait to display
        cv2.waitKey(WAITKEY_TIME)

    return arena_circle, arena_mask


def circle_to_mask(circle, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (circle[0], circle[1]), int(circle[2]), 255, -1)  # white, filled circle
    return mask


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


DEFAULT_DROPLET_HOUGH_CONFIG = {
    'minDist': 5,
    'hough_config': {
        'param1':80,
        'param2':5,
        'minRadius':5,
        'maxRadius':30
    },
    'max_detected': 20
}


def hough_droplet_detector(frame, detect_circle=None, config=DEFAULT_DROPLET_HOUGH_CONFIG, debug=False):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # using hough searches for small circles
    circles = None
    dp = 0  # we increase dp till we find something
    while circles is None:
        dp += 1
        circles = cv2.HoughCircles(gray_frame, cv2.cv.CV_HOUGH_GRADIENT, dp, config['minDist'], **config['hough_config'])

    # we keep only circle inside detect_circle and only max max_detected of them
    # circles are in order of accumulator, best first
    droplet_circles = []
    for circle in circles[0]:
        if detect_circle is None:
            droplet_circles.append(circle)
        else:
            dist_to_center = np.linalg.norm(circle[0:2] - detect_circle[0:2])
            if dist_to_center <= detect_circle[2]:
                droplet_circles.append(circle)
        if len(droplet_circles) == config['max_detected']:
            break

    # we define a mask
    droplet_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    for drop_circle in droplet_circles:
        cv2.circle(droplet_mask, (drop_circle[0], drop_circle[1]), int(drop_circle[2]), 255, -1)  # white, filled circle

    if debug:
        # draw the dish circle
        plot_frame = frame.copy()
        for drop_circle in droplet_circles:
            cv2.circle(plot_frame, (drop_circle[0], drop_circle[1]), int(drop_circle[2]), (0, 0, 255), 3)
        cv2.imshow("hough_droplet_circles", plot_frame)

        # draw the mask
        cv2.imshow("hough_droplet_circles_mask", droplet_mask)

        # wait to display
        cv2.waitKey(WAITKEY_TIME)

    return droplet_circles, droplet_mask

###
DEFAULT_DROPLET_BLOB_CONFIG = {
    'minThreshold': 10,
    'maxThreshold': 200,
    'filterByArea': True,
    'minArea': 50,
    'filterByCircularity': True,
    'minCircularity': 0.1,
    'filterByConvexity': True,
    'minConvexity': 0.8,
    'filterByInertia': True,
    'minInertiaRatio': 0.01,
}


def blob_droplet_detector(frame, detect_circle=None, config=DEFAULT_DROPLET_BLOB_CONFIG, debug=False):

    params = cv2.SimpleBlobDetector_Params()
    for k, v in DEFAULT_DROPLET_BLOB_CONFIG.items():
        setattr(params, k, v)

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    keypoints = detector.detect(frame)

    # we keep only blobs inside detect_circle and make then circles t conform with other methods
    droplet_circles = []
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        r = int(keypoint.size)
        circle = np.array([x, y, r])
        if detect_circle is None:
            droplet_circles.append(circle)
        else:
            dist_to_center = np.linalg.norm(circle[0:2] - detect_circle[0:2])
            if dist_to_center <= detect_circle[2]:
                droplet_circles.append(circle)

    # we define a mask
    droplet_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    for drop_circle in droplet_circles:
        cv2.circle(droplet_mask, (drop_circle[0], drop_circle[1]), int(drop_circle[2]), 255, -1)  # white, filled circle

    if debug:
        # draw the dish circle
        plot_frame = frame.copy()
        for drop_circle in droplet_circles:
            cv2.circle(plot_frame, (drop_circle[0], drop_circle[1]), int(drop_circle[2]), (0, 0, 255), 3)
        cv2.imshow("blob_droplet_circles", plot_frame)

        # draw the mask
        cv2.imshow("blob_droplet_circles_mask", droplet_mask)

        # wait to display
        cv2.waitKey(WAITKEY_TIME)

    return droplet_circles, droplet_mask


def watershed(frame, debug=False):
    """
    http://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    """

    dilate_kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, dilate_kernel, iterations=2)
    sure_background = cv2.dilate(opening, dilate_kernel, iterations=3)

    #
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
    return contours


def contour_to_mask(contour, frame):
    mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    cv2.fillPoly(mask, pts =[contour], color=(255))
    return mask


def contour_to_droplet_hypothesis(contour, frame, mask=None, width_ratio=1.5):
    x, y, w, h = cv2.boundingRect(contour)
    r = max(w, h) / 2.0
    square = (x+r, y+r, 2*r)
    return square_to_droplet_hypothesis(square, frame, mask, width_ratio)


def circle_to_droplet_hypothesis(circle, frame, mask=None, width_ratio=1.5):
    x  = circle[0]
    y  = circle[1]
    r  = circle[2]
    square = (x, y, 2*r)
    return square_to_droplet_hypothesis(square, frame, mask, width_ratio)


def square_to_droplet_hypothesis(square, frame, mask=None, width_ratio=1.5):
    # square is (x_center, y_center, width),
    x = square[0]
    y = square[1]
    w = square[2]
    shift = (w / 2.0) * width_ratio

    #
    square_shift = (x, y, 2.0 * shift)

    # extract the img to be later detected by the classifier
    img = frame[y-shift:y+shift, x-shift:x+shift]

    # check image valid
    if img.size == 0:
        return None, None, None

    # img must be square
    if img.shape[0] != img.shape[1]:
        return None, None, None

    if mask is None:
        # we define a mask from the contour taken using an automatic threshold method
        mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

        # we go to hsv, we want to differentiate colors from bacground that is white, we do thresholding on the saturation taken as a gray scale image
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        threshold_img = hsv_img[:, :, 1]  # we take the saturation only

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(threshold_img, (5, 5), 0)
        _, threshold_mask = cv2.threshold(blur, np.min(blur), np.max(blur), cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        mask[y-shift:y+shift, x-shift:x+shift] = threshold_mask[:, :]

    return img, mask, square_shift


def mask_droplet_hypotheses(frame, droplet_mask, width_ratio=1.5, debug=False):

    droplet_contours, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    hypotheses = []
    for contour in droplet_contours:
        # we do not let the contour_to_droplet_hypothesis function look for the mask by threshold but directly provide the mask to use
        countour_mask = contour_to_mask(contour, frame)

        img, mask, square_shift = contour_to_droplet_hypothesis(contour, frame, mask=countour_mask, width_ratio=width_ratio)

        if img is not None:
            hypotheses.append((img, mask, square_shift))

    return hypotheses


DEFAULT_CANNY_HYPOTHESIS_CONFIG = {
    'canny_config': DEFAULT_CANNY_CONFIG,
    'width_ratio': 1.5
}


def canny_droplet_hypotheses(frame, detect_mask=None, config=DEFAULT_CANNY_HYPOTHESIS_CONFIG, debug=False):

    droplet_mask = canny_droplet_detector(frame, mask=detect_mask, config=config['canny_config'], debug=debug)

    hypotheses = mask_droplet_hypotheses(frame, droplet_mask, width_ratio=config['width_ratio'], debug=debug)

    return hypotheses


DEFAULT_DROPLET_HOUGH_HYPOTHESIS_CONFIG = {
    'hough_config': DEFAULT_DROPLET_HOUGH_CONFIG,
    'width_ratio': 1.5
}


def hough_droplet_hypotheses(frame, detect_circle=None, config=DEFAULT_DROPLET_HOUGH_HYPOTHESIS_CONFIG, debug=False):

    droplet_circles, _ = hough_droplet_detector(frame, detect_circle=detect_circle, config=config['hough_config'], debug=debug)

    hypotheses = []
    for circle in droplet_circles:
        img, mask, square_shift = circle_to_droplet_hypothesis(circle, frame, mask=None, width_ratio=config['width_ratio'])

        if img is not None:
            hypotheses.append((img, mask, square_shift))

    return hypotheses


DEFAULT_DROPLET_BLOB_HYPOTHESIS_CONFIG = {
    'blob_config': DEFAULT_DROPLET_BLOB_CONFIG,
    'width_ratio': 1.5
}

def blob_droplet_hypotheses(frame, detect_circle=None, config=DEFAULT_DROPLET_BLOB_HYPOTHESIS_CONFIG, debug=False):

    droplet_circles, _ = blob_droplet_detector(frame, detect_circle=detect_circle, config=config['blob_config'], debug=debug)

    hypotheses = []
    for circle in droplet_circles:
        img, mask, square_shift = circle_to_droplet_hypothesis(circle, frame, mask=None, width_ratio=config['width_ratio'])

        if img is not None:
            hypotheses.append((img, mask, square_shift))

    return hypotheses


from ..droplet_classification import DEFAULT_DROPLET_CLASSIFIER

DEFAULT_HYPOTHESIS_CONFIG = {
    'droplet_classifier': DEFAULT_DROPLET_CLASSIFIER,
    'class_name': 'droplet'
}


def hypotheses_to_droplet_contours(frame, hypotheses, config=DEFAULT_HYPOTHESIS_CONFIG, debug=False):

        #read config
        droplet_classifier = config['droplet_classifier']
        class_name = config['class_name']

        # prepare mask
        droplet_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

        hypothesis_frame = frame.copy()
        for (img, mask, square) in hypotheses:
            #
            cls_name, cls_proba = droplet_classifier.predict_img(img)
            if cls_name == class_name:
                droplet_mask = cv2.bitwise_or(droplet_mask, mask)

            if debug:
                if cls_name == class_name:
                    color = (0,255,0)  # Green
                else:
                    color = (0,0,255)  # Red
                # draw the dish circle
                shift = square[2] / 2.0
                x1 = int(square[0] - shift)
                y1 = int(square[1] - shift)
                x2 = int(square[0] + shift)
                y2 = int(square[1] + shift)
                cv2.rectangle(hypothesis_frame, (x1, y1), (x2, y2), color, 1)


        if debug:
            cv2.imshow('hypotheses', hypothesis_frame)
            # show the mask before contour function plays with it
            cv2.imshow('hypotheses_droplet_mask', droplet_mask)

        # contours
        droplet_contours, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if debug:
            contour_frame = frame.copy()
            cv2.drawContours(contour_frame, droplet_contours, -1, (0, 255, 0))
            cv2.imshow('hypotheses_droplet_contours', contour_frame)

            # wait to display
            cv2.waitKey(WAITKEY_TIME)

        return droplet_contours


def clean_contours_outside_arena(droplet_contours, arena_circle):

    valid_contours = []

    arena_center = np.array(arena_circle[0:2])
    arena_radius = arena_circle[2]

    for contour in droplet_contours:

        contour_center = None
        # error are usually raised for too small contours
        # they will be discared
        try:
            M = cv2.moments(contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            contour_center = np.array([centroid_x, centroid_y])
        except:
            continue

        if contour_center is not None:
            # if centroid within arena add to the valid contour
            delta_center = arena_center - contour_center
            dist_center = np.linalg.norm(delta_center)
            if dist_center < arena_radius:
                valid_contours.append(contour)

    return valid_contours


def binarize_frame(frame, threshold, mask=None, backgroud_intensity=None, debug=False):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blured_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    if mask is not None:
        #foreground
        foreground = cv2.bitwise_or(blured_frame, blured_frame, mask=mask)
        # get second masked value (background) mask must be inverted
        if backgroud_intensity is None:
            backgroud_intensity = np.median(blured_frame)
        mask_inv = cv2.bitwise_not(mask)
        background = np.full(blured_frame.shape, backgroud_intensity, dtype=np.uint8)
        background = cv2.bitwise_or(background, background, mask=mask_inv)

        blured_frame = cv2.bitwise_or(foreground, background)

    _, droplet_mask = cv2.threshold(blured_frame, threshold, 255, cv2.THRESH_BINARY_INV)

    if debug:
        cv2.imshow('video', frame)
        cv2.waitKey(1)

        cv2.imshow('processed', blured_frame)
        cv2.waitKey(1)

        cv2.imshow('droplet_mask', droplet_mask)
        cv2.waitKey(1)

    return droplet_mask


DEFAULT_BINARIZATION_THRESHOLD_CONFIG = {
    'cut_proba': 0.001,
    'spectrum_filename': None
}


def compute_frame_binarization_threshold(gray_frame, mask=None, config=DEFAULT_BINARIZATION_THRESHOLD_CONFIG):
    # we compute the histogram of intensities within the masked region
    # this should be representative of the background gray
    # we cpmpute the mean and variance of this average histogram
    # we define a threshold as when the left tail account for cut_proba of the Gaussian
    #
    if mask is not None:
        mask_bool = np.array(mask, dtype='bool')
        to_process_frame = gray_frame[mask_bool].ravel()
    else:
        to_process_frame = gray_frame.ravel()

    # compute mean and std of histogram
    hist_data, bins = np.histogram(to_process_frame, np.arange(256))
    positions = bins[:-1] + np.diff(bins)/2.0
    mu = np.average(positions, weights=hist_data)
    sigma = np.sqrt(np.average((positions-mu)**2, weights=hist_data))
    threshold = int(round(norm.ppf(config['cut_proba'], mu, sigma)))

    # plot if requested
    if 'spectrum_filename' in config:
        if config['spectrum_filename'] is not None:

            import matplotlib
            import matplotlib.pyplot as plt

            # design figure
            fontsize = 22
            matplotlib.rc('xtick', labelsize=20)
            matplotlib.rc('ytick', labelsize=20)
            matplotlib.rcParams.update({'font.size': fontsize})

            ##
            fig = plt.figure(figsize=(12,8))
            plt.plot(positions, hist_data, 'b')

            y_pdf = norm.pdf(positions, mu, sigma)
            y_pdf = y_pdf / np.max(y_pdf) * np.max(hist_data)
            plt.plot(positions, y_pdf, 'r--', linewidth=2)

            plt.plot([threshold, threshold], [0, np.max(hist_data)], 'g')

            plt.legend(['Mean Histogram', 'Gaussian fit', 'Threshold: {}'.format(threshold)], loc='best', fontsize=22)
            plt.xlabel('Intensity', fontsize=22)
            plt.ylabel('Pixel Count', fontsize=22)

            plt.savefig(config['spectrum_filename'], dpi=100)
            plt.close(fig)

    return threshold, mu, sigma


def compute_avg_gray_frame(video_filename):
    # we compute the average gray scale image across the all video
    # this smooth out the droplet and leave the background
    video_capture = cv2.VideoCapture(video_filename)
    ret, frame = video_capture.read()

    sum_frame = np.zeros((frame.shape[0], frame.shape[1]))
    frame_count = 0
    while ret:
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sum_frame = sum_frame + gray_frame
        ret, frame = video_capture.read()
    video_capture.release()

    return np.array(sum_frame/float(frame_count), dtype='uint8')


def compute_video_binarization_threshold(video_filename, mask=None, config=DEFAULT_BINARIZATION_THRESHOLD_CONFIG):
    avg_gray_frame = compute_avg_gray_frame(video_filename)
    return compute_frame_binarization_threshold(avg_gray_frame, mask=mask, config=config)
