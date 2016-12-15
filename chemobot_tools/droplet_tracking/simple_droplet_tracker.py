import os
import json
import time

import cv2

import tools

WAITKEY_TIME = 1

DEFAULT_FRAME_CONFIG = {
    'dish_detect_config': tools.DEFAULT_DISH_CONFIG,
    'arena_ratio': 0.8,
    'binarization_threshold_config': tools.DEFAULT_BINARIZATION_THRESHOLD_CONFIG
}


def detect_droplet_frame(frame, config=DEFAULT_FRAME_CONFIG, debug=True, deep_debug=False):

    # dish detection
    dish_circle, dish_mask = tools.find_petri_dish(frame, config=config['dish_detect_config'], debug=deep_debug)

    arena_circle, arena_mask = tools.create_dish_arena(dish_circle, dish_mask, config['arena_ratio'])

    # threshold
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold, _, _ = tools.compute_frame_binarization_threshold(gray_frame, mask=dish_mask, config=config['binarization_threshold_config'])

    # find mask an countour
    droplet_mask = tools.binarize_frame(frame, threshold, mask=dish_mask, debug=deep_debug)

    droplet_contours, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # removing non valid contour outside arena
    valid_droplet_contours = tools.clean_contours_outside_arena(droplet_contours, arena_circle)

    if debug:
        plot_frame = tools.draw_frame_detection(frame, dish_circle, arena_circle, valid_droplet_contours)
        cv2.imshow("droplet_detection", plot_frame)
        cv2.waitKey(WAITKEY_TIME)

    return valid_droplet_contours



DEFAULT_PROCESS_CONFIG = {
    'dish_detect_config': tools.DEFAULT_DISH_CONFIG,
    'dish_frame_spacing': 20,
    'arena_ratio': 0.8,
    'binarization_threshold_config': tools.DEFAULT_BINARIZATION_THRESHOLD_CONFIG
}


def process_video(video_filename, process_config=DEFAULT_PROCESS_CONFIG, video_out=None, droplet_info_out=None, dish_info_out=None, debug=False, deep_debug=False, verbose=False, debug_window_name='droplet_detection'):

    start_time = time.time()

    if verbose:
        print '###\nProcessing video {} ...'.format(video_filename)

    droplet_info = []
    droplet_info_list = []

    # dish detection accross frame
    dish_circle, dish_mask = tools.get_median_dish_from_video(video_filename, process_config['dish_detect_config'], frame_spacing=process_config['dish_frame_spacing'])

    arena_circle, arena_mask = tools.create_dish_arena(dish_circle, dish_mask, process_config['arena_ratio'])

    # save dish info
    if dish_info_out is not None:
        dish_info = {}
        dish_info['dish_circle'] = [float(v) for v in dish_circle]
        dish_info['arena_circle'] = [float(v) for v in arena_circle]
        with open(dish_info_out, 'w') as f:
            json.dump(dish_info, f)

    # compute bonarization threshold based on full video
    threshold, mu, sigma = tools.compute_video_binarization_threshold(video_filename, dish_mask, process_config['binarization_threshold_config'])

    # open video to play with frames
    video_capture = cv2.VideoCapture(video_filename)
    ret, frame = video_capture.read()

    # creating the writer
    if video_out is not None:
        video_format = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
        fps = 20
        video_writer = cv2.VideoWriter(video_out, video_format, fps, (frame.shape[1], frame.shape[0]))

    frame_count = 0
    while ret:
        frame_count += 1

        #detect dark region below threshold
        # we use dish mask here and then we trim the contour to arena_mask
        droplet_mask = tools.binarize_frame(frame, threshold, mask=dish_mask, backgroud_intensity=mu, debug=deep_debug)
        #extract contours
        droplet_contours, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # removing non valid contour outside arena
        valid_droplet_contours = tools.clean_contours_outside_arena(droplet_contours, arena_circle)

        # storing
        droplet_info.append(valid_droplet_contours)
        droplet_info_list.append(tools.contours_to_list(valid_droplet_contours))

        # plot
        if debug or video_out is not None:

            plot_frame = tools.draw_frame_detection(frame, dish_circle, arena_circle, valid_droplet_contours)

            if video_out is not None:
                video_writer.write(plot_frame)

            if debug:
                cv2.imshow(debug_window_name, plot_frame)
                cv2.waitKey(WAITKEY_TIME)

        if deep_debug:
            plot_frame_droplet_contours = tools.draw_frame_detection(frame, dish_circle, arena_circle, droplet_contours)
            cv2.imshow('raw_droplet_contours', plot_frame_droplet_contours)
            cv2.waitKey(WAITKEY_TIME)

        # next frame
        ret, frame = video_capture.read()

    video_capture.release()
    if video_out is not None:
        video_writer.release()

    if droplet_info_out is not None:
        with open(droplet_info_out, 'w') as f:
            json.dump(droplet_info_list, f)

    if debug:
         cv2.destroyWindow(debug_window_name)
         for _ in range(10):  # ensure it kills the window
             cv2.waitKey(WAITKEY_TIME)

    elapsed = time.time() - start_time

    if verbose:
        print '###\nFinished processing video {} in {} seconds.'.format(video_filename, elapsed)

    return droplet_info
