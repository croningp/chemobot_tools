import os
import json

import cv2

import tools

WAITKEY_TIME = 1


DEFAULT_FRAME_CONFIG = {
    'dish_detect_config': tools.DEFAULT_DISH_CONFIG,
    'arena_ratio': 0.85,
    'canny_hypothesis_config': tools.DEFAULT_CANNY_HYPOTHESIS_CONFIG,
    'hough_hypothesis_config': tools.DEFAULT_DROPLET_HOUGH_HYPOTHESIS_CONFIG
}


def detect_droplet_frame(frame, droplet_classifier, config=DEFAULT_FRAME_CONFIG, class_name='droplet', debug=False, deep_debug=False):

    # dish detection
    dish_circle, dish_mask = tools.find_petri_dish(frame, config=config['dish_detect_config'], debug=deep_debug)

    arena_circle, arena_mask = tools.create_dish_arena(dish_circle, dish_mask, config['arena_ratio'])

    # hypothesis making
    hypotheses = tools.canny_droplet_hypotheses(frame, detect_mask=arena_mask, config=config['canny_hypothesis_config'])
    hypotheses += tools.hough_droplet_hypotheses(frame, detect_circle=arena_circle, config=config['hough_hypothesis_config'])

    # hypothesis solving
    droplet_contours = tools.hypotheses_to_droplet_contours(frame, hypotheses, droplet_classifier, class_name=class_name, debug=deep_debug)

    if debug:
        plot_frame = draw_frame_detection(frame, dish_circle, arena_circle, droplet_contours)
        cv2.imshow("droplet_detection", plot_frame)
        cv2.waitKey(WAITKEY_TIME)

    return droplet_contours


def draw_frame_detection(frame, dish_circle, arena_circle, droplet_contours):
    # draw the dish circle
    plot_frame = frame.copy()
    cv2.circle(plot_frame, (dish_circle[0], dish_circle[1]), int(dish_circle[2]), (0, 0, 255), 3)
    cv2.circle(plot_frame, (arena_circle[0], arena_circle[1]), int(arena_circle[2]), (255, 0, 0), 3)
    cv2.drawContours(plot_frame, droplet_contours, -1, (0, 255, 0))
    return plot_frame


DEFAULT_PROCESS_CONFIG = {
    'dish_detect_config': tools.DEFAULT_DISH_CONFIG,
    'dish_frame_spacing': 100,
    'arena_ratio': 0.85,
    'canny_hypothesis_config': tools.DEFAULT_CANNY_HYPOTHESIS_CONFIG,
    'hough_hypothesis_config': tools.DEFAULT_DROPLET_HOUGH_HYPOTHESIS_CONFIG,
    'mog_hypothesis_config': {
        'learning_rate': 0.005,
        'delay_by_n_frame': 50,
        'width_ratio': 1.5
    },
    'resolve_hypothesis_config': tools.DEFAULT_HYPOTHESIS_CONFIG
}


def process_video(video_filename, process_config=DEFAULT_PROCESS_CONFIG, video_out=None, droplet_info_out=None, dish_info_out=None, debug=False, deep_debug=False, verbose=False, debug_window_name='droplet_detection'):

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

    # open video to play with frames
    video_capture = cv2.VideoCapture(video_filename)
    ret, frame = video_capture.read()

    # creating the writer
    if video_out is not None:
        video_format = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
        fps = 20
        video_writer = cv2.VideoWriter(video_out, video_format, fps, (frame.shape[1], frame.shape[0]))

    # prepare BackgroundSubtractorMOG
    backsub = cv2.BackgroundSubtractorMOG()
    backsub_learning_rate = process_config['mog_hypothesis_config']['learning_rate']
    backsub_delay = process_config['mog_hypothesis_config']['delay_by_n_frame']
    backsub_width_ratio = process_config['mog_hypothesis_config']['width_ratio']

    frame_count = 0
    while ret:
        frame_count += 1

        # hypothesis making
        # canny
        hypotheses = tools.canny_droplet_hypotheses(frame, detect_mask=arena_mask, config=process_config['canny_hypothesis_config'], debug=deep_debug)
        # hough
        hypotheses += tools.hough_droplet_hypotheses(frame, detect_circle=arena_circle, config=process_config['hough_hypothesis_config'], debug=deep_debug)
        # background suppression
        backsub_mask = backsub.apply(frame, None, backsub_learning_rate)
        backsub_mask = cv2.bitwise_and(backsub_mask, arena_mask)

        if deep_debug:
            cv2.imshow("backsub_mask", backsub_mask)
            cv2.waitKey(WAITKEY_TIME)

        if frame_count > backsub_delay:  # apply only after backsub_delay
            watershed_backsub_mask = tools.watershed(backsub_mask)
            hypotheses += tools.mask_droplet_hypotheses(frame, backsub_mask, width_ratio=backsub_width_ratio, debug=deep_debug)

        # hypothesis solving
        droplet_contours = tools.hypotheses_to_droplet_contours(frame, hypotheses, config=process_config['resolve_hypothesis_config'], debug=deep_debug)

        # storing
        droplet_info.append(droplet_contours)
        droplet_info_list.append(tools.contours_to_list(droplet_contours))

        # plot
        if debug or video_out is not None:
            plot_frame = draw_frame_detection(frame, dish_circle, arena_circle, droplet_contours)

            if video_out is not None:
                video_writer.write(plot_frame)

            if debug:
                cv2.imshow(debug_window_name, plot_frame)
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

    if verbose:
        print '###\nFinished processing video {}.'.format(video_filename)

    return droplet_info
