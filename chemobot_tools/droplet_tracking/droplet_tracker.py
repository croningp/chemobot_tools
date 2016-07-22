import json

import cv2

import tools

WAITKEY_TIME = 1


DEFAULT_FRAME_CONFIG = {
    'dish_config': tools.DEFAULT_DISH_CONFIG,
    'arena_ratio': 0.8,
    'canny_config': tools.DEFAULT_CANNY_CONFIG
}


def detect_droplet_frame(frame, config=DEFAULT_FRAME_CONFIG, prior_mask=None, debug=False, deep_debug=False):

    dish_circle, dish_mask = tools.find_petri_dish(frame, config=config['dish_config'], debug=deep_debug)

    arena_circle, arena_mask = tools.create_dish_arena(dish_circle, dish_mask, config['arena_ratio'])

    droplet_mask = tools.canny_droplet_detector(frame, arena_mask, config=config['canny_config'], debug=deep_debug)

    if prior_mask is not None:
        droplet_mask = cv2.bitwise_or(droplet_mask, prior_mask)

    watershed_island_mask = tools.watershed(droplet_mask, debug=deep_debug)

    droplet_contours, _ = cv2.findContours(watershed_island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
    'dish_config': tools.DEFAULT_DISH_CONFIG,
    'arena_ratio': 0.8,
    'canny_config': tools.DEFAULT_CANNY_CONFIG,
    'mog_config': {
        'learning_rate': 0.005,
        'delay_by_n_frame': 100
    }
}


def process_video(video_filename, process_config=DEFAULT_PROCESS_CONFIG, video_out=None, droplet_info_out=None, dish_info_out=None, debug=False, deep_debug=False):

    droplet_info = []
    droplet_info_list = []

    dish_circle, dish_mask = tools.get_median_dish_from_video(video_filename, process_config['dish_config'])

    arena_circle, arena_mask = tools.create_dish_arena(dish_circle, dish_mask, process_config['arena_ratio'])

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
    backsub_learning_rate = process_config['mog_config']['learning_rate']
    backsub_delay = process_config['mog_config']['delay_by_n_frame']

    frame_count = 0
    while ret:
        frame_count += 1

        droplet_mask = tools.canny_droplet_detector(frame, arena_mask, config=process_config['canny_config'], debug=deep_debug)

        backsub_mask = backsub.apply(frame, None, backsub_learning_rate)
        backsub_mask = cv2.bitwise_and(backsub_mask, arena_mask)

        if frame_count > backsub_delay:
            droplet_mask = cv2.bitwise_or(droplet_mask, backsub_mask)

        watershed_island_mask = tools.watershed(droplet_mask, debug=deep_debug)

        droplet_contours, _ = cv2.findContours(watershed_island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        droplet_info.append(droplet_contours)
        droplet_info_list.append(tools.contours_to_list(droplet_contours))

        # plot
        if debug or video_out is not None:
            plot_frame = draw_frame_detection(frame, dish_circle, arena_circle, droplet_contours)

            if video_out is not None:
                video_writer.write(plot_frame)

            if debug:
                cv2.imshow("droplet_detection", plot_frame)
                cv2.waitKey(WAITKEY_TIME)

        if deep_debug:
            cv2.imshow("backsub_mask", backsub_mask)
            cv2.waitKey(WAITKEY_TIME)

        # next frame
        ret, frame = video_capture.read()

    video_capture.release()
    if video_out is not None:
        video_writer.release()

    if droplet_info_out is not None:
        with open(droplet_info_out, 'w') as f:
            json.dump(droplet_info_list, f)

    return droplet_info
