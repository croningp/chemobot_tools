import time

import filetools

from chemobot_tools.droplet_tracking.droplet_tracker import process_video
from chemobot_tools.droplet_tracking.pool_workers import PoolDropletTracker, create_default_tracker_config_from_folder
from chemobot_tools.droplet_classification.droplet_classifier import DropletClassifier


if __name__ == '__main__':

    dropclf = DropletClassifier.from_folder('classifier_info')


    import cv2
    import numpy as np

    from chemobot_tools.droplet_tracking.tools import hough_droplet_detector, circle_to_droplet_hypothesis, canny_droplet_detector, contour_to_droplet_hypothesis

    # open video to play with frames
    video_capture = cv2.VideoCapture('videos/0/video.avi')

    ret, frame = video_capture.read()
    while ret:

        droplet_circles, droplet_mask = hough_droplet_detector(frame, debug=True)

        denoised_flood_fill = canny_droplet_detector(frame, debug=False)

        droplet_contours, _ = cv2.findContours(denoised_flood_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.imshow('a', denoised_flood_fill)

        droplet_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        for circle in droplet_circles:
            img, mask = circle_to_droplet_hypothesis(circle, frame)

            if img is not None:
                cls_name, cls_proba = dropclf.predict_img(img)

                if cls_name == 'droplet':
                    droplet_mask = cv2.bitwise_or(droplet_mask, mask)
        cv2.imshow('dmsk1', droplet_mask)

        droplet_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        for contour in droplet_contours:
            img, mask = contour_to_droplet_hypothesis(contour, frame)


            if img is not None:
                cv2.imshow('img', img)

                cls_name, cls_proba = dropclf.predict_img(img)

                print cls_name

                if cls_name == 'droplet':
                    droplet_mask = cv2.bitwise_or(droplet_mask, mask)
            raw_input()

        cv2.imshow('dmsk2', droplet_mask)


        for i in range(100):
            cv2.waitKey(1)
        raw_input()

        ret, frame = video_capture.read()





    # # sequential
    # start_time = time.time()
    #
    # droplet_info = process_video('0/video.avi', debug=True, deep_debug=True)
    #
    # elapsed = time.time() - start_time
    # print 'It took {} seconds to analyse one video'.format(elapsed)

    # parallel
    # start_time = time.time()
    #
    # droptracker = PoolDropletTracker()
    #
    # folder_list = filetools.list_folders('.')
    # folder_list.sort()
    # for folder in folder_list:
    #     droptracker.add_task(create_default_tracker_config_from_folder(folder))  # need an abspath
    #
    # droptracker.wait_until_idle()
    #
    # elapsed = time.time() - start_time
    # print 'It took {} seconds to analyse {} videos in parallel'.format(elapsed, len(folder_list))
