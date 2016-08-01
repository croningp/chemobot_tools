import os
import sys
import json

import cv2


def forge_database_filename(video_filename):
    base, ext = os.path.splitext(video_filename)
    return base + '.json'


def load_database(database_filename):
    with open(database_filename) as f:
        db = json.load(f)
    return db['database']


def extract_marker_img(frames, marker, windows_size, frame_stepping):

    imgs = []

    ref_frame_id = marker['ref_frame_id']
    selection_width = marker['selection_width']
    selection_position = marker['selection_position']

    print('Extracting maker at frame {} at {} of size {}'.format(ref_frame_id, selection_position, selection_width))

    ##
    half_width = selection_width / 2

    x = selection_position[0] - half_width
    y = selection_position[1] - half_width
    h = selection_width

    for i in range(-windows_size, windows_size + frame_stepping, frame_stepping):
        frame_id = ref_frame_id + i
        cropped = frames[frame_id][y:y + h, x:x + h]
        if cropped.size == 0:
            print 'Marker {} is invalid'.format(marker)
            return None
        imgs.append(cropped)

    return imgs


def extract_markers_from_video(video_filename, windows_size=50, frame_stepping=10):

    ##
    capture = cv2.VideoCapture(video_filename)

    frames = []
    ret, frame = capture.read()
    while ret:
        frames.append(frame)
        ret, frame = capture.read()
    capture.release()

    ##
    database_filename = forge_database_filename(video_filename)
    database = load_database(database_filename)

    ##
    marker_imgs = []
    for marker in database:
        imgs = extract_marker_img(frames, marker, windows_size, frame_stepping)
        if imgs is not None:
            marker_imgs.append(imgs)

    return marker_imgs


def extract_markers_from_folder(folderpath, windows_size=50, frame_stepping=10, verbose=True):

    all_marker_imgs = []
    for (dirpath, dirnames, filenames) in os.walk(folderpath):
        for filename in filenames:
            if filename.endswith('.json'):

                base, ext = os.path.splitext(filename)
                video_filename = os.path.join(dirpath, base + '.avi')

                if os.path.exists(video_filename):
                    if verbose:
                        print('###\nProcessing {} ...'.format(video_filename))

                    marker_imgs = extract_markers_from_video(video_filename, windows_size, frame_stepping)

                    all_marker_imgs += marker_imgs

    return all_marker_imgs


def save_marker_imgs(maker_img_list, dest_folder):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for i, imgs in enumerate(maker_img_list):

        save_folder = os.path.join(dest_folder, str(i))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for j, img in enumerate(imgs):

            save_filename = os.path.join(save_folder, str(j) + '.png')
            cv2.imwrite(save_filename, img)
