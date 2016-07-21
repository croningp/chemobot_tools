import json

import cv2

import numpy as np
from scipy.spatial.distance import cdist

from .tools import list_to_contours


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def load_frame_contours_json(filename):
    return list_to_contours(load_json(filename))


def load_video_contours_json(filename):
    data = load_json(filename)

    droplet_info = []
    for d in data:
        droplet_info.append(list_to_contours(d))
    return droplet_info


def statistics_from_frame_countours(contours):

    frame_stats = []

    for contour in contours:

        # too smal contours are not considered
        if contour.shape[0] < 10:
            continue

        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        equi_diameter = np.sqrt(4 * area / np.pi)

        _, radius = cv2.minEnclosingCircle(contour)
        _, (MA, ma), angle = cv2.fitEllipse(contour)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        rect_area = w * h

        extent = float(area) / rect_area

        form_factor = 4 * np.pi * area / perimeter**2
        roundness = 4 * area / (np.pi * MA**2)
        compactness = np.sqrt(4 * area / np.pi) / MA
        modification_ratio = radius / MA

        M = cv2.moments(contour)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        # save
        stats = {
            'position': (x, y),
            'perimeter': perimeter,
            'area': area,
            'radius': radius,
            'equi_diameter': equi_diameter,
            'aspect_ratio': aspect_ratio,
            'rect_area': rect_area,
            'extent': extent,
            'form_factor': form_factor,
            'roundness': roundness,
            'compactness': compactness,
            'modification_ratio': modification_ratio,
            'moments': M,
            'centroid': (centroid_x, centroid_y)
        }

        frame_stats.append(stats)

    return frame_stats


def statistics_from_video_countours(droplet_info):

    droplets_statistics = []

    for contours in droplet_info:
        droplets_statistics.append(statistics_from_frame_countours(contours))
    return droplets_statistics


def track_droplets(droplets_statistics, max_distance=np.inf):
    """
    Based just on closer center betwen frames
    """

    droplets_ids = []

    prev_pos = None
    prev_id = []
    next_id = 0
    for i, frame_stats in enumerate(droplets_statistics):

        new_id = [None] * len(frame_stats)
        new_pos = np.array([d['position'] for d in frame_stats])

        if i == 0 or len(prev_id) == 0:
            # fill with new ids
            for j in range(len(frame_stats)):
                new_id[j] = next_id
                next_id += 1
        else:
            dist_mat = cdist(prev_pos, new_pos, 'euclidean')
            pairs = find_pair(dist_mat, max_distance)

            # make association of ids
            for row, col in pairs:
                new_id[col] = prev_id[row]

            # fill non associated drops with new id
            for j, value in enumerate(new_id):
                if value is None:  # -1 is default value
                    new_id[j] = next_id
                    next_id += 1

        prev_pos = new_pos
        prev_id = new_id

        droplets_ids.append(new_id)

    return droplets_ids


def find_pair(dist_mat, max_distance):
    dist_mat[dist_mat > max_distance] = np.inf
    pairs = []
    while not np.all(np.isinf(dist_mat)):
        row, col, _ = find_min_idx(dist_mat)
        pairs.append((row, col))
        dist_mat[row, :] = np.inf
        dist_mat[:, col] = np.inf
    return pairs


def find_min_idx(dist_mat):
    min_value = dist_mat.min()
    row, col = np.where(dist_mat == min_value)
    return row[0], col[0], min_value


def group_stats_per_droplets_ids(droplets_statistics, droplets_ids):

    # create empty_stat as list to fill them
    stat_keys = {}
    for i, stats in enumerate(droplets_statistics):
        if len(stats) > 0:
            stat_keys = stats[0].keys()

    # max number of droplet to create array of good size
    max_id = -1
    for i, ids in enumerate(droplets_ids):
        if len(ids) > 0:
            i_max = np.max(ids)
            if np.max(ids) > max_id:
                max_id = i_max

    # intialize stats with empty list for each stats field
    grouped_stats = [{} for _ in range(max_id + 1)]
    for group_stat in grouped_stats:
        for k in stat_keys:
            group_stat[k] = []
        group_stat['frame_id'] = []  # to store the frame number

    for i, drop_ids in enumerate(droplets_ids):
        for j, group_id in enumerate(drop_ids):
            for k, v in droplets_statistics[i][j].items():
                grouped_stats[group_id][k].append(v)
            grouped_stats[group_id]['frame_id'].append(i)

    return grouped_stats
