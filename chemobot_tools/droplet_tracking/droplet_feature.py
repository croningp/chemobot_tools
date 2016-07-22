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


def load_dish_info(filename):
    return load_json(filename)


def statistics_from_frame_countours(contours):

    frame_stats = []

    for contour in contours:

        # contour should be >= 5 for fitEllipse
        if len(contour) < 5:
            continue

        # error are usually raised for too small contours
        try:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            equi_diameter = np.sqrt(4 * area / np.pi)

            _, radius = cv2.minEnclosingCircle(contour)
            _, (MA, ma), angle = cv2.fitEllipse(contour)

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            rect_area = w * h

            extent = float(area) / rect_area

            form_factor = 4 * np.pi * area / perimeter ** 2
            roundness = 4 * area / (np.pi * MA ** 2)
            compactness = np.sqrt(4 * area / np.pi) / MA
            modification_ratio = radius / MA

            M = cv2.moments(contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])

        except:
            continue

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
            'centroid': (centroid_x, centroid_y),
            'ellipse_angle': angle
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

        if len(new_pos) != 0:

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


def group_stats_per_droplets_ids(droplets_statistics, droplets_ids, min_sequence_length=3):

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

    grouped_stats = clean_grouped_stats(grouped_stats, min_sequence_length)

    complement_grouped_stats(grouped_stats)

    return grouped_stats


def clean_grouped_stats(grouped_stats, min_sequence_length):
    # removes all droplet sequence of less than min_sequence_length

    new_grouped_stats = []

    for drop_stats in grouped_stats:
        if len(drop_stats['frame_id']) >= min_sequence_length:
            new_grouped_stats.append(drop_stats)

    return new_grouped_stats


def complement_grouped_stats(grouped_stats):

    for drop_stats in grouped_stats:

        positions = np.array(drop_stats['position'])
        delta_positions = np.diff(positions, axis=0)

        speed = np.linalg.norm(delta_positions, axis=1)  # euclidian distance between droplet
        drop_stats['speed'] = speed
        drop_stats['x_for_speed'] = drop_stats['frame_id'][1:]

        acceleration = np.diff(speed)
        drop_stats['acceleration'] = acceleration
        drop_stats['x_for_acceleration'] = drop_stats['x_for_speed'][1:]


def compute_high_level_frame_descriptor(droplets_statistics):

    center_of_mass = []
    total_area = []

    for frame_stats in droplets_statistics:

        if len(frame_stats) > 0:
            # center of mass is the mean position of droplet weighted by their respective areas
            positions = [d['position'] for d in frame_stats]
            weights = [d['area'] for d in frame_stats]
            center_of_mass.append(np.average(positions, axis=0, weights=weights))
            total_area.append(np.sum(weights))
        else:
            # we want to have value associated to frame, so we fill with nan or others when no data available
            center_of_mass.append((np.nan, np.nan))
            total_area.append(np.nan)

    high_level_frame_stats = {}
    high_level_frame_stats['center_of_mass'] = np.array(center_of_mass)
    high_level_frame_stats['total_area'] = np.array(total_area)

    return high_level_frame_stats


def compute_ratio_of_frame_with_droplets(dish_info, droplets_statistics, high_level_frame_stats, grouped_stats):

    n_frame_with_droplet = 0
    for stats in droplets_statistics:
        if len(stats) > 0:
            n_frame_with_droplet += 1

    return float(n_frame_with_droplet) / len(droplets_statistics)


def compute_weighted_mean_speed(dish_info, droplets_statistics, high_level_frame_stats, grouped_stats):

    if len(grouped_stats) == 0:
        return 0

    speeds = [np.mean(d['speed']) for d in grouped_stats]
    weights = [len(d['speed']) for d in grouped_stats]
    weighted_mean_speed = np.average(speeds, weights=weights)

    dish_diameter = 2 * dish_info['dish_circle'][2]

    return weighted_mean_speed / dish_diameter


def compute_center_of_mass_spread(dish_info, droplets_statistics, high_level_frame_stats, grouped_stats):

    positions = high_level_frame_stats['center_of_mass']
    weights = high_level_frame_stats['total_area']

    # remove all nan
    positions = positions[~np.isnan(positions).any(axis=1), :]
    weights = weights[np.logical_not(np.isnan(weights))]

    if weights.size == 0:
        return 0

    # weighted mean
    weighted_mean = np.average(positions, axis=0, weights=weights)

    dist_to_mean = cdist(positions, np.atleast_2d(weighted_mean), 'euclidean')
    spread = np.average(dist_to_mean[:, 0], axis=0, weights=weights)

    dish_diameter = 2 * dish_info['dish_circle'][2]

    return spread / dish_diameter


def compute_relative_perimeter_variation(dish_info, droplets_statistics, high_level_frame_stats, grouped_stats):

    means = [np.mean(stats['perimeter']) for stats in grouped_stats]
    stds = [np.std(stats['perimeter']) for stats in grouped_stats]
    relative_score = np.array(stds) / np.array(means)

    weights = [len(stats['perimeter']) for stats in grouped_stats]

    if weights.size == 0:
        return 0

    return np.average(relative_score, weights=weights)


def compute_average_drolet_area(dish_info, droplets_statistics, high_level_frame_stats, grouped_stats):

    means = [np.mean(stats['area']) for stats in grouped_stats]

    weights = [len(stats['area']) for stats in grouped_stats]

    if weights.size == 0:
        return 0

    dish_area = np.pi * dish_info['dish_circle'][2] ** 2

    return np.average(means, weights=weights) / dish_area
