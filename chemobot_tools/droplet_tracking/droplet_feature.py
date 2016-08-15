import copy
import time
import json

import cv2

import numpy as np
from scipy.spatial.distance import cdist

from .tools import list_to_contours

WAITKEY_TIME = 1


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


def statistics_from_frame_countours(contours, min_droplet_radius):

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

            form_factor = (4 * np.pi * area) / (perimeter ** 2)
            roundness = (4 * area) / (np.pi * MA ** 2)
            compactness = np.sqrt(4 * area / np.pi) / MA
            modification_ratio = radius / MA

            M = cv2.moments(contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])

        except:
            continue

        # radius should be >= min_droplet_radius
        if radius < min_droplet_radius:
            continue

        # save
        stats = {
            'position': (centroid_x, centroid_y),
            'bounding_rect': (x, y, w, h),
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
            'ellipse_angle': angle,
            'contour': contour
        }

        frame_stats.append(stats)

    return frame_stats


def statistics_from_video_countours(droplet_info, min_droplet_radius=5):

    droplets_statistics = []

    for contours in droplet_info:
        droplets_statistics.append(statistics_from_frame_countours(contours, min_droplet_radius=min_droplet_radius))
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


def group_stats_per_droplets_ids(droplets_statistics, droplets_ids, min_sequence_length=20, min_frame_dist=1, max_frame_dist=20, max_position_dist=40, verbose=True):

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

    if verbose:
        print 'Found {} droplet sequences'.format(len(grouped_stats))

    grouped_stats = clean_grouped_stats(grouped_stats, min_sequence_length)
    if verbose:
        print 'Cleaned down to {} droplet sequences'.format(len(grouped_stats))

    grouped_stats = join_grouped_stats(grouped_stats, min_frame_dist, max_frame_dist, max_position_dist)
    if verbose:
        print 'Joined to reach {} droplet sequences'.format(len(grouped_stats))

    complement_grouped_stats(grouped_stats)

    return grouped_stats


def clean_grouped_stats(grouped_stats, min_sequence_length):
    # removes all droplet sequence of less than min_sequence_length

    new_grouped_stats = []

    for drop_stats in grouped_stats:
        if len(drop_stats['frame_id']) >= min_sequence_length:
            new_grouped_stats.append(drop_stats)

    return new_grouped_stats


def bind_stats(stats1, stats2):
    stats1 = copy.deepcopy(stats1)
    stats2 = copy.deepcopy(stats2)

    joined_stats = stats1

    frame_end = stats1['frame_id'][-1]
    frame_re_start = stats2['frame_id'][0]

    # fill the frame gap with same values as end of stats1
    # arbitrary, we do not have the info, so better not try to make up anything, linear transition often don't make sense
    frames_to_bind = range(frame_end+1, frame_re_start)
    for i, frame_id in enumerate(frames_to_bind):
        for key, value in joined_stats.items():
            if key == 'frame_id':
                joined_stats[key].append(frame_id)
            else:
                joined_stats[key].append(value[-1])

    # add stats2
    for key, value in stats2.items():
        joined_stats[key] += value

    return joined_stats


def join_grouped_stats(grouped_stats, min_frame_dist=1, max_frame_dist=20, max_position_dist=40):

    # geeting a bunch of info from the group
    start_frame_id = [ds['frame_id'][0] for ds in grouped_stats]
    end_frame_id = [ds['frame_id'][-1] for ds in grouped_stats]
    start_position = [np.array(ds['position'][0]) for ds in grouped_stats]
    end_position = [np.array(ds['position'][-1]) for ds in grouped_stats]

    # make it useful format
    start_id = np.atleast_2d(start_frame_id).T
    end_id = np.atleast_2d(end_frame_id).T
    start_position = np.atleast_2d(start_position)
    end_position = np.atleast_2d(end_position)

    # compute frame distance between tail and head of droplet sequence
    sequence_frame_dist = cdist(start_id, end_id, lambda s, e: s-e)

    # check the one that satisfies condition
    min_dist_frame = sequence_frame_dist > min_frame_dist
    max_dist_frame = sequence_frame_dist < max_frame_dist
    valid_frame_dist = np.logical_and(min_dist_frame, max_dist_frame)

    # compute distance between tail and head of group
    sequence_position_dist = cdist(start_position, end_position, 'euclidean')
    # chech the valid ones
    valid_position_dist = sequence_position_dist < max_position_dist

    # find the indices that are satisfies both
    valid_start, valid_end = np.where(np.logical_and(valid_frame_dist, valid_position_dist))

    #explore the valid one, by ordre of frame distance
    ok_frame_dist = sequence_frame_dist[valid_start, valid_end]
    sort_id = np.argsort(ok_frame_dist)
    # if any join ending group to starting group
    if sort_id.size > 0:
        # we take the first in the list ordered by frame distance
        index = 0
        # the one that ends is the one to bind with the one that starts
        stats1 = grouped_stats[valid_end[sort_id[index]]]
        stats2 = grouped_stats[valid_start[sort_id[index]]]
        joined_stats = bind_stats(stats1, stats2)
        # we replace the one that ended with its extended version
        grouped_stats[valid_end[sort_id[index]]] = joined_stats
        # and delete the other one from the list
        del(grouped_stats[valid_start[sort_id[index]]])

        # Then we do everything again until no more valid group to bind
        grouped_stats = join_grouped_stats(grouped_stats)

    return grouped_stats


def complement_grouped_stats(grouped_stats):

    for drop_stats in grouped_stats:

        positions = np.array(drop_stats['position'])
        delta_positions = np.diff(positions, axis=0)

        speed = np.linalg.norm(delta_positions, axis=1)  # euclidian distance between droplet
        drop_stats['speed'] = list(speed)
        drop_stats['x_for_speed'] = drop_stats['frame_id'][1:]

        acceleration = np.diff(speed)
        drop_stats['acceleration'] = list(acceleration)
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


def aggregate_droplet_info(dish_info_filename, droplet_info_filename, max_distance_tracking=40, min_sequence_length=20, join_min_frame_dist=1, join_max_frame_dist=20, min_droplet_radius=5):

    # getting basic info
    dish_info = load_dish_info(dish_info_filename)
    droplet_info = load_video_contours_json(droplet_info_filename)

    droplets_statistics = statistics_from_video_countours(droplet_info, min_droplet_radius=min_droplet_radius)
    high_level_frame_stats = compute_high_level_frame_descriptor(droplets_statistics)

    droplets_ids = track_droplets(droplets_statistics, max_distance=max_distance_tracking)
    grouped_stats = group_stats_per_droplets_ids(droplets_statistics, droplets_ids, min_sequence_length=min_sequence_length, min_frame_dist=join_min_frame_dist, max_frame_dist=join_max_frame_dist, max_position_dist=max_distance_tracking)

    return dish_info, droplets_statistics, high_level_frame_stats, droplets_ids, grouped_stats


### VISU GROUPING


def generate_tracking_info_frame(frame, frame_count, grouped_stats, debug=True, debug_window_name='droplet_sequence'):

    font = cv2.FONT_HERSHEY_SIMPLEX

    plot_frame = frame.copy()
    for i, drop_stats in enumerate(grouped_stats):
        if frame_count in drop_stats['frame_id']:
            frame_index = drop_stats['frame_id'].index(frame_count)
            drop_id = str(i)
            drop_pos = drop_stats['position'][frame_index]
            x = int(drop_pos[0])
            y = int(drop_pos[1])

            cv2.putText(plot_frame, drop_id, (x, y), font, 0.75, (255,255,255), 2)
            cv2.line(plot_frame, (x-5, y), (x+5, y), (255,0,0))
            cv2.line(plot_frame, (x, y-5), (x, y+5), (255,0,0))

            cv2.drawContours(plot_frame, drop_stats['contour'][frame_index], -1, (0, 255, 0))

    if debug:
        cv2.imshow(debug_window_name, plot_frame)
        cv2.waitKey(WAITKEY_TIME)

    return plot_frame


def generate_tracking_info_video(video_filename, grouped_stats, video_out=None, pause=False, debug=True, debug_window_name='droplet_sequence'):
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
        plot_frame = generate_tracking_info_frame(frame, frame_count, grouped_stats, debug=debug, debug_window_name=debug_window_name)

        if video_out is not None:
            video_writer.write(plot_frame)

        if pause:
            raw_input()

        ret, frame = video_capture.read()
        frame_count += 1

    video_capture.release()
    if video_out is not None:
        video_writer.release()

    if debug:
         cv2.destroyWindow(debug_window_name)
         for _ in range(10):  # ensure it kills the window
             cv2.waitKey(WAITKEY_TIME)


### FEATURES

def compute_ratio_of_frame_with_droplets(droplets_statistics, grouped_stats):

    frame_ids = []
    for stats in grouped_stats:
        frame_ids += stats['frame_id']

    n_frame_with_droplet = len(set(frame_ids))  # set() finds unique value in frame_ids

    # ratio between 0 and 1
    return float(n_frame_with_droplet) / len(droplets_statistics)


def compute_weighted_mean_speed(grouped_stats, dish_info, dish_diameter_mm, frame_per_seconds):

    if len(grouped_stats) == 0:
        return 0

    speeds = [np.mean(d['speed']) for d in grouped_stats]
    weights = [len(d['speed']) for d in grouped_stats]
    weighted_mean_speed_pixel = np.average(speeds, weights=weights)

    dish_diameter_pixel = 2 * dish_info['dish_circle'][2]

    # return in mm/s
    return weighted_mean_speed_pixel / dish_diameter_pixel * dish_diameter_mm * frame_per_seconds


def compute_center_of_mass_spread(high_level_frame_stats, dish_info, dish_diameter_mm):

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
    spread_pixel = np.average(dist_to_mean[:, 0], axis=0, weights=weights)

    dish_diameter_pixel = 2 * dish_info['dish_circle'][2]

    # return in mm
    return spread_pixel / dish_diameter_pixel * dish_diameter_mm


def compute_average_drolet_area(grouped_stats, dish_info, dish_diameter_mm):

    means = [np.mean(stats['area']) for stats in grouped_stats]

    weights = [len(stats['area']) for stats in grouped_stats]

    if len(weights) == 0:
        return 0

    droplet_area_pixel = np.average(means, weights=weights)
    dish_area_pixel = np.pi * dish_info['dish_circle'][2] ** 2
    dish_area_mm = np.pi * dish_diameter_mm ** 2

    # return in mm2
    return droplet_area_pixel / dish_area_pixel * dish_area_mm


def compute_average_circularity(grouped_stats):

    means = [np.mean(stats['form_factor']) for stats in grouped_stats]

    weights = [len(stats['form_factor']) for stats in grouped_stats]

    if len(weights) == 0:
        return 1

    # circularity unit between 0 (non circular) and 1 (perfect circle)
    return np.average(means, weights=weights)


def compute_median_absolute_circularity_deviation(grouped_stats):

    weights = [len(stats['form_factor']) for stats in grouped_stats]

    if len(weights) == 0:
        return 0

    medians = [np.median(stats['form_factor']) for stats in grouped_stats]

    mads = []
    for i, stats in enumerate(grouped_stats):
        abs_deviations = np.abs(stats['form_factor'] - medians[i])
        mads.append(np.median(abs_deviations))

    # weigthed mean MAD (median absolute deviation) circularity
    return np.average(mads, weights=weights)



### ALL

def compute_droplet_features(dish_info_filename, droplet_info_filename, max_distance_tracking=40, min_sequence_length=20, join_min_frame_dist=1, join_max_frame_dist=10, min_droplet_radius=5, dish_diameter_mm=32, frame_per_seconds=20, features_out=None, video_in=None,  video_out=None, debug=False, debug_window_name='droplet_sequence', verbose=False):

    start_time = time.time()

    if verbose:
        print '###\nExtracting features from {} ...'.format(droplet_info_filename)

    # getting basic info
    dish_info, droplets_statistics, high_level_frame_stats, droplets_ids, grouped_stats = aggregate_droplet_info(dish_info_filename, droplet_info_filename, max_distance_tracking=max_distance_tracking, min_sequence_length=min_sequence_length, join_min_frame_dist=join_min_frame_dist, join_max_frame_dist=join_max_frame_dist)

    #
    generate_tracking_info_video(video_in, grouped_stats, video_out=video_out, debug=debug, debug_window_name=debug_window_name)

    #
    features = {}

    features['ratio_frame_active'] = compute_ratio_of_frame_with_droplets(droplets_statistics, grouped_stats)

    features['average_speed'] = compute_weighted_mean_speed(grouped_stats, dish_info, dish_diameter_mm, frame_per_seconds)

    features['average_spread'] = compute_center_of_mass_spread(high_level_frame_stats, dish_info, dish_diameter_mm)

    features['average_area'] = compute_average_drolet_area(grouped_stats, dish_info, dish_diameter_mm)

    features['average_circularity'] = compute_average_circularity(grouped_stats)

    features['median_absolute_circularity_deviation'] = compute_median_absolute_circularity_deviation(grouped_stats)

    if features_out is not None:
        with open(features_out, 'w') as f:
            json.dump(features, f)

    elapsed = time.time() - start_time

    if verbose:
        print '###\nFinished processing droplet info {} in {} seconds.'.format(droplet_info_filename, elapsed)

    return features
