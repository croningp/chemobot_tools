import os

import filetools

import chemobot_tools.droplet_tracking.droplet_feature as df

folder_list = filetools.list_folders('.')
folder_list.sort()

for folder in folder_list:

    droplet_info = df.load_video_contours_json(os.path.join(folder, 'droplet_info.json'))
    dish_info = df.load_dish_info(os.path.join(folder, 'dish_info.json'))

    droplets_statistics = df.statistics_from_video_countours(droplet_info)
    high_level_frame_stats = df.compute_high_level_frame_descriptor(droplets_statistics)

    droplets_ids = df.track_droplets(droplets_statistics, max_distance=40)
    grouped_stats = df.group_stats_per_droplets_ids(droplets_statistics, droplets_ids, min_sequence_length=10)

    ratio = df.compute_ratio_of_frame_with_droplets(dish_info, droplets_statistics, high_level_frame_stats, grouped_stats)

    speed = df.compute_weighted_mean_speed(dish_info, droplets_statistics, high_level_frame_stats, grouped_stats)

    spread = df.compute_center_of_mass_spread(dish_info, droplets_statistics, high_level_frame_stats, grouped_stats)

    print '###\n{}'.format(folder)
    print ratio, speed, spread, 10000 * ratio * speed * spread

# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# import numpy as np
#
# #
# for s in grouped_stats:
#     plt.plot(s['x_for_speed'], s['speed'])
# plt.show()
#
# #
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=len(grouped_stats))
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# for i, s in enumerate(grouped_stats):
#     colorVal = scalarMap.to_rgba(i)
#     pos = np.array(s['position'])
#     plt.scatter(pos[:, 0], pos[:, 1], c=colorVal)
#
# plt.scatter(high_level_frame_stats['center_of_mass'][:, 0], high_level_frame_stats['center_of_mass'][:, 1], c='k')
# plt.xlim([0, 640])
# plt.ylim([0, 480])
# plt.show()
