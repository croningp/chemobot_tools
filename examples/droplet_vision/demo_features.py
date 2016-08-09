import time

import filetools

from chemobot_tools.droplet_tracking.pool_workers import PoolDropletFeatures, create_default_features_config_from_folder

if __name__ == '__main__':


    import os
    from chemobot_tools.droplet_tracking import droplet_feature as df

    folder_list = filetools.list_folders('videos')
    folder_list.sort()
    for folder in folder_list:
        print folder

        video_filename = os.path.join(folder, 'video.avi')
        dish_info_filename = os.path.join(folder, 'dish_info.json')
        droplet_info_filename = os.path.join(folder, 'droplet_info.json')
        video_out = os.path.join(folder, 'video_tracking.avi')

        dish_info, droplets_statistics, high_level_frame_stats, droplets_ids, grouped_stats = df.aggregate_droplet_info(dish_info_filename, droplet_info_filename, max_distance_tracking=40, min_sequence_length=20, join_min_frame_dist=1, join_max_frame_dist=10)

        df.generate_tracking_info_video(video_filename, grouped_stats, video_out=video_out)


    # sequence
    # start_time = time.time()
    #
    # dropfeatures = PoolDropletFeatures()
    #
    # folder_list = filetools.list_folders('videos')
    # folder_list.sort()
    # for folder in folder_list:
    #     dropfeatures.add_task(create_default_features_config_from_folder(folder))
    #
    # dropfeatures.wait_until_idle()
    #
    # elapsed = time.time() - start_time
    # print 'It took {} seconds to extract features from {} videos in parallel'.format(elapsed, len(folder_list))

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
