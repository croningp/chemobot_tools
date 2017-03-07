import time

import filetools

from chemobot_tools.droplet_tracking.pool_workers import PoolDropletFeatures, create_default_features_config_from_folder

if __name__ == '__main__':


    import os
    from chemobot_tools.droplet_tracking import droplet_feature as df

    folder_list = filetools.list_folders('simple_videos')
    folder_list.sort()
    for folder in folder_list:
        print folder

        video_filename = os.path.join(folder, 'video.avi')
        dish_info_filename = os.path.join(folder, 'dish_info.json')
        droplet_info_filename = os.path.join(folder, 'droplet_info.json')
        video_out = os.path.join(folder, 'video_tracking.avi')

        dish_info, droplets_statistics, high_level_frame_stats, droplets_ids, grouped_stats = df.aggregate_droplet_info(dish_info_filename, droplet_info_filename, max_distance_tracking=100, min_sequence_length=20, join_min_frame_dist=1, join_max_frame_dist=10, min_droplet_radius=5)

        df.generate_tracking_info_video(video_filename, droplets_statistics, grouped_stats, video_out=video_out)


    # sequence
    # start_time = time.time()
    #
    # dropfeatures = PoolDropletFeatures()
    #
    # folder_list = filetools.list_folders('simple_videos')
    # folder_list.sort()
    # for folder in folder_list:
    #     dropfeatures.add_task(create_default_features_config_from_folder(folder))
    #
    # dropfeatures.wait_until_idle()
    #
    # elapsed = time.time() - start_time
    # print 'It took {} seconds to extract features from {} videos in parallel'.format(elapsed, len(folder_list))
