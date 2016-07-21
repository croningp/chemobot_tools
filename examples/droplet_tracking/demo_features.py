from chemobot_tools.droplet_tracking.droplet_feature import load_video_contours_json, statistics_from_video_countours, track_droplets, group_stats_per_droplets_ids


droplet_info = load_video_contours_json('0/droplet_info.json')

droplets_statistics = statistics_from_video_countours(droplet_info)

droplets_ids = track_droplets(droplets_statistics)

grouped_stats = group_stats_per_droplets_ids(droplets_statistics, droplets_ids)
