import time

from chemobot_tools.droplet_tracking.droplet_tracker import process_video
from chemobot_tools.droplet_tracking.pool_workers import PoolDropletTracker, create_default_tracker_config_from_folder


if __name__ == '__main__':

    # # sequential
    # start_time = time.time()
    #
    # droplet_info = process_video('0/video.avi', debug=True, deep_debug=True)
    #
    # elapsed = time.time() - start_time
    # print 'It took {} seconds to analyse one video'.format(elapsed)

    # parallel
    start_time = time.time()

    droptracker = PoolDropletTracker(verbose=True)

    for i in range(6):
        droptracker.add_task(create_default_tracker_config_from_folder(str(i)))  # need an abspath

    droptracker.wait_until_idle()

    elapsed = time.time() - start_time
    print 'It took {} seconds to analyse {} videos in parallel'.format(elapsed, i + 1)
