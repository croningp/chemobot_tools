import time

from chemobot_tools.droplet_tracking.droplet_tracker import process_video
from chemobot_tools.droplet_tracking.pool_workers import PoolDropletTracker, create_default_tracker_config_from_folder


if __name__ == '__main__':

    # # sequential
    # start_time = time.time()
    #
    # droplet_info = process_video('0/video.avi', debug=True, deep_debug=False)
    #
    # elapsed = time.time() - start_time
    # print 'It took {} seconds to analyse one video'.format(elapsed)

    # parallel
    start_time = time.time()

    droptracker = PoolDropletTracker(verbose=True)

    droptracker.add_task(create_default_tracker_config_from_folder('0'))  # need an abspath
    droptracker.add_task(create_default_tracker_config_from_folder('1'))
    droptracker.add_task(create_default_tracker_config_from_folder('2'))

    droptracker.wait_until_idle()

    elapsed = time.time() - start_time
    print 'It took {} seconds to analyse 3 videos in parallel'.format(elapsed)
