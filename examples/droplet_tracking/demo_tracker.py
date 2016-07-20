import time

from chemobot_tools.droplet_tracking.droplet_tracker import process_video

start_time = time.time()

droplet_info = process_video('video.avi', video_out='video_analysed.avi', debug=True, deep_debug=False)

# droplet_info = process_video('video2.avi', video_out='video2_analysed.avi', debug=True, deep_debug=False)

# droplet_info = process_video('video3.avi', video_out='video3_analysed.avi', debug=True, deep_debug=False)

elapsed = time.time() - start_time
print 'It took {} seconds to analyse the video'.format(elapsed)
