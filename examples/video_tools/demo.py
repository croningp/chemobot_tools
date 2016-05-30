# import the recorder class
from chemobot_tools.video_tools import VideoRecorder

# create an instance with the device number, usually 0 if only one camera connected
vidrec = VideoRecorder(0)

# start to record, this won't block
vidrec.record_to_file('video.avi', duration_in_sec=10)

# you can do stuff while the video is recorded
import time
for i in range(5):
    print '###\nDoing something {}'.format(i)
    time.sleep(1)

# there is a function to wait_until the recording is over
print '###\nWaiting for recording to end'
vidrec.wait_until_idle()

print '###\nVideo recorded in video.avi, open it with any video viewer to check'
