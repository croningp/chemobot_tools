# import
from chemobot_tools.droplet_labelling.video_labelling import label_video

window_event_selection = 0  # size of +/- frame to take around the selected frame

# add a video.avi to label in a '0' folder and test
label_video('labelled/droplet/0/video.avi', window_event_selection)

# refer to /docs/label_video_commands.md for how to use this tool
