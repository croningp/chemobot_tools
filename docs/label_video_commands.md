## Commands for the label script

To start analyzing a video, run:

```python
from chemobot_tools.droplet_tracking import label_video
label_video('video.avi')
```

This opens a window with the video.

### Summary of command

#### keypad

- 4 : decrement frame
- 6 : increment frame
- 8 : increment frame stepping
- 5 : decrement frame stepping

#### direction keys

- up : move prospective marker 1 pixel up
- down : move prospective marker 1 pixel down
- left : move prospective marker 1 pixel left
- right : move prospective marker 1 pixel right

#### mouse

Right click to move the prospective marker quickly

#### letters

- p : increase marker width (p for "plus")
- l : decrease marker width (l for "less")
- s : save marker to the database (s for "save")
- d : delete selected marker from the database (d for "delete")
- space : cycle through the marker in the database
- q : quit

#### Moving through frames

The frame number is written on the top left corner, in red: "F:frame_number"
To change frame use the key 4 and 6 from your keypad.
The increment by which the frames are changed is written in white below the frame number.

To change the increment use the key 8 and 5 from the keypad. This will cycle between 4 choices [1, 10, 50, 200], which is well suited to move through the video both precisely and quickly as needed.
Positioning the marker window

A marker is square that has a center and width. The prospective new marker is displayed in green.
Right click with the mouse on the location you want to track. You can move the prospective marker finely with the arrow keys.
Changing the with of the marker

With the key 'p' and 'l', you can respectively increase (p for "plus") and decrease (l for "less") the width of the marker

### Managing the database

For each video file, let's call it name_of_video.avi, we create a json file holding the information about saved markers. This file is called name_of_video.json.

To save the prospective marker to the database, use the 's' key. This will change the color of the marker to red.

You can add as many markers to the database as you want. To cycle through the markers use the space key. The current selected marker from the database will be highlighted in bright red, the others ones in darker red/brown.

You can deleted the currently selected marker form the database by pressing the 'd' key.

The databased is saved to the name_of_video.json file each time a 's' or 'd' action is made. If name_of_video.json exists when you start a program, it simply reload it content, you can continue working where you left it.

### Additional feature

The marker applies for N frames before and after the selected frame. The color of the marker will change if it is within that interval but not on the frame selected originally.

At the frame you clicked, the prospective marker is green, at +/- N frames away, the marker is blue. The same applies to the marker in the database, which will have a dimmer color.

This is useful because we want to label event that occurs in multiple frames, such as fission or fusion. We need N to be constant, so you cannot change this parameter. N affect the frame at which you can apply markers, i.e. only between the N frames and the TotalFrame-N frame.
