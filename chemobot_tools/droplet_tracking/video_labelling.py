import os
import sys
import json

import cv2

WINDOW_EVENT_SELECTION = 50  # plus minus WINDOW_EVENT_SELECTION frames around selection

_platform = sys.platform
if _platform == "linux" or _platform == "linux2":
    KEY_LEFT_CODE = 81
    KEY_RIGHT_CODE = 83
    KEY_UP_CODE = 82
    KEY_DOWN_CODE = 84
elif _platform == "win32" or _platform == "win64":
    KEY_LEFT_CODE = 2424832
    KEY_RIGHT_CODE = 2555904
    KEY_UP_CODE = 2490368
    KEY_DOWN_CODE = 2621440

if _platform == "linux" or _platform == "linux2":
    KEY_4_CODE = 180
    KEY_6_CODE = 182
    KEY_8_CODE = 184
    KEY_5_CODE = 181
elif _platform == "win32" or _platform == "win64":
    KEY_4_CODE = ord('4')
    KEY_6_CODE = ord('6')
    KEY_8_CODE = ord('8')
    KEY_5_CODE = ord('5')

FRAME_INCREMENT = [1, 10, 50, 200]

KEY_Q_CODE = ord('q')

KEY_P_CODE = ord('p')
KEY_L_CODE = ord('l')
KEY_P_L_MOVE = 5
MIN_SELECTION_WIDTH = 20
MAX_SELECTION_WIDTH = 400
DEFAULT_SELECTION_WIDTH = 50

KEY_SAVE_CODE = ord('s')
KEY_DELETE_CODE = ord('d')

KEY_NEXT_MARKER_CODE = ord(' ')


# global variable

# the current frame selected, defined here because used in handle mouse
frame_id = 0
frame_increment_id = 0

# initialize the list of reference points and boolean indicating
# whether selection is being performed or not
selection_width = DEFAULT_SELECTION_WIDTH
selection_position = (0, 0)
selecting = False
frame_id_selected = 0


def convert_key_value(key_pressed):
    return key_pressed & 0xFF


def compare_key_pressed(key_pressed, key_code):
    if _platform == "linux" or _platform == "linux2":
        retval = convert_key_value(key_pressed) == key_code
    elif _platform == "win32" or _platform == "win64":
        if key_pressed > 1000:
            retval = key_pressed == key_code
        else:
            retval = convert_key_value(key_pressed) == key_code

    return retval


def update_frame_increment_id(frame_increment_id, key_pressed):
    if compare_key_pressed(key_pressed, KEY_8_CODE):
        frame_increment_id += 1
    elif compare_key_pressed(key_pressed, KEY_5_CODE):
        frame_increment_id -= 1

    if frame_increment_id < 0:
        frame_increment_id = len(FRAME_INCREMENT) - 1

    if frame_increment_id > len(FRAME_INCREMENT) - 1:
        frame_increment_id = 0

    return frame_increment_id


def update_frame_id(frame_id, key_pressed, frame_increment_id, min_id, max_id):

    if compare_key_pressed(key_pressed, KEY_6_CODE):
        frame_id += FRAME_INCREMENT[frame_increment_id]
    elif compare_key_pressed(key_pressed, KEY_4_CODE):
        frame_id -= FRAME_INCREMENT[frame_increment_id]

    frame_id = min(frame_id, max_id)
    frame_id = max(frame_id, min_id)

    return frame_id


def update_selection_width(selection_width, key_pressed, MIN_SELECTION_WIDTH, MAX_SELECTION_WIDTH):

    if compare_key_pressed(key_pressed, KEY_P_CODE):
        selection_width += KEY_P_L_MOVE
    elif compare_key_pressed(key_pressed, KEY_L_CODE):
        selection_width -= KEY_P_L_MOVE

    selection_width = min(selection_width, MAX_SELECTION_WIDTH)
    selection_width = max(selection_width, MIN_SELECTION_WIDTH)

    return selection_width


def update_selection_position(selection_position, key_pressed):

    new_selection_position = list(selection_position)
    if compare_key_pressed(key_pressed, KEY_UP_CODE):
        new_selection_position[1] -= 1
    elif compare_key_pressed(key_pressed, KEY_DOWN_CODE):
        new_selection_position[1] += 1
    elif compare_key_pressed(key_pressed, KEY_LEFT_CODE):
        new_selection_position[0] -= 1
    elif compare_key_pressed(key_pressed, KEY_RIGHT_CODE):
        new_selection_position[0] += 1

    return tuple(new_selection_position)


def cast_selection_position(selection_position, selection_width, min_pos, max_pos):

    half_width = selection_width / 2

    # fit into screen
    new_selection_position = list(selection_position)
    new_selection_position[0] = min(new_selection_position[0], max_pos[0] - half_width)
    new_selection_position[1] = min(new_selection_position[1], max_pos[1] - half_width)
    new_selection_position[0] = max(new_selection_position[0], min_pos[0] + half_width)
    new_selection_position[1] = max(new_selection_position[1], min_pos[1] + half_width)

    return tuple(new_selection_position)


def handle_mouse(event, x, y, flags, param):
    global selection_position, selecting, frame_id, frame_id_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True

    if selecting:
        selection_position = (x, y)

    if event == cv2.EVENT_LBUTTONUP:
        selecting = False


def cast_frame_id_selected(frame_id_selected, number_of_frames):

    global frame_id

    frame_id_selected = min(frame_id_selected, number_of_frames - 1 - WINDOW_EVENT_SELECTION)
    frame_id_selected = max(frame_id_selected, WINDOW_EVENT_SELECTION)

    return frame_id_selected


def draw_selection(image, pos, width, ref_frame_id, current_frame_id):

    half_width = width / 2
    up_left_corner = (int(pos[0] - half_width), int(pos[1] - half_width))
    up_right_corner = (int(pos[0] + half_width), int(pos[1] + half_width))

    dist = abs(current_frame_id - ref_frame_id)

    if dist == 0:
        color = (0, 255, 0)  # green
        cv2.rectangle(image, up_left_corner, up_right_corner, color, 1)
    elif dist <= WINDOW_EVENT_SELECTION:
        color = (255, 0, 0)  # blue
        cv2.rectangle(image, up_left_corner, up_right_corner, color, 1)


# saving and loading labels
watched_marker_id = None


def draw_selection_from_database(image, database, current_frame_id):

    global watched_marker_id

    for i, marker in enumerate(database):
        dist = abs(current_frame_id - marker['ref_frame_id'])

        if dist <= WINDOW_EVENT_SELECTION:

            pos = marker['selection_position']
            half_width = marker['selection_width'] / 2
            up_left_corner = (int(pos[0] - half_width), int(pos[1] - half_width))
            up_right_corner = (int(pos[0] + half_width), int(pos[1] + half_width))

            if i == watched_marker_id:
                if dist == 0:
                    color = (0, 0, 255)  # red
                else:
                    color = (255, 0, 255)
            else:
                if dist == 0:
                    color = (0, 51, 102)  # brown
                else:
                    color = (51, 102, 153)

            cv2.rectangle(image, up_left_corner, up_right_corner, color, 1)


def update_database(database, database_filename, key_pressed, selection_position, selection_width, ref_frame_id):

    global watched_marker_id

    if compare_key_pressed(key_pressed, KEY_SAVE_CODE):
        print("Adding new marker at frame {} at {} of width {}".format(ref_frame_id, selection_position, selection_width))

        marker = {}
        marker['selection_position'] = selection_position
        marker['selection_width'] = selection_width
        marker['ref_frame_id'] = ref_frame_id

        database.append(marker)
        watched_marker_id = len(database) - 1

        save_database(database, database_filename)

    elif compare_key_pressed(key_pressed, KEY_DELETE_CODE):
        if len(database) > 0:
            print("Removing watched marker")
            database.remove(database[watched_marker_id])
            save_database(database, database_filename)
        print("No watched left")
        cast_watched_marker_id(database)


def forge_database_filename(video_filename):
    base, ext = os.path.splitext(video_filename)
    return base + '.json'


def save_database(database, database_filename):
    db = {}
    db['database'] = database
    with open(database_filename, 'w') as f:
        json.dump(db, f)


def load_database(database_filename):
    with open(database_filename) as f:
        db = json.load(f)
    return db['database']


def cast_watched_marker_id(database):
    global watched_marker_id

    if len(database) == 0:
        watched_marker_id = None
        return

    if watched_marker_id is None:
        watched_marker_id = len(database) - 1

    if watched_marker_id > len(database) - 1:
        watched_marker_id = 0

    if watched_marker_id < 0:
        watched_marker_id = len(database) - 1


def update_frame_id_from_database(frame_id, key_pressed, database):

    global watched_marker_id

    if compare_key_pressed(key_pressed, KEY_NEXT_MARKER_CODE):

        if watched_marker_id is not None:
            watched_marker_id += 1
            cast_watched_marker_id(database)

            frame_id = database[watched_marker_id]['ref_frame_id']

    return frame_id


def label_video(video_filename):

    global frame_id, frame_increment_id, frame_id_selected, selection_width, selection_position

    print("Labelling {}".format(video_filename))

    # loading frames in memory to be able to go back in time
    print("Loading the video in memory, this might take a few seconds...")
    capture = cv2.VideoCapture(video_filename)

    frames = []
    ret, frame = capture.read()
    while ret:
        frames.append(frame)
        ret, frame = capture.read()
    capture.release()

    ##
    print("The video contains {} frames".format(len(frames)))
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]
    print("Each frame is {}x{}".format(frame_width, frame_height))

    ##

    database_filename = forge_database_filename(video_filename)
    if os.path.exists(database_filename):
        print("Loading previous database..")
        database = load_database(database_filename)
    else:
        print("Creating new database..")
        database = []
    cast_watched_marker_id(database)

    ##
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', handle_mouse)

    while True:

        key_pressed = cv2.waitKey(1)
        if key_pressed != -1:
            if compare_key_pressed(key_pressed, KEY_Q_CODE):
                break
            else:
                frame_increment_id = update_frame_increment_id(frame_increment_id, key_pressed)
                frame_id = update_frame_id(frame_id, key_pressed, frame_increment_id, 0, len(frames) - 1)
                selection_width = update_selection_width(selection_width, key_pressed, MIN_SELECTION_WIDTH, MAX_SELECTION_WIDTH)
                selection_position = update_selection_position(selection_position, key_pressed)
                update_database(database, database_filename, key_pressed, selection_position, selection_width, frame_id_selected)
                frame_id = update_frame_id_from_database(frame_id, key_pressed, database)

        if selecting:
            frame_id_selected = frame_id
            frame_id_selected = cast_frame_id_selected(frame_id_selected, len(frames))

        ##
        image = frames[frame_id].copy()

        selection_position = cast_selection_position(selection_position, selection_width, (0, 0), (frame_width - 1, frame_height - 1))
        draw_selection(image, selection_position, selection_width, frame_id_selected, frame_id)

        draw_selection_from_database(image, database, frame_id)

        cv2.putText(image, "F:" + str(frame_id), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(image, "I:" + str(FRAME_INCREMENT[frame_increment_id]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('image', image)

    cv2.destroyAllWindows()


if __name__ == "__main__":

    video_filename = sys.argv[1]
    label_video(video_filename)
