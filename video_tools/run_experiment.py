import sys
import time

import os
import inspect
current_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory


import pycont.controller
SETUP_CONFIG_FILE = os.path.join(current_folder, 'pumps', 'pump_setup_config.json')

controller = pycont.controller.MultiPumpController.from_configfile(SETUP_CONFIG_FILE)

##
sys.path.append('./platform')
import robot_control

import video_tools
video_recorder = video_tools.VideoRecorder(0)

from coordinate import Coordinate
coordinate_config_file = os.path.join(current_folder, 'platform', 'coordinate_config.json')
c = Coordinate.from_configfile(coordinate_config_file)


##
AQUEOUS_VOLUME = 2  # ml
CLEANING_VOLUME = 5  # ml
DROPLET_VOLUME = 4  # uL
EVAPORATION_PAUSE_TIME = 7 * 60  # sec
PETRI_DISH_VOLUME = 3  # ml
SYRINGUE_ZERO = 480


def fill_aqueous_phase(move_level=c.free_level):
    controller.aqueous.wait_until_idle()
    controller.aqueous.pump(AQUEOUS_VOLUME, 'I')
    robot_control.move_tool_at_object(c.aqueous, c.petri_dish_two, c.dispensing_level, move_level)
    controller.wait_until_all_pumps_idle()
    controller.aqueous.deliver(AQUEOUS_VOLUME, 'O', wait=True)


def empty_petri_dish(volume_to_pump_out=CLEANING_VOLUME, wait=False):
    controller.waste.wait_until_idle()
    controller.waste.pump(volume_to_pump_out, 'O', wait=True)
    controller.waste.deliver(volume_to_pump_out, 'I', wait=wait)


def clean_petri_dish(move_level=c.free_level):
    initial_volume = PETRI_DISH_VOLUME-AQUEOUS_VOLUME
    controller.acetone.pump(initial_volume, 'I', wait=False)

    controller.water.pump(PETRI_DISH_VOLUME, 'I', wait=False)

    robot_control.move_tool_at_object(c.cleanhead, c.petri_dish_two, c.cleaning_level, move_level)

    controller.acetone.deliver(initial_volume, 'O', wait=True)
    controller.acetone.pump(PETRI_DISH_VOLUME, 'I', wait=False)
    empty_petri_dish()

    controller.acetone.deliver(PETRI_DISH_VOLUME, 'O', wait=True)
    controller.acetone.pump(PETRI_DISH_VOLUME, 'I', wait=False)
    empty_petri_dish()

    controller.water.deliver(PETRI_DISH_VOLUME, 'O', wait=True)
    empty_petri_dish()

    controller.acetone.deliver(PETRI_DISH_VOLUME, 'O', wait=True)
    empty_petri_dish(12.5, True)

    robot_control.cariage.move([0, -14])
    empty_petri_dish(12.5, True)


def zero_syringes():
    robot_control.syringe1.move_to(SYRINGUE_ZERO, wait=False)
    robot_control.syringe2.move_to(SYRINGUE_ZERO, wait=False)
    robot_control.syringe1.wait_until_idle()
    robot_control.syringe2.wait_until_idle()

def zero_syringe_one():
    robot_control.syringe1.move_to(SYRINGUE_ZERO, wait=True)

def zero_syringe_two():
    robot_control.syringe2.move_to(SYRINGUE_ZERO, wait=True)

def pump_from_vials():
    robot_control.move_tool_at_object(c.syringe_one, c.vial_one, c.pumping_level)
    zero_syringes()
    robot_control.syringe1.move_to(SYRINGUE_ZERO-120, wait=False)
    robot_control.syringe2.move_to(SYRINGUE_ZERO-120, wait=False)
    robot_control.syringe1.wait_until_idle()
    robot_control.syringe2.wait_until_idle()
    robot_control.syringe1.move_to(SYRINGUE_ZERO-100, wait=False)
    robot_control.syringe2.move_to(SYRINGUE_ZERO-100, wait=False)
    robot_control.syringe1.wait_until_idle()
    robot_control.syringe2.wait_until_idle()

def empty_syringe_in_vials():
    robot_control.move_tool_at_object(c.syringe_one, c.vial_one)
    zero_syringes()

def dispense_droplet_syringe_one(relative_position, droplet_volume, start_with_level=c.free_level):
    robot_control.move_tool_at_object(c.syringe_one, c.petri_dish_two, start_with_level=start_with_level)
    robot_control.cariage.move(relative_position, wait=True)
    robot_control.tool_holder.move_to(45.5, wait=True)
    robot_control.syringe1.move(droplet_volume, wait=True)
    # robot_control.tool_holder.move_to(43, wait=True)


def dispense_droplet_syringe_two(relative_position, droplet_volume, start_with_level=c.free_level):
    robot_control.move_tool_at_object(c.syringe_two, c.petri_dish_two, start_with_level=start_with_level)
    robot_control.cariage.move(relative_position)
    robot_control.tool_holder.move_to(42.5, wait=True)
    robot_control.syringe2.move(droplet_volume, wait=True)
    # robot_control.tool_holder.move_to(42, wait=True)


def clean_tubing():
    robot_control.tool_holder.move_to(c.free_level, wait=True)
    robot_control.cariage.move_to([110, 0], wait=True)
    robot_control.tool_holder.move_to(39, wait=True)

    to_clean_pumps = controller.pumps.keys()
    to_clean_pumps.remove('waste')

    controller.apply_command_to_pumps(to_clean_pumps, 'set_valve_position', 'I')
    controller.wait_until_all_pumps_idle()
    controller.apply_command_to_pumps(to_clean_pumps, 'go_to_max_volume')
    controller.wait_until_all_pumps_idle()
    controller.apply_command_to_pumps(to_clean_pumps, 'set_valve_position', 'O')
    controller.wait_until_all_pumps_idle()
    controller.apply_command_to_pumps(to_clean_pumps, 'go_to_volume', 0)
    controller.wait_until_all_pumps_idle()

    controller.waste.transfer(25, 'O', 'I')


def clean_syringes():
    robot_control.tool_holder.move_to(c.free_level, wait=True)
    robot_control.cariage.move_to([160, 0], wait=True)
    zero_syringes()

    controller.acetone.pump(PETRI_DISH_VOLUME, 'I', wait=False)
    robot_control.move_tool_at_object(c.cleanhead, c.petri_dish_two, c.dispensing_level)
    controller.acetone.deliver(PETRI_DISH_VOLUME, 'O', wait=True)

    robot_control.move_tool_at_object(c.syringe_one, c.petri_dish_two, 45)
    robot_control.syringe1.home(wait=True)
    zero_syringe_one()
    robot_control.syringe1.move(-200, wait=True)
    zero_syringe_one()
    robot_control.syringe1.move(-200, wait=True)
    zero_syringe_one()

    robot_control.move_tool_at_object(c.syringe_two, c.petri_dish_two, 43)
    robot_control.syringe2.home(wait=True)
    zero_syringe_two()
    robot_control.syringe2.move(-200, wait=True)
    zero_syringe_two()
    robot_control.syringe2.move(-200, wait=True)
    zero_syringe_two()

    robot_control.tool_holder.move_to(c.free_level, wait=True)
    robot_control.syringe1.move(-100, wait=False)
    robot_control.syringe2.move(-100, wait=True)
    zero_syringes()
    robot_control.syringe1.move(-100, wait=False)
    robot_control.syringe2.move(-100, wait=True)
    zero_syringes()
    robot_control.syringe1.move(-100, wait=False)
    robot_control.syringe2.move(-100, wait=True)
    zero_syringes()

    clean_petri_dish()


def go_to_rest_position(wait=True):
    robot_control.tool_holder.move_to(c.free_level, wait=wait)


def init():
    robot_control.init()

    robot_control.tool_holder.move_to(c.free_level, wait=True)
    robot_control.cariage.move_to([110, 0], wait=True)
    controller.smart_initialize()

    robot_control.tool_holder.move_to(c.free_level, wait=True)
    robot_control.cariage.move_to([160, 0], wait=True)
    zero_syringes()


def wait_print(seconds):
    for t in range(seconds):
        s = 'Running new xp in {} seconds...'.format(seconds-t)
        print s
        time.sleep(1)
        sys.stdout.write("\033[F") # Cursor up one line


def cycle(xp_dict, video_recorder, video_filepath):
    robot_control.open_petri_dish(2)
    robot_control.move_camera_at_object(c.petri_dish_two)
    pump_from_vials()
    fill_aqueous_phase()

    recording_time = xp_dict['recording_time']
    video_recorder.record_to_file(video_filepath,  recording_time)

    level = c.above_petri_dish
    for droplet in xp_dict['droplets']:
        volume = droplet['volume']
        syringe = droplet['syringe']
        relative_position = droplet['position']
        if syringe == 1:
            dispense_droplet_syringe_one(relative_position, volume, level)
        elif syringe == 2:
            dispense_droplet_syringe_two(relative_position, volume, level)

    robot_control.tool_holder.move_to(c.above_petri_dish, wait=True)
    robot_control.cariage.move([70, 0], wait=True)
    robot_control.close_petri_dish(2)
    go_to_rest_position(wait=False)

    print 'Recording...'
    video_recorder.wait_until_idle()
    robot_control.open_petri_dish(2)
    clean_petri_dish()
    go_to_rest_position()

    robot_control.fan.high()
    wait_print(xp_dict['evaporation_time'])
    robot_control.fan.low()


def watch_for_xp():
    import os
    import json

    import filetools


    def run_xp(f):
        print('Running {}'.format(f.abspath))
        with open(f.path) as tmpf:
            xp_dict = json.loads(tmpf.read())

        videofile = f.duplicate()
        videofile.change_filename('video.avi')
        cycle(xp_dict, video_recorder, videofile.path)


    def ignore(f):
        videofile = f.duplicate()
        videofile.change_filename('video.avi')
        return videofile.exists

    xp_watch_path = '/home/group/workspace/diels_alder/XP/ongoing'
    xp_param_filename = 'params.json'

    while True:
        folders = filetools.list_folders(xp_watch_path)
        folders.sort()
        for folder in folders:
            param_file = os.path.join(folder, xp_param_filename)
            if os.path.exists(param_file):
                f = filetools.File(param_file)
                if not ignore(f):
                    run_xp(f)
        time.sleep(10)
