# import the v4l2 class
from chemobot_tools.v4l2 import V4L2

# create an instance with the device number, usually 0 if only one camera connected
vidconf = V4L2(0)

# this print info about the camera and driver
print '##### driver info #####'
vidconf.print_driver_info()

# you can list all controls and menu for this devices and print them on the terminal using
print '##### list controls #####'
vidconf.print_controls()
# this is just mean to be used when playing around with the device

print '##### read original values #####'
# to read the brightness control field
original_saturation_value = vidconf.get_control_value('saturation')
print 'original saturation value is: {}'.format(original_saturation_value)

# to read the contrast control field
original_contrast_value = vidconf.get_control_value('contrast')
print 'original contrast value is: {}'.format(original_contrast_value)

print '##### changing contrast to 0 #####'
# to change the contrast control field, here to 0
vidconf.set_control_value('contrast', 0)
current_contrast_value = vidconf.get_control_value('contrast')
print 'current contrast_value is: {}'.format(current_contrast_value)

print '##### changing contrast back to original #####'
# let's make it back to original the contrast control field
vidconf.set_control_value('contrast', original_contrast_value)
current_contrast_value = vidconf.get_control_value('contrast')
print 'current contrast_value is: {}'.format(current_contrast_value)

# to apply a serie of command, use the apply_config function
# a config is dictionary of controls as field
print '##### applying config to contrast and saturation to 0 #####'
control_config = {
    'contrast': 0,
    'saturation': 0
}
vidconf.apply_config(control_config)

#cheking this went fine
current_saturation_value = vidconf.get_control_value('saturation')
print 'current saturation value is: {}'.format(current_saturation_value)
current_contrast_value = vidconf.get_control_value('contrast')
print 'current contrast value is: {}'.format(current_contrast_value)

# making it back
print '##### applying config to contrast and saturation back to original #####'
original_control_config = {
    'contrast': original_contrast_value,
    'saturation': original_saturation_value
}
vidconf.apply_config(original_control_config)

#cheking this went fine
current_saturation_value = vidconf.get_control_value('saturation')
print 'current saturation value is: {}'.format(current_saturation_value)
current_contrast_value = vidconf.get_control_value('contrast')
print 'current contrast value is: {}'.format(current_contrast_value)

# A final option is to load a config from a file using the apply_config_from_file function
# The file must be json formated
import json

print '##### applying config to contrast and saturation to 0 from file #####'
config_filename = 'control_config.json'
# saving config to file
with open(config_filename, 'w') as f:
    json.dump(control_config, f)

vidconf.apply_config_from_file(config_filename)

#cheking this went fine
current_saturation_value = vidconf.get_control_value('saturation')
print 'current saturation value is: {}'.format(current_saturation_value)
current_contrast_value = vidconf.get_control_value('contrast')
print 'current contrast value is: {}'.format(current_contrast_value)

# making it back
print '##### applying config to contrast and saturation back to original #####'
vidconf.apply_config(original_control_config)

#cheking this went fine
current_saturation_value = vidconf.get_control_value('saturation')
print 'current saturation value is: {}'.format(current_saturation_value)
current_contrast_value = vidconf.get_control_value('contrast')
print 'current contrast value is: {}'.format(current_contrast_value)
