import json

from subprocess import Popen, PIPE


class V4L2ReturnCodeError(Exception):

    def __init__(self, returncode):
        self.returncode = returncode

    def __str__(self):
        return "v4l2 call exited with status returncode {}".format(self.returncode)


class V4L2(object):

    def __init__(self, device):
        self.device = device

    def apply_control_config_from_file(self, control_config):
        with open(control_config) as f:
            return self.apply_config(json.load(f))

    @classmethod
    def from_config(cls, config):
        instance = cls(config['device'])
        instance.apply_control_config(config['control_config'])
        return instance

    @classmethod
    def from_configfile(cls, configfile):
        with open(configfile) as f:
            return V4L2.from_config(json.load(f))

    def call_command(self, command):
        proc = Popen(command, stdout=PIPE)
        proc.wait()
        if proc.returncode != 0:
            raise V4L2ReturnCodeError(proc.returncode)
        return proc

    def forge_command(self, command_list):
        base_command = ['v4l2-ctl', '-d', str(self.device)]
        return base_command + command_list

    def print_driver_info(self):
        """
        -D, --info: Show driver info.
        """
        command = self.forge_command(['--info'])
        proc = self.call_command(command)
        for line in proc.stdout:
            print line

    def print_controls(self):
        """
        -L, --list-ctrls-menus: Display all controls and their menus.
        """
        command = self.forge_command(['-L'])
        proc = self.call_command(command)
        for line in proc.stdout:
            print line

    def get_control_value(self, control_str):
        """
        -C, --get-ctrl=<ctrl>[,<ctrl>...]
        Get the value of the controls.
        """
        command = self.forge_command(['-C', control_str])
        proc = self.call_command(command)
        out = proc.stdout.readline()  # first line is the result
        # out formatted as control_str: value
        # I have seen onyl int or bool, so output is a int, if this pause problem, we will just print the ouput
        return int(out.split(':')[1].strip())

    def set_control_value(self, control_str, value):
        """
        -c, --set-ctrl=<ctrl>=<val>[,<ctrl>=<val>...]
        Set the value of the controls [VIDIOC_S_EXT_CTRLS].
        """
        command_string = control_str + '=' + str(value)
        command = self.forge_command(['-c', command_string])
        self.call_command(command)

    def apply_control_config(self, control_config):
        for key, value in control_config.items():
            self.set_control_value(key, value)
