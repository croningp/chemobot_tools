import os
import time
import threading
import multiprocessing

from collections import deque

from .droplet_tracker import DEFAULT_PROCESS_CONFIG
from .droplet_tracker import process_video
from .droplet_feature import compute_droplet_features

SLEEP_TIME = 0.1


class PoolTemplate(threading.Thread):

    def __init__(self, n_jobs=multiprocessing.cpu_count()):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interrupted = threading.Lock()

        self.config_waiting = deque()
        self.task_ongoing = deque([None] * n_jobs, maxlen=n_jobs)
        self.pool = multiprocessing.Pool(processes=n_jobs)

        self.start()

    def close(self):
        self.pool.close()
        self.pool.join()

    def __del__(self):
        self.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def stop(self):
        self.interrupted.release()

    def add_task(self, config):
        self.config_waiting.append(config)

    def remove_task(self, config):
        self.config_waiting.remove(config)

    def wait_until_idle(self):
        while len(self.config_waiting) > 0:
            time.sleep(SLEEP_TIME)
        for p in self.task_ongoing:
            if p is not None:
                p.wait()

    def run(self):
        self.interrupted.acquire()
        while self.interrupted.locked():
            if len(self.config_waiting) > 0:
                # if video ongoing is full, wait
                if self.task_ongoing[0] is not None:
                    self.task_ongoing[0].wait()

                config = self.config_waiting.popleft()

                func, args, kwds = self.handle_config(config)

                proc = self.pool.apply_async(func, args=args, kwds=kwds)
                self.task_ongoing.append(proc)

            time.sleep(SLEEP_TIME)
        self.close()

    def handle_config(self, config):
        """
            To implement for your need
            func: a function to be run in the pool
            args: tuple of argument (arg1, arg2, )
            kwds: a dict of named arguments
        """

        # return func, args, kwds


def create_default_tracker_config_from_folder(foldername, debug=True):
    tracker_config = {
        'video_filename': os.path.join(foldername, 'video.avi'),
        'process_config': DEFAULT_PROCESS_CONFIG,
        'video_out': os.path.join(foldername, 'video_analysed.avi'),
        'droplet_info_out': os.path.join(foldername, 'droplet_info.json'),
        'dish_info_out': os.path.join(foldername, 'dish_info.json'),
        'debug': debug
    }
    return tracker_config


class PoolDropletTracker(PoolTemplate):

    def handle_config(self, config):
        """
            To implement for your need
            func: a function to be run in the pool
            args: tuple of argument (arg1, arg2, )
            kwds: a dict of named arguments
        """

        func = process_video

        video_filename = config['video_filename']
        del config['video_filename']

        args = (video_filename, )
        kwds = config

        return func, args, kwds


def create_default_features_config_from_folder(foldername):
    features_config = {
        'dish_info_filename': os.path.join(foldername, 'dish_info.json'),
        'droplet_info_filename': os.path.join(foldername, 'droplet_info.json'),
        'max_distance_tracking': 40,
        'min_sequence_length': 20,
        'dish_diameter_mm': 32,
        'frame_per_seconds': 20,
        'features_out': os.path.join(foldername, 'droplet_features.json')
    }
    return features_config


class PoolDropletFeatures(PoolTemplate):

    def handle_config(self, config):
        """
            To implement for your need
            func: a function to be run in the pool
            args: tuple of argument (arg1, arg2, )
            kwds: a dict of named arguments
        """

        func = compute_droplet_features

        dish_info_filename = config['dish_info_filename']
        del config['dish_info_filename']

        droplet_info_filename = config['droplet_info_filename']
        del config['droplet_info_filename']

        args = (dish_info_filename, droplet_info_filename)
        kwds = config

        return func, args, kwds