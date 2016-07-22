import os
import time
import threading
import multiprocessing

from collections import deque

from .droplet_tracker import DEFAULT_PROCESS_CONFIG
from .droplet_tracker import process_video

SLEEP_TIME = 0.1


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


class PoolDropletTracker(threading.Thread):

    def __init__(self, n_jobs=multiprocessing.cpu_count(), verbose=False):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interrupted = threading.Lock()
        self.verbose = verbose

        self.config_waiting = deque()
        self.tracking_ongoing = deque([None] * n_jobs, maxlen=n_jobs)
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

    def add_task(self, tracker_config):
        self.config_waiting.append(tracker_config)

    def remove_task(self, tracker_config):
        self.config_waiting.remove(tracker_config)

    def wait_until_idle(self):
        while len(self.config_waiting) > 0:
            time.sleep(SLEEP_TIME)
        for p in self.tracking_ongoing:
            if p is not None:
                p.wait()

    def run(self):
        self.interrupted.acquire()
        while self.interrupted.locked():
            if len(self.config_waiting) > 0:
                # if video ongoing is full, wait
                if self.tracking_ongoing[0] is not None:
                    self.tracking_ongoing[0].wait()

                config = self.config_waiting.popleft()

                video_filename = config['video_filename']
                del config['video_filename']

                p = self.pool.apply_async(process_video, args=(video_filename, ), kwds=config)
                self.tracking_ongoing.append(p)

            time.sleep(SLEEP_TIME)
        self.close()
