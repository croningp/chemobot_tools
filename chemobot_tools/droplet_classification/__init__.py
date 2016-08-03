import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

from .droplet_classifier import DropletClassifier

DEFAULT_DROPLET_CLASSIFIER = DropletClassifier.from_folder(os.path.join(HERE_PATH, 'classifier_info'))
