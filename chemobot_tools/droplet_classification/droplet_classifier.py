import os
import json
import pickle

import numpy as np

import cv2

from .tools import get_all_img_from_folder
from .tools import compute_descriptor
from .tools import compute_descriptors_for_img_list
from .tools import descriptors_to_vocabulary


CLF_FILENAME = 'clf.pkl'
DROPCLASS_INFO = 'dropclass_info.json'


class DropletClassifier(object):

    def __init__(self, class_info, size_desc=5, words_size=None):
        """
        class info is of list of dict, each dict contain two field:
            'name': the name of the class
            'path': the path to the sample images
        """

        self.class_info = class_info
        self.size_desc = size_desc
        self.words_size = words_size

    @classmethod
    def from_folder(cls, foldername):
        with open(os.path.join(foldername, DROPCLASS_INFO)) as f:
            dropclass_info = json.load(f)
        new_cls = cls(**dropclass_info)
        new_cls.load_clf_from_file(os.path.join(foldername, CLF_FILENAME))
        return new_cls

    def get_training_data(self):

        X = []
        y = []

        for i, info in enumerate(self.class_info):

            rgb_img_list = get_all_img_from_folder(info['path'])

            descriptors = compute_descriptors_for_img_list(rgb_img_list, size_desc=self.size_desc)

            if self.words_size is not None:
                vocab = descriptors_to_vocabulary(descriptors, words_size=self.words_size)
            else:
                vocab = descriptors

            #
            vocab_list = list(vocab)
            X += vocab_list
            y += [i] * len(vocab_list)

        return np.array(X), np.array(y)

    def load_clf(self, clf):
        self.clf = clf

    def load_clf_from_file(self, filename):
        with open(filename, "rb" ) as f:
            self.load_clf(pickle.load(f))

    def save_clf_to_file(self, filename):
        if not hasattr(self, 'clf'):
            raise Exception('clf not defined yet')
        with open(filename, "wb" ) as f:
            pickle.dump(self.clf,  f)

    def train(self, clf):
        X, y = self.get_training_data()
        clf.fit(X, y)
        self.load_clf(clf)

    def predict(self, descriptor):
        if not hasattr(self, 'clf'):
            raise Exception('clf not defined yet')

        class_id = self.clf.predict(descriptor)
        class_proba = self.clf.predict_proba(descriptor)[0][class_id]
        class_name = self.class_info[class_id]['name']

        return class_name, float(class_proba)

    def predict_img(self, rgb_img):
        descriptor = compute_descriptor(rgb_img, size_desc=self.size_desc)
        return self.predict(descriptor)

    def predict_file(self, rgb_img_filename):
        rgb_img = cv2.imread(rgb_img_filename)
        return self.predict_img(rgb_img)

    def save_dropclass_info_to_json(self, filename):
        dropclass_info = {}
        dropclass_info['class_info'] = self.class_info
        dropclass_info['size_desc'] = self.size_desc
        dropclass_info['words_size'] = self.words_size

        with open(filename, 'w') as f:
            json.dump(dropclass_info, f)

    def save(self, foldername):
        if not os.path.exists(foldername):
            os.makedirs(dest_folder)

        self.save_clf_to_file(os.path.join(foldername, CLF_FILENAME))
        self.save_dropclass_info_to_json(os.path.join(foldername, DROPCLASS_INFO))
