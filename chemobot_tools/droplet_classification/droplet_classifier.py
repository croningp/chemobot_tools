import os
import json
import pickle

import numpy as np

import cv2

from sklearn.metrics import accuracy_score

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

    def get_training_data(self, size_desc, words_size=None):

        X = []
        y = []
        img_paths = []

        for i, info in enumerate(self.class_info):

            bgr_img_list, bgr_img_list_path = get_all_img_from_folder(info['path'])

            descriptors = compute_descriptors_for_img_list(bgr_img_list, size_desc=size_desc)

            if words_size is not None:
                vocab = descriptors_to_vocabulary(descriptors, words_size=words_size)
            else:
                vocab = descriptors

            #
            vocab_list = list(vocab)
            X += vocab_list
            y += [i] * len(vocab_list)
            img_paths += bgr_img_list_path

        return np.array(X), np.array(y), img_paths

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
        X_train, y_train, _ = self.get_training_data(self.size_desc, self.words_size)
        clf.fit(X_train, y_train)
        self.load_clf(clf)

    def accuracy(self):
        if not hasattr(self, 'clf'):
            raise Exception('clf not defined yet')
        # we test with the raw data, before going reducing to vocab via kmeans, if applied
        X_test, y_test, _ = self.get_training_data(self.size_desc, None)
        return accuracy_score(self.clf.predict(X_test), y_test)

    def predict(self, descriptor):
        if not hasattr(self, 'clf'):
            raise Exception('clf not defined yet')

        # ensure 2D for new version of sklearn
        descriptor = np.array(descriptor).reshape(1, -1)

        class_id = self.clf.predict(descriptor)
        class_proba = self.clf.predict_proba(descriptor)[0][class_id]
        class_name = self.class_info[class_id]['name']

        return class_name, float(class_proba)

    def predict_img(self, bgr_img):
        descriptor = compute_descriptor(bgr_img, size_desc=self.size_desc)
        return self.predict(descriptor)

    def predict_file(self, bgr_img_filename):
        bgr_img = cv2.imread(bgr_img_filename)
        return self.predict_img(bgr_img)

    def save_dropclass_info_to_json(self, filename):
        dropclass_info = {}
        dropclass_info['class_info'] = self.class_info
        dropclass_info['size_desc'] = self.size_desc
        dropclass_info['words_size'] = self.words_size

        with open(filename, 'w') as f:
            json.dump(dropclass_info, f)

    def save(self, foldername):
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        self.save_clf_to_file(os.path.join(foldername, CLF_FILENAME))
        self.save_dropclass_info_to_json(os.path.join(foldername, DROPCLASS_INFO))
