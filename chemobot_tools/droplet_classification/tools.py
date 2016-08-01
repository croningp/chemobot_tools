import os

import numpy as np

import cv2

from sklearn.cluster import KMeans


def get_all_img_from_folder(folderpath, img_ext='.png'):

    all_img = []
    for (dirpath, dirnames, filenames) in os.walk(folderpath):
        for filename in filenames:
            if filename.endswith(img_ext):
                img_filename = os.path.join(dirpath, filename)
                img = cv2.imread(img_filename)
                if img is None:
                    print img_filename
                    print img
                    raw_input()

                all_img.append(img)

    return all_img


def rgb_to_hsv(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)


def compute_descriptor(rgb_img, size_desc, mask=None):

    hsv_img = rgb_to_hsv(rgb_img)

    hist_1 = cv2.calcHist([hsv_img], [0, 1], mask, [size_desc, size_desc], [0, 256, 0, 256])
    hist_2 = cv2.calcHist([hsv_img], [1, 2], mask, [size_desc, size_desc], [0, 256, 0, 256])
    hist_3 = cv2.calcHist([hsv_img], [2, 0], mask, [size_desc, size_desc], [0, 256, 0, 256])

    hist_ocv = np.array([hist_1.flatten(), hist_2.flatten(), hist_3.flatten()])
    hists = hist_ocv.flatten()

    cv2.normalize(hists, hists, cv2.NORM_L2)

    return hists


def compute_descriptors_for_img_list(rgb_img_list, size_desc, mask_list=None):

    descriptors = []
    for i, img in enumerate(rgb_img_list):

        if mask_list is not None:
            mask = mask_list[i]
        else:
            mask = None

        descriptors.append(compute_descriptor(img, size_desc, mask))

    return descriptors


def descriptors_to_vocabulary(descriptors, words_size):
        '''
        Computes a codebook vocabulary from computed descriptors, using k-means

        Args:
            words_size: number of codebook words for each image class defined

        Returns:
            A compact vocabulary describing the data
        '''

        kmeans_clf = KMeans(n_clusters=words_size)
        kmeans_clf.fit(np.array(descriptors))

        return kmeans_clf.cluster_centers_
