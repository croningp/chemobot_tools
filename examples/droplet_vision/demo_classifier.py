import os
import time
import random
import multiprocessing

import cv2
import numpy as np

from chemobot_tools.droplet_classification.droplet_classifier import DropletClassifier

from sklearn import grid_search
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':

    ## ensure consistent run
    random.seed(0)
    np.random.seed(0)

    ##
    droplet_info = {'name': 'droplet', 'path': os.path.join('extracted', 'droplet')}
    empty_info = {'name': 'empty', 'path': os.path.join('extracted', 'empty')}

    class_info = [droplet_info, empty_info]

    # try different classifiers
    n_fold_cv = 10

    classifiers = {}

    # svm
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C':[0.1, 1, 10],
        'gamma': [0.1, 1, 10]
    }
    svr = SVC(probability=True)
    clf = grid_search.GridSearchCV(svr, param_grid, n_jobs=multiprocessing.cpu_count(), cv=n_fold_cv)
    classifiers['SVM'] = clf

    #logistic regression
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C':[0.1, 1, 10]
    }
    logreg = LogisticRegression()
    clf = grid_search.GridSearchCV(logreg, param_grid, n_jobs=multiprocessing.cpu_count(), cv=n_fold_cv)
    classifiers['LogReg'] = clf

    # KNeighborsClassifier
    param_grid = {
        'n_neighbors': range(5,105,5)
    }
    knn = KNeighborsClassifier()
    clf = grid_search.GridSearchCV(knn, param_grid, n_jobs=multiprocessing.cpu_count(), cv=n_fold_cv)
    classifiers['KNN'] = clf

    #
    n_repeat = 1000
    size_desc = 5
    words_size=None

    results = {}
    for k, clf in classifiers.items():
        dropcls = DropletClassifier(class_info, size_desc=size_desc, words_size=words_size)
        dropcls.train(clf)
        # time
        img = cv2.imread('extracted/droplet/0/0.png')
        start_time = time.time()
        for _ in range(n_repeat):
             dropcls.predict_img(img)
        elasped = time.time() - start_time

        result = {}
        result['cv_score'] = dropcls.clf.best_score_
        result['accuracy'] = dropcls.accuracy()
        result['predict_time'] = elasped / n_repeat
        result['param'] = dropcls.clf.best_params_
        result['clf'] = dropcls.clf
        results[k] = result

    # save the svm one
    dropcls = DropletClassifier(class_info, size_desc=size_desc, words_size=words_size)
    dropcls.load_clf(results['SVM']['clf'])
    dropcls.save('classifier_info')
