import os
import multiprocessing


from chemobot_tools.droplet_classification.droplet_classifier import DropletClassifier


from sklearn import svm, grid_search

if __name__ == '__main__':

    droplet_info = {'name': 'droplet', 'path': os.path.join('extracted', 'droplet')}
    empty_info = {'name': 'empty', 'path': os.path.join('extracted', 'empty')}

    class_info = [droplet_info, empty_info]

    dropcls = DropletClassifier(class_info, size_desc=5, words_size=100)

    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C':[0.1, 1, 10],
        'gamma': [0.1, 1, 10],
        'probability': [True]
    }
    svr = svm.SVC(probability=True)
    clf = grid_search.GridSearchCV(svr, param_grid, n_jobs=multiprocessing.cpu_count())
    clf.probability = True

    dropcls.train(clf)
