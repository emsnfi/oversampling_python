from sklearn.model_selection import KFold
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from typing_extensions import final
from Py_FS.wrapper.nature_inspired import GA
from sklearn import datasets
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, Conv_plot

# from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, Conv_plot
# import ga

# data = datasets.load_iris()
# d = GA(20, 100, data.data, data.target)
'''
Py FS.wrapper.nature inspired.GA(num agents, max iter, train data,
train label, obj function=compute fitness, trans function shape=‘s’, prob cross=0.4,
prob mut=0.3, save conv graph=False)
'''
# First: oversampling  Second: feature selection
from collections import Counter
import numpy as np
import smote_variants as sv
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import statistics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import math
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from collections import Counter
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as pl
import random

from sklearn import svm, ensemble
# package
from numpy.core.fromnumeric import mean, size
from openpyxl import load_workbook
from openpyxl.styles import Font
from sklearn import tree

from itertools import permutations
import os
import numpy as np
import pandas as pd
import sys


from sklearn.tree import DecisionTreeClassifier
import statistics
from sklearn.metrics import roc_auc_score
from sklearn import svm, ensemble
import time
import datetime
import math
from sklearn.tree import DecisionTreeClassifier
import statistics
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE
from collections import Counter
import math
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from collections import Counter
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as pl
import random

from sklearn import svm, ensemble
from imblearn.over_sampling._smote.base import SMOTE
from imblearn.over_sampling import SMOTE
import smote_variants as sv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import random
import math
import os
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from collections import Counter

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score


def readData(filepath):
    if 'csv' in filepath:
        df = pd.read_csv(filepath)
    elif 'xlsx' in filepath:
        df = pd.read_excel(filepath)
    return df


def GAselect(df):
    labelencoder = LabelEncoder()
    # df 會是資料 dataframe 模式的
    array = df.values
    X = array[:, :-1]
    Y = array[:, -1]
    Y = labelencoder.fit_transform(Y)
    X = MinMaxScaler().fit(X).transform(X)
    # Y = array[:,128]
    columnselect = GA(df, 20, 100, X, Y)
    lastcol = df.columns[-1]
    columnselect = columnselect.append(lastcol)
    dfafterselect = pd.DataFrame(df, columns=columnselect)
    return dfafterselect

# GA


def foldvalGA(df):

    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # enumerate the splits and summarize the distributions
    # print(X)
    # print(y)
    accuracyDe = []
    for train_ix, test_ix in kfold.split(X, y):
        # select rows
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        # summarize train and test composition 兩個不同類別的比例
        train_0, train_1 = len(train_y[train_y == 0]), len(
            train_y[train_y == 1])
        test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
        print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' %
              (train_0, train_1, test_0, test_1))
        # train_X, train_y = sm.fit_resample(train_X, train_y)
        train_X, train_y = pd.DataFrame(train_X), pd.DataFrame(train_y)

        df = pd.concat([train_X, train_y], axis=1)

        df = GAselect(df)
        train_X, train_y = df.iloc[:, :-1], df.iloc[:, -1]
        # train_X, train_y = over.fit_resample(train_X, train_y)

        clf = DecisionTreeClassifier().fit(train_X, train_y)
        # clf = svm.SVC(kernel='linear', C=1, gamma='auto').fit(train_X, train_y)

        test_y_predicted = clf.predict(test_X)

        accuracyDe.append(roc_auc_score(test_y, clf.predict_proba(
            test_X), multi_class="ovr", average='macro'))
        # accuracyDe.append(f1_score(test_y, test_y_predicted))

        # accuracyDe.append( geometric_man_score(test_y, test_y_predicted))

    print(accuracyDe)

    meanDe = statistics.mean(accuracyDe)
    return meanDe


# SMOTE
def foldvalSMOTE(df):
    over = SMOTE(k_neighbors=2)
    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    X = MinMaxScaler().fit(X).transform(X)
    # kfold = KFold(n_splits=5)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracyDe = []
    for train_ix, test_ix in kfold.split(X, y):
        # select rows

        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        # oversampler= sv.MulticlassOversampling(sv.polynom_fit_SMOTE())
        train_X, train_y = over.fit_resample(train_X, train_y)
        clf = DecisionTreeClassifier().fit(train_X, train_y)

        # roc_auc_score(test_y, clf.predict_proba(test_X),multi_class='ovr')
        accuracyDe.append(roc_auc_score(test_y, clf.predict_proba(
            test_X), multi_class="ovr", average='macro'))

    print(accuracyDe)
    meanDe = statistics.mean(accuracyDe)
    return meanDe
# poly prowsyn SMOTE-ipf


def foldvalpoly(df):

    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    X = MinMaxScaler().fit(X).transform(X)
    # kfold = KFold(n_splits=5)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracyDe = []
    for train_ix, test_ix in kfold.split(X, y):
        # select rows

        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        oversampler = sv.MulticlassOversampling(sv.polynom_fit_SMOTE())
        # oversampler = sv.MulticlassOversampling(sv.SMOTE_IPF())
        # oversampler = sv.MulticlassOversampling(sv.ProWSyn())
        train_X, train_y = oversampler.sample(train_X, train_y)
        clf = DecisionTreeClassifier().fit(train_X, train_y)

        # roc_auc_score(test_y, clf.predict_proba(test_X),multi_class='ovr')
        accuracyDe.append(roc_auc_score(test_y, clf.predict_proba(
            test_X), multi_class="ovr", average='macro'))

    print(accuracyDe)
    meanDe = statistics.mean(accuracyDe)
    return meanDe


def foldvalprowsyn(df):

    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    X = MinMaxScaler().fit(X).transform(X)
    # kfold = KFold(n_splits=5)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracyDe = []
    for train_ix, test_ix in kfold.split(X, y):
        # select rows

        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        # oversampler= sv.MulticlassOversampling(sv.polynom_fit_SMOTE())
        # oversampler = sv.MulticlassOversampling(sv.SMOTE_IPF())
        oversampler = sv.MulticlassOversampling(sv.ProWSyn())
        train_X, train_y = oversampler.sample(train_X, train_y)
        clf = DecisionTreeClassifier().fit(train_X, train_y)

        # roc_auc_score(test_y, clf.predict_proba(test_X),multi_class='ovr')
        accuracyDe.append(roc_auc_score(test_y, clf.predict_proba(
            test_X), multi_class="ovr", average='macro'))

    print(accuracyDe)
    meanDe = statistics.mean(accuracyDe)
    return meanDe


def foldvalipf(df):

    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    X = MinMaxScaler().fit(X).transform(X)
    # kfold = KFold(n_splits=5)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracyDe = []
    for train_ix, test_ix in kfold.split(X, y):
        # select rows

        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        # oversampler = sv.MulticlassOversampling(sv.polynom_fit_SMOTE())
        oversampler = sv.MulticlassOversampling(sv.SMOTE_IPF())
        # oversampler = sv.MulticlassOversampling(sv.ProWSyn())
        train_X, train_y = oversampler.sample(train_X, train_y)
        clf = DecisionTreeClassifier().fit(train_X, train_y)

        # roc_auc_score(test_y, clf.predict_proba(test_X),multi_class='ovr')
        accuracyDe.append(roc_auc_score(test_y, clf.predict_proba(
            test_X), multi_class="ovr", average='macro'))

    print(accuracyDe)
    meanDe = statistics.mean(accuracyDe)
    return meanDe


if __name__ == '__main__':
    filepath = '/Users/emily/Desktop/mouseType.csv'
    df = readData(filepath)
    meanDe = foldvalipf(df)
    print('準確率', meanDe)
