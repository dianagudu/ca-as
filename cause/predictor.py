import numpy as np
import pandas as pd
import csv

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import autosklearn.classification
import pickle

from cause.helper import Heuristic_Algorithm_Names


class Predictor():

    def __init__(self, lstats):
        self.__weight = lstats.weight

    @property
    def weight(self):
        return self.__weight

    def run(self):
        pass


class ClassificationSet():

    def __init__(self, X, y, c, le):
        self.__X = X
        self.__y = y
        self.__c = c
        self.__le = le

    @staticmethod
    def sanitize_and_init(features, winners, costs):
        # encode class labels to numbers
        le = LabelEncoder().fit([x.name for x in Heuristic_Algorithm_Names])
        # merge on index
        merged_set = pd.merge(features, winners, left_index=True, right_index=True)
        merged_set = pd.merge(merged_set, costs, left_index=True, right_index=True)
        # reorder cost columns to be sorted in encoded order
        new_costs_columns = le.inverse_transform(np.sort(le.transform(costs.columns)))
        # turn everything into numpy arrays
        X = merged_set[features.columns].values
        y = le.transform(merged_set[winners.columns].values)
        c = merged_set[new_costs_columns].values
        # reshape y
        y = np.reshape(y, (y.shape[0], 1))
        
        return ClassificationSet(X, y, c, le)

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y

    @property
    def c(self):
        return self.__c

    @property
    def le(self):
        return self.__le


class Evaluator():

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def mre(y_true, y_pred, costs):
        return np.mean([(
                costs[i, y_pred[i]] - costs[i, y_true[i]]
            ) ** 2 for i in range(len(y_true))])


class GenericClassifier():

    def __init__(self, train):
        pass

    @property    
    def algo(self):
        return 0

    @property
    def name(self):
        return "GENERIC"

    def predict(self, test):
        # by default, returns perfect classification
        return test.y

    def evaluate(self, test):
        y_pred = self.predict(test)
        acc = Evaluator.accuracy(test.y, y_pred)
        mre = Evaluator.mre(test.y, y_pred, test.c)
        return acc, mre


class RandomClassifier(GenericClassifier):

    def __init__(self, train):
        super().__init__(train)
        self.__labels = train.le.transform(train.le.classes_)
        # todo: init random state
        # np.random.seed(xx)

    @property
    def name(self):
        return "RANDOM"

    @property
    def labels(self):
        return self.__labels

    def predict(self, test):
        return np.random.choice(self.labels, test.X.shape[0])


class BestAlgoClassifier(GenericClassifier):

    def __init__(self, train):
        super().__init__(train)
        # get best algorithm on average on training set
        u, indices = np.unique(train.y, return_inverse=True)
        self.__algo = u[np.argmax(np.bincount(indices))]

    @property    
    def algo(self):
        return self.__algo

    @property
    def name(self):
        return "BEST"

    def predict(self, test):
        return np.full(test.X.shape[0], self.algo, dtype=int)


class MLClassifier(GenericClassifier):

    def __init__(self, train):
        super().__init__(train)
        # train classifier on training set
        self.__cls = autosklearn.classification.AutoSklearnClassifier()
        self.__cls.fit(train.X, train.y)

    @property
    def name(self):
        return "MALAISE"

    @property
    def cls(self):
        return self.__cls

    def dump(self, pickle_file):
        # print models
        self.__cls.show_models()
        # dump model to file
        with open(pickle_file, 'wb') as fio:
            pickle.dump(self.cls, fio)

    def predict(self, test):
        return self.cls.predict(test.X)


class MALAISEPredictor(Predictor):

    def __init__(self, lstats, features):
        super().__init__(lstats)
        self.__clsset = ClassificationSet.sanitize_and_init(
            features.features, lstats.winners, lstats.costs)
  
    @property
    def clsset(self):
        return self.__clsset

    def run(self, outfolder="/tmp"):
        pickle_file = "%s/malaise_cls_model" % outfolder
        stats_file = "%s/malaise_stats" % outfolder

        # split into training and test set
        train, test = self._preprocess_and_split()

        # random prediction
        rand_cls = RandomClassifier(train)
        self.__dump_stats(stats_file, rand_cls, train, test)

        # best algo on average prediction
        algo_cls = BestAlgoClassifier(train)
        self.__dump_stats(stats_file, algo_cls, train, test)
        
        # ml-based prediction
        #ml_cls = MLClassifier(train)
        #self.__dump_stats(stats_file, ml_cls, train, test)
        #ml_cls.dump(pickle_file)

    def __dump_stats(self, stats_file, clsf, train, test):
        acc_train, mre_train = clsf.evaluate(train)
        acc_test, mre_test = clsf.evaluate(test)
        stats = [clsf.name, self.weight,
                 self.clsset.le.inverse_transform(clsf.algo),
                 acc_train, acc_test, mre_train, mre_test]
        with open(stats_file, "a") as f:
            writer = csv.writer(f).writerow(stats)


    def _preprocess_and_split(self):
        X = self.clsset.X
        y = self.clsset.y
        c = self.clsset.c
        ## HACK: for stratified sampling, remove events from classes with just 1 member
        unique_elements, counts_elements = np.unique(y, return_counts=True)
        one_member_classes = [unique_elements[i]
                              for i, count in enumerate(counts_elements)
                              if count == 1]
        for outclass in one_member_classes:
            index = np.argwhere(y==outclass)[0][0]
            y = np.delete(y, index)
            X = np.delete(X, index, 0)
            c = np.delete(c, index, 0)

        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
            X, y, c, test_size=0.3, stratify=y, random_state=8)
        sc = RobustScaler()  # StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        train = ClassificationSet(X_train_std, y_train, c_train, self.clsset.le)
        test = ClassificationSet(X_test_std, y_test, c_test, self.clsset.le)

        return train, test


