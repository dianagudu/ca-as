import numpy as np
import pandas as pd

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
    def labels(self):
        return self.__labels

    def predict(self, test):
        return np.random.choice(self.labels, test.X.shape[0])


class BestAlgoClassifier(GenericClassifier):

    def __init__(self, train):
        super().__init__(train)
        # get best algorithm on average
        self.__best_algo = 0

    def predict(self, test):
        return 0


class MLClassifier(GenericClassifier):

    def __init__(self, train):
        super().__init__(train)
        # train classifier on training set
        self.__cls = autosklearn.classification.AutoSklearnClassifier()
        self.__cls.fit(train.X, train.y)

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
        self.__clsset = ClassificationSet.sanitize_and_init(
            features.features, lstats.winners, lstats.costs)
  
    @property
    def clsset(self):
        return self.__clsset

    def run(self, outfolder="/tmp"):
        pickle_file = "%s/ml_cls_model" % (outfolder)

        # split into training and test set
        train, test = self._preprocess_and_split()

        # random prediction
        rand_cls = RandomClassifier(train)
        rand_cls.evaluate(train)
        rand_cls.evaluate(test)

        # best algo on average prediction
        algo_cls = BestAlgoClassifier(train)
        
        # ml-based prediction
        ml_cls = MLClassifier(train)
        ml_cls.dump(pickle_file)

    # todo: function to dump stats to file


    def _preprocess_and_split(self):
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
            self.clsset.X, self.clsset.y, self.clsset.c,
            test_size=0.3, stratify=self.clsset.y, random_state=8)
        sc = RobustScaler()  # StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        train = ClassificationSet(X_train_std, y_train, c_train, self.clsset.le)
        test = ClassificationSet(X_test_std, y_test, c_test, self.clsset.le)

        return train, test


