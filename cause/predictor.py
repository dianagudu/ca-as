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
        self.__lstats = lstats

    @property
    def lstats(self):
        return self.__lstats

    def evaluate(self):
        pass

    @staticmethod
    def _preprocess_and_split(clsset):
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
            clsset.X, clsset.y, clsset.c,
            test_size=0.3, stratify=clsset.y, random_state=8)
        sc = RobustScaler()  # StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        train = ClassificationSet(X_train_std, y_train, c_train, clsset.le)
        test = ClassificationSet(X_test_std, y_test, c_test, clsset.le)

        return train, test


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

    @property
    def y_pred(self):
        return self.__y_pred

    def predict(self, cls):
        self.__y_pred = cls.predict(self.X)

    def accuracy(self):
        return accuracy_score(self.y, self.y_pred)

    def mre(self):
        return np.mean([
            (
                self.c[i, self.y_pred[i]] - self.c[i, self.y[i]]
            ) ** 2 for i in range(len(self.y))
            ])


class RandomClassifier():

    def __init__(self, labels):
        self.__labels = labels
        # todo: init random state
        # np.random.seed(xx)

    @property
    def labels(self):
        return self.__labels

    def predict(self, X):
        return np.random.choice(self.labels, X.shape[0])


class MalaisePredictor(Predictor):

    def __init__(self, lstats, features):
        super().__init__(lstats)
        self.__features = features

    @property
    def features(self):
        return self.__features

    def predict(self):
        # split into training and test set
        clsset = ClassificationSet.sanitize_and_init(
            self.features.features, self.lstats.winners, self.lstats.costs)
        train, test = Predictor._preprocess_and_split(clsset)

        if True:
            # train classifier on training set
            cls = autosklearn.classification.AutoSklearnClassifier()
            cls.fit(train.X, train.y)
            # print models
            cls.show_models()
            # dump model to file
            #with open(pickled_model, 'wb') as fio:
            #    pickle.dump(cls, fio)

            # predict winners using trained model for training and test sets
            train.predict(cls)
            test.predict(cls)

            print("acc [train]", train.accuracy())
            print("acc [test]", test.accuracy())
            print("mre [train]", train.mre())
            print("mre [test]", test.mre())

        # random prediction
        rcls = RandomClassifier(clsset.le.transform(clsset.le.classes_))
        train.predict(rcls)
        test.predict(rcls)

        print("acc [train]", train.accuracy())
        print("acc [test]", test.accuracy())
        print("mre [train]", train.mre())
        print("mre [test]", test.mre())


