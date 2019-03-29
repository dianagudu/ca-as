from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import autosklearn.classification
import pickle

import numpy as np


class Predictor():

    def __init__(self, lstats):
        self.__lstats = lstats

    @property
    def lstats(self):
        return self.__lstats

    def evaluate(self):
        pass

    @staticmethod
    def __preprocess_and_split(set):
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
            set.X, set.y, set.c, test_size=0.3, stratify=set.y, random_state=8)
        sc = RobustScaler()  # StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        train = ClassificationSet(X_train_std, y_train, c_train)
        test = ClassificationSet(X_test_std, y_test, c_test)

        return train, test


class ClassificationSet():

    def __init__(self, X, y, c):
        self.__X = X
        self.__y = y
        self.__c = c

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
    def y_pred(self):
        return self.__y_pred   # todo: dict of predictions for each cls used?

    def predict(self, cls):
        self.__y_pred = cls.predict(self.X)

    def accuracy(self):
        return accuracy_score(self.y, self.y_pred)

    def mre(self):
        return ((self.c[self.y_pred] - self.c[self.y]) ** 2).mean()


class RandomClassifier():

    def __init__(self, labels):
        self.__labels = labels
        # todo: init random state
        # np.random.seed(xx)

    @property
    def labels(self):
        return self.__labels

    def predict(self, X):
        # use shape of X to return an array
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
        set = ClassificationSet(
            self.features, self.lstats.winners, self.lstats.costs)
        train, test = Predictor.__preprocess_and_split(set)

        if False:
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
        rcls = RandomClassifier(self.lstats.costs.columns.values)
        train.predict(rcls)
        test.predict(rcls)

        print("acc [train]", train.accuracy())
        print("acc [test]", test.accuracy())
        print("mre [train]", train.mre())
        print("mre [test]", test.mre())


