import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

from cause.helper import Heuristic_Algorithm_Names
from cause.plotter import Plotter
from cause.predictor import ClassificationSet


class Breakdown():

    def __init__(self, data, weights, algos, name):
        self.__data = data
        self.__weights = weights
        self.__algos = algos
        self.__name = name
        # todo validate input:
        # data is an np.array with dims (num algos, num weights)

    @property
    def data(self):
        return self.__data

    @property
    def weights(self):
        return self.__weights

    @property
    def algos(self):
        return self.__algos

    @property
    def name(self):
        return self.__name

    def save_to_latex(self, outfolder="/tmp", weight=1.):
        outfile = outfolder + "/breakdown_" + self.name
        index = np.where(self.weights==weight)[0][0]  # location for lambda=weight
        breakdown_perc = self.data[:,index] * 100. / self.data[:,index].sum()
        # write latex table to file
        with open(outfile, 'w') as f:
            for algo in range(self.data.shape[0]):
                f.write("&\t%s\t&\t%.2f\\%%\t\t\n" % (self.data[algo, index], breakdown_perc[algo]))

    def plot(self, outfolder="/tmp"):
        Plotter.plot_breakdown(self, outfolder)


class Postprocessor():

    def __init__(self, dataset):
        self.__dataset = dataset

    @property
    def dataset(self):
        return self.__dataset

    def breakdown(self):
        breakdown = np.empty(shape=(0,0))
        for weight in self.dataset.weights:
            column = self.dataset.lstats[weight].get_breakdown(self.dataset.algos)
            if breakdown.shape[0] == 0:
                breakdown = column
            else:
                breakdown = np.vstack([breakdown, column])
        breakdown = np.transpose(breakdown)

        return Breakdown(breakdown, self.dataset.weights,
                         self.dataset.algos, self.dataset.name)


class FeatsPostprocessor(Postprocessor):

    def __init__(self, dataset, features):
        super().__init__(dataset)
        self.__features = features

    @property
    def features(self):
        return self.__features

    def save_feature_importances(self, outfolder):
        # compute feature importances for each weight
        importances = np.empty(shape=(0,0))
        for weight in self.dataset.weights:
            lstats = self.dataset.lstats[weight]
            clsset = ClassificationSet.sanitize_and_init(
                self.features.features, lstats.winners, lstats.costs)
            clf = ExtraTreesClassifier()
            clf = clf.fit(clsset.X, clsset.y)
            if importances.shape[0] == 0:
                importances = clf.feature_importances_
            else:
                importances = np.vstack([importances, clf.feature_importances_])
        # sort feature names by average importance
        sorted_feature_names = [name for _,name in 
                                sorted(zip(importances.mean(axis=0), self.features.features.columns))
                                ][::-1]
        importances = pd.DataFrame(data=importances, columns=self.features.features.columns)
        importances = importances[sorted_feature_names]
        feats = pd.DataFrame(columns=['order', 'value', 'name', 'error'])#, \
                        #dtype={'order': np.int64, 'value': np.float64, 'name':np.object_, 'error': np.float64})
        feats['order'] = np.arange(len(self.features.features.columns))[::-1]
        feats['value'] = importances.mean(axis=0).values
        feats['error'] = importances.std(axis=0).values
        feats['name'] = sorted_feature_names
        feats.to_csv(outfolder + "/feats", sep='&', index=False, line_terminator='\\\\\n')#, fmt="%.5f")
        
        Plotter.plot_feature_importances(importances, outfolder, 30)


