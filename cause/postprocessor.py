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
        outfile = "%s/breakdown_%s" % (outfolder, self.name)
        index = np.where(self.weights==weight)[0][0]  # location for lambda=weight
        breakdown_perc = self.data[:,index] * 100. / self.data[:,index].sum()
        # write latex table to file
        with open(outfile, 'w') as f:
            for algo in range(self.data.shape[0]):
                f.write("&\t%s\t&\t%.2f\\%%\t\t\n" % (
                    self.data[algo, index], breakdown_perc[algo]))

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
        feats.to_csv("%s/feats" % outfolder,
                     sep='&', index=False, line_terminator='\\\\\n')
        
        Plotter.plot_feature_importances(importances, outfolder, 30)


class PredictionPostprocessor():

    def __init__(self, stats_file):
        schema = {
            'classifier': np.object_,
            'weight': np.float64,
            'algorithm': np.object_,
            'acc_train': np.float64,
            'acc_test': np.float64,
            'mre_train': np.float64,
            'mre_test': np.float64
        }
        columns = ['classifier', 'weight', 'algorithm',
                   'acc_train', 'acc_test', 'mre_train', 'mre_test']
        stats = pd.read_csv(stats_file, header=None, names=columns, dtype=schema)
        # average random classifier runs
        self.__stats = stats.groupby(['weight', 'classifier', 'algorithm']).mean().reset_index()

    @property
    def stats(self):
        return self.__stats

    def save(self, outfolder):
        acc_file = "%s/accuracy_data" % outfolder
        rmre_file = "%s/rmre_data" % outfolder
        # save accuracy to file for pgfplots
        acc = self.stats[self.stats.classifier == "MALAISE"][[
            'weight', 'acc_train', 'acc_test']]
        acc.acc_train *= 100
        acc.acc_test *= 100
        np.savetxt(acc_file, acc,
                   fmt="%.1f\t&\t%.8f\t&\t%.8f",
                   newline="\t\\\\\n")
        # compute rmre
        mre_best = self.stats[self.stats.classifier == "BEST"][[
            'weight', 'algorithm', 'mre_train', 'mre_test']].set_index('weight')
        mre_ml = self.stats[self.stats.classifier == "MALAISE"][[
            'weight', 'mre_train', 'mre_test']].set_index('weight')
        mre_rand = self.stats[self.stats.classifier == "RANDOM"][[
            'weight', 'mre_train', 'mre_test']].set_index('weight')
        rmre = mre_best[['algorithm']]
        rmre.loc[:,'rmre_train_rand'] = mre_ml.mre_train / mre_rand.mre_train 
        rmre.loc[:,'rmre_train_best'] = mre_ml.mre_train / mre_best.mre_train
        rmre.loc[:,'rmre_test_rand'] = mre_ml.mre_test / mre_rand.mre_test
        rmre.loc[:,'rmre_test_best'] = mre_ml.mre_test / mre_best.mre_test
        # save rmre to file for pgfplots
        np.savetxt(rmre_file, rmre.reset_index(),
                   fmt="%.1f\t&\t%s\t&\t%.8f\t&\t%.8f\t&\t%.8f\t&\t%.8f",
                   newline="\t\\\\\n")
        #rmre.to_csv(rmre_file, sep='&', header=False, float_format="%.5f",
        #            index=True, line_terminator='\\\\\n')
