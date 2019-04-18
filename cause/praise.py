import csv
from sklearn.preprocessing import LabelEncoder

from cause.predictor import Predictor
from cause.predictor import Evaluator


class PRAISEPredictor(Predictor):
    def __init__(self, pstats, lstats, sstats, lstats_sample):
        super().__init__(lstats)
        self.__pstats = pstats
        self.__lstats = lstats
        self.__sstats = sstats
        self.__lstats_sample = lstats_sample

    @property
    def ratio(self):
        return self.sstats.ratio

    @property
    def name(self):
        return self.sstats.name

    @property
    def pstats(self):
        return self.__pstats

    @property
    def lstats(self):
        return self.__lstats

    @property
    def sstats(self):
        return self.__sstats

    @property
    def lstats_sample(self):
        return self.__lstats_sample

    def run(self, outfolder="/tmp"):
        stats_file = "%s/%s_stats_%.1f_%.2f" % (
            outfolder, self.name, self.weight, self.ratio)

        # only instances that are in the sample stats
        instances = self.lstats_sample.winners.index.tolist()

        y_true = self.lstats.winners.loc[instances].values
        y_pred = self.lstats_sample.winners.values
        y_pred_extra = self.lstats_sample.winners_extra.values
        costs = self.lstats.costs.loc[instances].values
        
        # encode class labels to numbers
        le = LabelEncoder().fit(self.lstats.costs.columns.values)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        y_pred_extra = le.transform(y_pred_extra)

        # compute costs with overhead
        costt_ovhd = PRAISEPredictor.__compute_costt_ovhd(
            self.pstats.times.loc[instances], self.sstats.t_ovhd)
        costs_ovhd = ((self.weight * self.pstats.costw.loc[instances]) ** 2 +
                ((1 - self.weight) * costt_ovhd) ** 2) ** 0.5
        costs_ovhd = costs_ovhd.values

        acc = Evaluator.accuracy(y_true, y_pred)
        mse = Evaluator.mre_ovhd(y_true, y_pred, costs, costs_ovhd)
        acc_extra = Evaluator.accuracy(y_true, y_pred_extra)
        mse_extra = Evaluator.mre_ovhd(y_true, y_pred_extra, costs, costs_ovhd)

        stats = ["PRAISE", self.weight, self.ratio,
                 acc, mse, acc_extra, mse_extra]
        with open(stats_file, "a") as f:
            csv.writer(f).writerow(stats)

    @staticmethod
    def __compute_costt_ovhd(times, t_ovhd):
        tmin = times.min(axis=1)
        tmax = times.max(axis=1)
        costt_ovhd = times.add(
            t_ovhd['0'], axis="index").sub(
            tmin, axis="index").div(
            tmax - tmin, axis="index").fillna(0.)
        ## what about nan values? => replace with 0
        #costt_ovhd = (times + t_ovhd - tmin).div(tmax - tmin)
        return costt_ovhd