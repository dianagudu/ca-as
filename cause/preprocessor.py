import numpy as np
import pandas as pd
import glob
import yaml

from .stats import RawStatsOptimal
from .stats import ProcessedStats
from .stats import LambdaStats
from .loader import RawStatsLoader

class StatsPreprocessor():
    def __init__(self, rawstats):
        self.__rawstats = rawstats

    @property
    def rawstats(self):
        return self.__rawstats

    def process(self):
        if isinstance(self.rawstats.df, RawStatsOptimal):
            pstats = pd.DataFrame(
                self.rawstats.df.groupby('instance')
                          .apply(StatsPreprocessor.__compute_costs_optimal))
        else:
            pstats = pd.DataFrame(
                self.rawstats.df.groupby('instance')
                          .apply(StatsPreprocessor.__compute_costs))

        costt = pstats.pivot(
            index='instance', columns='algorithm', values='costt')
        costw = pstats.pivot(
            index='instance', columns='algorithm', values='costw')

        return ProcessedStats(self.rawstats.name,
                              self.rawstats.algos,
                              self.rawstats.get_welfares(),
                              self.rawstats.get_times(),
                              costw[self.rawstats.algos],
                              costt[self.rawstats.algos])  # reorder columns by algo

    @staticmethod
    def __compute_costs(data):
        wmin = data.welfare.min()
        wmax = data.welfare.max()
        tmin = data.time.min()
        tmax = data.time.max()
        if wmax - wmin == 0:
            data.eval('costw = 0', inplace=True)
        else:
            data.eval(
                'costw = (@wmax - welfare) / (@wmax - @wmin)', inplace=True)
        if tmax - tmin == 0:
            data.eval('costt = 0', inplace=True)
        else:
            data.eval('costt = (time - @tmin) / (@tmax - @tmin)', inplace=True)
        return data

    @staticmethod
    def __compute_costs_optimal(data):
        wcplex = data[data.algorithm == "CPLEX"].welfare.values[0]
        tcplex = data[data.algorithm == "CPLEX"].time.values[0]

        if wcplex == 0:
            data.eval('costw = 0', inplace=True)
        else:
            data.eval('costw = 1. - welfare / @wcplex', inplace=True)

        data.eval('costt = time / @tcplex', inplace=True)
        return data


class LambdaStatsPreprocessor():
    def __init__(self, pstats):
        self.__pstats = pstats

    @property
    def pstats(self):
        return self.__pstats

    def process(self, weight):
        costs = ((weight * self.pstats.costw) ** 2 +
                ((1 - weight) * self.pstats.costt) ** 2) ** 0.5
        winners = costs.idxmin(axis=1)
        return LambdaStats(weight, costs, winners)


class DatasetCreator():
    @staticmethod
    def create(weights, infolder, outfolder, name):
        # filenames
        pstats_file = outfolder + "/" + name + "_pstats.yaml"
        lstats_files = {}
        for weight in weights.tolist():
            lstats_files[weight] = outfolder + "/" + name + "_lstats_" + str(weight) + ".yaml"
        metafile = outfolder + "/" + name + "_meta.yaml"

        # load raw stats
        rsl = RawStatsLoader(infolder, name)
        allstats = rsl.load()

        # process and save raw stats
        pstats = StatsPreprocessor(allstats).process()
        pstats.save(pstats_file)

        # process and save lambda stats per weight
        ls_preproc = LambdaStatsPreprocessor(pstats)
        for weight in weights:
            lstats = ls_preproc.process(weight)
            lstats.save(lstats_files[weight])

        # save dataset metafile
        dobj = {
            "pstats_file": pstats_file,
            "weights": weights.tolist(),
            "lstats_files": lstats_files
        }

        with open(metafile, "w") as f:
            yaml.dump(dobj, f, default_flow_style=False)