import yaml
import pandas as pd

from cause.plotter import Plotter


class Features():

    def __init__(self, infolder, name, features):
        self.__infolder = infolder
        self.__name = name
        self.__features = features

    @property
    def infolder(self):
        return self.__infolder

    @property
    def name(self):
        return self.__name

    @property
    def features(self):
        return self.__features

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            dobj = yaml.load(f, Loader=yaml.BaseLoader)
        return Features.from_dict(dobj)

    @staticmethod
    def from_dict(dobj):
        features = pd.read_csv(dobj["features"], index_col='instance')
        return Features(dobj["infolder"], dobj["name"], features)

    def plot(self, outfolder="/tmp"):
        Plotter.plot_feature_heatmap(self, outfolder)