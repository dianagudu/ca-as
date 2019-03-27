import yaml
import pandas as pd


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

    def save(self, outfolder):
        info = {
            "infolder": self.infolder,
            "name": self.name,
            "features": outfolder + "/" + self.name + ".features"
        }
        with open(outfolder + "/" + self.name + "_features.yaml", "w") as f:
            yaml.dump(info, f)
        self.features.to_csv(info["features"], float_format='%g')

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            dobj = yaml.load(f, Loader=yaml.BaseLoader)
        return Features.from_dict(dobj)

    @staticmethod
    def from_dict(dobj):
        features = pd.read_csv(dobj["features"], index_col='instance')
        return Features(dobj["infolder"], dobj["name"], features)