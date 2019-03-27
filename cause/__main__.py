import numpy as np

from cause.stats import ProcessedDataset
from cause.stats import ProcessedStats
from cause.preprocessor import RawStatsLoader
from cause.preprocessor import DatasetCreator
from cause.preprocessor import StatsPreprocessor
from cause.preprocessor import FeatureExtractor
from cause.postprocessor import Postprocessor
from cause.plotter import Plotter
from cause.features import Features

name = "ca-compare-3dims"
#name = "malaise"

infolder = "/tmp/stats"   # testing!
#infolder = "/home/diana/ca/stats/" + name
instance_folder = "/tmp/datasets"

outfolder = "/tmp/" + name

# 1st workflow: load stats for alg comparison (incl optimal) and plot avg case
#rsl = RawStatsLoader(infolder, name)
#rsl.load_optimal().plot(outfolder)
# 2nd workflow: load stats for alg comparison for stochastic algos and plot
#rsl.load_random().plot(outfolder)

# 3rd workflow: preprocess dataset (stats for heuristic algos and features)
#               run prediction using ML, plot and save results
#weights = np.array([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
#DatasetCreator.create(weights, infolder, outfolder, name)

# extract features
feats = FeatureExtractor.extract(instance_folder, name)
feats.save(outfolder)

feats = Features.load(outfolder + "/" + name + "_features.yaml")


# load processed dataset
#ds = ProcessedDataset.load(outfolder + "/" + name + ".yaml")

# some postprocessing: breakdown
#postp = Postprocessor(ds)
# get breakdown by algorithms and weights
#breakdown = postp.breakdown()
# save to file for latex table
#breakdown.save_to_latex(outfolder)
# plot breakdown as heatmap
#breakdown.plot(outfolder)
