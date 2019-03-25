import numpy as np

from .stats import ProcessedDataset
from .loader import RawStatsLoader
from .preprocessor import DatasetCreator
from .preprocessor import StatsPreprocessor
from .postprocessor import Postprocessor
from .plotter import Plotter

name = "ca-compare-3dims"

# infolder = "/tmp/stats"   # testing!
infolder = "/home/diana/ca/stats/" + name
outfolder = "/tmp/" + name

# 1st workflow: load stats for alg comparison (incl optimal) and plot avg case
rsl = RawStatsLoader(infolder, name)
#rsl.load_optimal().plot(outfolder)
# 2nd workflow: load stats for alg comparison for stochastic algos and plot
rsl.load_random().plot(outfolder)

# 3rd workflow: preprocess dataset (stats for heuristic algos and features)
#               run prediction using ML, plot and save results
weights = np.arange(0, 1.1, 0.1)
DatasetCreator.create(weights, infolder, outfolder, name)

ds = ProcessedDataset.load(outfolder + "/" + name + "_meta.yaml")
postp = Postprocessor(ds)

# get breakdown by algorithms and weights
breakdown = postp.breakdown()
# save to file for latex table
breakdown.save_to_latex(outfolder)
# plot breakdown as heatmap
breakdown.plot()
