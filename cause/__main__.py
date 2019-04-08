import numpy as np

from cause.stats import ProcessedDataset
from cause.stats import ProcessedStats

from cause.preprocessor import RawStatsLoader
from cause.preprocessor import DatasetCreator
from cause.preprocessor import StatsPreprocessor
from cause.preprocessor import FeatureExtractor
from cause.preprocessor import LambdaStats

from cause.postprocessor import Postprocessor
from cause.postprocessor import FeatsPostprocessor

from cause.plotter import Plotter
from cause.features import Features
from cause.predictor import MALAISEPredictor


import sys

#name = "ca-compare-3dims"
name = "malaise"

#infolder = "/tmp/stats"   # testing!
#instance_folder = "/tmp/instances"   # testing!
#outfolder = "/tmp/out"   # testing!

infolder = "/home/deedee/ca/stats/" + name
instance_folder = "/home/deedee/ca/datasets/" + name
outfolder = "/home/deedee/ca/processed/" + name


def main():
    # get weight and stats file name
    weight = float(sys.argv[1])
    outfile = sys.argv[2]
    feats = Features.load("%s/%s_features.yaml" % (outfolder, name))
    lstats = LambdaStats.load("%s/%s_lstats_%.1f" % (outfolder, name, weight), weight)
    MALAISEPredictor(lstats, feats).run(
        outfolder, "%s/malaise_stats_%.1f" % (outfolder, weight))

if __name__ == "__main__":
    main()

# 1st workflow: load stats for alg comparison (incl optimal) and plot avg case
#rsl = RawStatsLoader(infolder, name)
#rsl.load_optimal().plot(outfolder)
# 2nd workflow: load stats for alg comparison for stochastic algos and plot
#rsl.load_random().plot(outfolder)

# 3rd workflow: preprocess dataset (stats for heuristic algos and features)
#               run prediction using ML, plot and save results
#weights = np.array([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
#weights = np.array([0., .5, 1.]) # testing!
#DatasetCreator.create(weights, infolder, outfolder, name)

# extract features
#FeatureExtractor.extract(instance_folder, name, outfolder)

# load processed dataset
#ds = ProcessedDataset.load("%s/%s.yaml" % (outfolder, name))

## some postprocessing: breakdown
#postp = Postprocessor(ds)
## get breakdown by algorithms and weights
#breakdown = postp.breakdown()
## save to file for latex table
#breakdown.save_to_latex(outfolder)
## plot breakdown as heatmap
#breakdown.plot(outfolder)

# load processed features
#feats = Features.load("%s/%s_features.yaml" % (outfolder, name))

## postprocessing: feature importances
#fpostp = FeatsPostprocessor(ds, feats)
#fpostp.save_feature_importances(outfolder)
## plot features as heatmap
#feats.plot(outfolder)

#weight = 0.5
#lstats = LambdaStats.load("%s/%s_lstats_%.1f" % (outfolder, name, weight), weight)
#MALAISEPredictor(lstats, feats).run()

#for weight in ds.weights:
#    MALAISEPredictor(ds.lstats[weight], feats).run(
#        outfolder, "%s/malaise_stats_%.1f" % (outfolder, weight))
