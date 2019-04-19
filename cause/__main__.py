import numpy as np
import pickle

from cause.stats import ProcessedDataset
from cause.stats import ProcessedStats

from cause.preprocessor import RawStatsLoader
from cause.preprocessor import DatasetCreator
from cause.preprocessor import StatsPreprocessor
from cause.preprocessor import FeatureExtractor
from cause.preprocessor import LambdaStats

from cause.postprocessor import Postprocessor
from cause.postprocessor import FeatsPostprocessor
from cause.postprocessor import PredictionPostprocessor

from cause.plotter import Plotter
from cause.features import Features
from cause.malaise import MALAISEPredictor
from cause.helper import Heuristic_Algorithm_Names


import sys

#name = "ca-compare-3dims"
name = "malaise"

#infolder = "/tmp/stats"   # testing!
#instance_folder = "/tmp/instances"   # testing!
#outfolder = "/tmp/out"   # testing!

infolder = "/home/deedee/ca/stats/" + name
instance_folder = "/home/deedee/ca/datasets/" + name
outfolder = "/home/deedee/ca/processed/" + name


def old_main():
    # get weight and stats file name
    weight = float(sys.argv[1])
    num_processes = int(sys.argv[2])
    feats = Features.load("%s/%s_features.yaml" % (outfolder, name))
    lstats = LambdaStats.load("%s/%s_lstats_%.1f" % (outfolder, name, weight), weight)

    # filter out algos from dataset and reprocess lstats
    algos = [x.name for x in Heuristic_Algorithm_Names \
                if x != Heuristic_Algorithm_Names.GREEDY2 and \
                   x != Heuristic_Algorithm_Names.GREEDY3]
    MALAISEPredictor(lstats.filter(algos), feats).run(
        outfolder=outfolder + "_filtered", num_processes=num_processes)

def main():
    ### test test ###
    # load pickled model and print out
    weight = 1.0
    pickled_model = "%s/%s_cls_model_%.1f" % (outfolder, name, weight)
    cls_ml = pickle.load(open(pickled_model, "rb" ))
    print(cls_ml.show_models())
    print(cls_ml.get_params())

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

# filter out GREEDY2 and GREEDY3 algorithms
#ds = ds.filter([x.name for x in Heuristic_Algorithm_Names \
#                if x != Heuristic_Algorithm_Names.GREEDY2 and \
#                   x != Heuristic_Algorithm_Names.GREEDY3])

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
#MALAISEPredictor(lstats, feats).run(
#    outfolder=outfolder, num_processes=8)

#for weight in ds.weights:
#    MALAISEPredictor(ds.lstats[weight], feats).run(
#        outfolder=outfolder, num_processes=6)

# load MALAISE results
#PredictionPostprocessor("%s/%s_stats" % (outfolder, name)).save(outfolder)

