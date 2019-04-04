import numpy as np

from cause.stats import ProcessedDataset
from cause.stats import ProcessedStats

from cause.preprocessor import RawStatsLoader
from cause.preprocessor import DatasetCreator
from cause.preprocessor import StatsPreprocessor
from cause.preprocessor import FeatureExtractor
from cause.preprocessor import LambdaStats

from cause.postprocessor import Postprocessor
from cause.plotter import Plotter
from cause.features import Features

from cause.predictor import MalaisePredictor


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
    # quick'n'dirty parallel extraction
    # print command line arguments
    for arg in sys.argv[1:]:
        print(arg)
    task_queue_file = sys.argv[1]
    outfile = sys.argv[2]
    FeatureExtractor.extract_from_queue(instance_folder, name, outfolder, task_queue_file, outfile)

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
#FeatureExtractor.extract(instance_folder, name, outfolder,
#                         in_parallel=True, num_threads=13,
#                         task_queue_file=outfolder + "/features_task_queue")



#feats = Features.load(outfolder + "/" + name + "_features.yaml")


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

#weight = 0.5
#lstats = LambdaStats.load(outfolder + "/" + name + "_lstats_" + str(weight), weight)
#MalaisePredictor(lstats, feats).predict()
