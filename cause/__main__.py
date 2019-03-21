import numpy as np

from .loader import RawStatsLoader
from .plotter import Plotter
from .postprocessor import Breakdown
from .preprocessor import StatsPreprocessor
from .preprocessor import LambdaStatsPreprocessor


#rsl = RawStatsLoader("/home/diana/ca/stats", "ca-compare-3dims")
rsl = RawStatsLoader("/tmp/stats", "ca-compare-3dims")

#allstats = rsl.load_optimal()
#allstats.plot()

#randstats = rsl.load_random()
#randstats.plot()

allstats = rsl.load()
pstats = StatsPreprocessor(allstats).process()

ls_preproc = LambdaStatsPreprocessor(pstats)

# get breakdown of dataset by best algorithm
weights = np.arange(0, 1.1, 0.5)
l_lstats = []
for weight in weights:
    l_lstats.append(ls_preproc.process(weight))
breakdown = Breakdown.from_lstats(l_lstats, weights, pstats.algos, pstats.name)

print(breakdown.algos)
print(breakdown.weights)
print(breakdown.data)

# save to file for latex table
breakdown.save_to_latex("/tmp")

# plot breakdown as heatmap
breakdown.plot()
