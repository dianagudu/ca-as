import numpy as np

from .loader import RawStatsLoader
from .plotter import Plotter
from .postprocessor import Postprocessor
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
print(pstats.welfares.head())
print(pstats.costw.head())
print(pstats.welfares.values.shape,
      pstats.times.values.shape,
      pstats.costw.values.shape,
      pstats.costt.values.shape)


lstats = LambdaStatsPreprocessor(pstats).process(1.)
print(lstats.costs.head())
print(lstats.winners.head())


#postp = Postprocessor(allstats)

# get breakdown of dataset by best algorithm
#weights = np.arange(0, 1.1, 0.1)
#breakdown = postp.get_breakdown_optimal(weights)

# save to file for latex table
#breakdown.save_to_latex("/tmp")

# plot breakdown as heatmap
#breakdown.plot()
