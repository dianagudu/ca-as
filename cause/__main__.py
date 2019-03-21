import numpy as np

from .loader import StatsLoader
from .plotter import Plotter
from .postprocessor import Postprocessor

sl = StatsLoader("/home/diana/ca/stats", "ca-compare-3dims")

allstats = sl.load_optimal()
allstats.plot()

randstats = sl.load_random()
randstats.plot()

postp = Postprocessor(allstats)

# get breakdown of dataset by best algorithm
weights = np.arange(0, 1.1, 0.1)
breakdown = postp.get_breakdown_optimal(weights)

# save to file for latex table
breakdown.save_to_latex("/tmp")

# plot breakdown as heatmap
breakdown.plot()
