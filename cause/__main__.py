import numpy as np

from .loader import StatsLoader
from .plotter import Plotter
from .postprocessor import Postprocessor

sl = StatsLoader("/home/diana/ca/stats", "ca-compare-3dims")

allstats = sl.load_optimal()
Plotter.plot_average_case(allstats, "/tmp")

randstats = sl.load_random()
Plotter.plot_random(randstats, "/tmp")

postp = Postprocessor(allstats)

# get breakdown of dataset by best algorithm
weights = np.arange(0, 1.1, 0.1)
breakdown = postp.get_breakdown_optimal(weights)

# save to file for latex table
breakdown.save_to_latex("/tmp")

# plot breakdown as heatmap
Plotter.plot_breakdown(breakdown)
