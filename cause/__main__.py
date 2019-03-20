import numpy as np

from cause.preprocessor import StatsLoader
from cause.plotter import Plotter
from cause.postprocessor import Postprocessor

sl = StatsLoader("/home/diana/ca/stats", "ca-compare-3dims")

allstats = sl.load()
#Plotter.plot_average_case(allstats, "/tmp")

#randstats = sl.load_random()
#Plotter.plot_random(randstats, "/tmp")

postp = Postprocessor(allstats)

weights = np.arange(0, 1.1, 0.1)
breakdown = postp.get_breakdown(weights)

print(breakdown)