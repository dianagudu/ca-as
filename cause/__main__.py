from .preprocessor import StatsLoader
from .postprocessor import Plotter

sl = StatsLoader("/home/diana/ca/stats", "ca-compare-3dims")

#allstats = sl.load()
#Plotter.plot_average_case(allstats, "/tmp")

randstats = sl.load_random()
Plotter.plot_random(randstats, "/tmp")