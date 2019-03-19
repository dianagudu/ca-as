from .preprocessor import StatsLoader
from .postprocessor import Plotter

sl = StatsLoader("/home/diana/ca/stats", "ca-compare-3dims")
allstats = sl.load()

#import matplotlib
#del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()

Plotter.plot_average_case(allstats, "/tmp")