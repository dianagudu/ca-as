import numpy as np

from .plotter import Plotter
from .postprocessor import Breakdown
from .preprocessor import DatasetCreator
from .stats import ProcessedDataset

weights = np.arange(0, 1.1, 0.5)
DatasetCreator.create(weights, "/tmp/stats", "/tmp", "ca-compare-3dims")

ds = ProcessedDataset.load("/tmp/ca-compare-3dims_meta.yaml")

breakdown = Breakdown.from_lstats(ds.l_lstats, ds.weights, ds.pstats.algos, ds.pstats.name)

# save to file for latex table
breakdown.save_to_latex("/tmp")

# plot breakdown as heatmap
breakdown.plot()
