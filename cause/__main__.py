from .preprocessor import StatsLoader


sl = StatsLoader("/home/diana/ca/stats", "ca-compare-3dims")
allstats = sl.load()

print(allstats.df.head())