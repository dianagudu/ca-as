import numpy as np
import pandas as pd
import pylab as plt
from matplotlib import cm
from matplotlib import rc


rc("text", usetex=False)
rc("mathtext", fontset="custom")
rc("mathtext", default="regular")
rc("font",**{"family":"serif",
                "serif":["EB Garamond"],
                "size":14})

class Plotter():

    @staticmethod
    def plot_average_case(allstats, outfolder):
        welfares = allstats.get_welfares_feasible()
        times = allstats.get_times_feasible()

        # normalize welfare and time by values of optimal algorithm (cplex)
        welfares = welfares.div(welfares.CPLEX, axis=0).multiply(100., axis=0)
        times = times.div(times.CPLEX, axis=0).multiply(100., axis=0)

        outfile_welfare = outfolder + "/" + "welfare_" + allstats.name
        outfile_time = outfolder + "/" + "time_" + allstats.name

        Plotter.__boxplot_average_case(welfares, allstats.algos, outfile_welfare,
                             ylabel="% of optimal welfare (CPLEX)")
        Plotter.__boxplot_average_case(times, allstats.algos, outfile_time,
                             top=100000, bottom=0.01, ylog=True,
                             ylabel="% of time of optimal algorithm (CPLEX)")

    @staticmethod
    def plot_random(randstats, outfolder):
        outfile = outfolder + "/random_" + randstats.name
        welfares = randstats.df[['instance','algorithm','welfare']]

        # normalize welfare by average value on each instance
        welfares_means = pd.DataFrame(welfares.groupby(['instance', 'algorithm']).mean().welfare.reset_index(name='mean_welfare'))
        welfares = welfares.merge(welfares_means)
        welfares[['welfare']] = welfares.welfare.div(welfares.mean_welfare, axis=0) * 100. - 100.
        welfares = welfares.dropna()

        print(welfares.index)
        #welfares = welfares.pivot(columns='algorithm', values='welfare')

        #Plotter.__boxplot_average_case(welfares, randstats.algos, outfile,
        #                               bottom=-100, top=100,
        #                               ylabel = "difference to mean welfare (%)")


    @staticmethod
    def __boxplot_average_case(data, algos, filename,
                               bottom=-10, top=110,
                               ylog=False,
                               ylabel="\% of optimal"):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        bp = plt.boxplot(np.array(data.values), notch=1, vert=1, whis=[5, 95],
                         bootstrap=100, showmeans=True, showfliers=True)
        plt.setp(bp["boxes"], color="black")
        plt.setp(bp["whiskers"], color="black")
        plt.setp(bp["fliers"], color="grey",
                 marker=".", mew=0.5, mec="grey", markersize=3.5)
        plt.setp(bp["means"], color="red",
                 marker="*", mec="red", mfc="red", mew=0.5)
        plt.setp(bp["medians"], color="blue")
        # add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax1.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
        ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
        # hide grid behind plot objects
        ax1.set_axisbelow(True)
        ax1.set_xlabel("algorithm")
        ax1.set_ylabel(ylabel)
        # axis limits
        ax1.set_ylim(bottom, top)
        if ylog:
            ax1.set_yscale("log")
        # Due to the Y-axis scale being different across samples, it can be
        # hard to compare differences in medians across the samples. Add upper
        # X-axis tick labels with the sample medians to aid in comparison
        # (just use two decimal places of precision)
        numBoxes = len(algos)
        pos = np.arange(numBoxes) + 1
        medians = [bp["medians"][i].get_ydata()[0] for i in range(0, numBoxes)]
        means = [bp["means"][i].get_ydata()[0] for i in range(0, numBoxes)]
        upperLabels = [str(np.round(s, 2)) for s in means]
        if ylog:
            labelpos = top - (top * 0.6)
        else:
            labelpos = top - (top * 0.07)
        for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
            k = tick % 2
            ax1.text(pos[tick], labelpos, upperLabels[tick],
                     horizontalalignment="center", size="x-small", color="r")
        # tick labels
        xtickNames = plt.setp(
            # ax1, xticklabels=["\\textsc{%s}" % a.lower() for a in algos])
            ax1, xticklabels=algos)
        plt.setp(xtickNames, rotation=45, fontsize=10)
        # Finally, add a basic legend
        plt.figtext(0.795, 0.09, "-", color="blue", weight="roman", size="medium")
        plt.figtext(0.815, 0.092, " median value", color="black", weight="roman", size="x-small")
        plt.figtext(0.795, 0.058, "*", color="red", weight="roman", size="medium")
        plt.figtext(0.815, 0.07, " average value", color="black", weight="roman", size="x-small")
        plt.figtext(0.7965, 0.045, "o", color="grey", weight="roman", size="x-small")
        plt.figtext(0.815, 0.045, " outliers", color="black", weight="roman", size="x-small")

        plt.savefig(filename, bbox_inches="tight", dpi=300)
