import numpy as np
import pandas as pd
import pylab as plt
from matplotlib import cm
from matplotlib import rc
import seaborn as sns


class Plotter():

    @staticmethod
    def __set_rc_params(fontsize=14):
        rc("text", usetex=False)
        rc("mathtext", fontset="custom")
        rc("mathtext", default="regular")
        rc("font", **{"family": "serif",
                      "serif": ["EB Garamond"],
                      "size": fontsize})

    @staticmethod
    def plot_breakdown(breakdown, outfolder="/tmp"):
        Plotter.__set_rc_params(18)
        f, ax = plt.subplots(figsize=(16, 9))
        snsplot = sns.heatmap(breakdown.data, cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0.2, as_cmap=True)  # , square=True
                              , ax=ax, annot=True, fmt="d", annot_kws={"fontsize": 16}, cbar=True
                              )
        snsplot.set_xticklabels(breakdown.weights, rotation=0, fontsize=18)
        snsplot.set_yticklabels(
            breakdown.algos, rotation=0, fontsize=18)
        snsplot.set_xlabel("$\lambda$", fontsize=18)
        # increase fontsize for colorbar
        ax.collections[0].colorbar.ax.tick_params(labelsize=16)
        outfile = outfolder + "/breakdown_" + breakdown.name + ".png"
        snsplot.get_figure().savefig(outfile, bbox_inches="tight", dpi=300)

    @staticmethod
    def boxplot_average_case(data, algos, filename,
                               bottom=-10, top=110,
                               ylog=False,
                               ylabel="\% of optimal"):
        # reset seaborn settings
        sns.reset_orig()
        Plotter.__set_rc_params()
        fig, ax1 = plt.subplots(figsize=(8, 5))
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        bp = plt.boxplot(data, notch=1, vert=1, whis=[5, 95],
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
        ax1.xaxis.grid(
            True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
        ax1.yaxis.grid(
            True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
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
        xtickNames = plt.setp(ax1, xticklabels=algos)
        plt.setp(xtickNames, rotation=45, fontsize=10)
        # Finally, add a basic legend
        plt.figtext(
            0.795, 0.09, "-", color="blue", weight="roman", size="medium")
        plt.figtext(0.815, 0.092, " median value",
                    color="black", weight="roman", size="x-small")
        plt.figtext(
            0.795, 0.058, "*", color="red", weight="roman", size="medium")
        plt.figtext(0.815, 0.07, " average value",
                    color="black", weight="roman", size="x-small")
        plt.figtext(0.7965, 0.045, "o",
                    color="grey", weight="roman", size="x-small")
        plt.figtext(0.815, 0.045, " outliers",
                    color="black", weight="roman", size="x-small")

        plt.savefig(filename, bbox_inches="tight", dpi=300)

    @staticmethod
    def boxplot_random(data, algos, outfile):
        # reset seaborn settings
        sns.reset_orig()
        Plotter.__set_rc_params()
        fig, ax1 = plt.subplots(figsize=(8, 5))
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        bp = plt.boxplot(data, notch=1, vert=False, whis=[5, 95],
                         bootstrap=10, showmeans=True, showfliers=True)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='grey',
                 marker='.', mew=0.5, mec='grey', markersize=3.5)
        plt.setp(bp['means'], color='red', marker='*', mec='red', mfc='red')
        plt.setp(bp['medians'], color='blue')
        ax1.xaxis.grid(
            True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.yaxis.grid(
            True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        # hide grid behind plot objects
        ax1.set_axisbelow(True)
        ytickNames = plt.setp(ax1, yticklabels=algos)
        ax1.set_xlim(-100, 100)
        plt.xlabel("difference to mean welfare (%)")
        # finally, add a basic legend
        plt.figtext(
            0.795, 0.13, '-', color='blue', weight='roman', size='medium')
        plt.figtext(0.815, 0.132, ' median value',
                    color='black', weight='roman', size='small')
        plt.figtext(
            0.795, 0.098, '*', color='red', weight='roman', size='medium')
        plt.figtext(0.815, 0.11, ' average value',
                    color='black', weight='roman', size='small')
        plt.figtext(0.7965, 0.085, 'o',
                    color='grey', weight='roman', size='small')
        plt.figtext(0.815, 0.085, ' outliers',
                    color='black', weight='roman', size='small')
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

        for i in range(len(algos)):
            print(ytickNames[i], bp['boxes'][i].get_xdata())
        for ws in bp['whiskers']:
            print(ws.get_xdata())
