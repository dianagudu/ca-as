#!/usr/bin/env python3
import click
import numpy as np

from cause.preprocessor import RawStatsLoader
from cause.preprocessor import FeatureExtractor
from cause.preprocessor import DatasetCreator
from cause.preprocessor import SamplesDatasetCreator

from cause.stats import ProcessedDataset

from cause.features import Features

from cause.postprocessor import Postprocessor
from cause.postprocessor import FeatsPostprocessor
from cause.postprocessor import MALAISEPostprocessor
from cause.postprocessor import PRAISEPostprocessor


def validate_weights(ctx, param, value):
    # domain = None --> return None
    # domain != None --> validate domain: list of floats>0

    if value is None:
        return None

    def positive_float(f):
        if float(f) <= 0:
            raise ValueError(None)
        else:
            return float(f)

    try:
        return np.array([positive_float(x) for x in value.split(",")])
    except ValueError:
        raise click.BadParameter('%s should be a comma-separated list of floats > 0, not \'%s\'' % (param.name, value))


@click.group()
def cli():
    pass


# compare all
# compare stochastic

@cli.group(short_help='subcommand to compare algorithms', name='compare')
def compare():
    pass


@compare.command(short_help='compare all algorithms w.r.t optimal', name='all')
@click.argument("name")
@click.argument("infolder", type=click.Path(exists=True))
@click.argument("oufolder", type=click.Path(exists=True))
def cmp_all(name, infolder, outfolder):
    """Loads raw stats in INFOLDER obtained by running the portfolio on
    the dataset NAME and then plots time and welfare of each algorithm w.r.t.
    the time and welfare of the optimal algorithm.
    The plots are saved to OUTFOLDER.
    """
    RawStatsLoader(infolder, name).load_optimal().plot(outfolder)


@compare.command(short_help='compare all stochastic algorithms', name='stochastic')
@click.argument("name")
@click.argument("infolder", type=click.Path(exists=True))
@click.argument("oufolder", type=click.Path(exists=True))
def cmp_rand(name, infolder, outfolder):
    """Loads raw stats in INFOLDER obtained by running the portfolio on
    the dataset NAME and then plots welfare of stochastic algorithms over
    multiple runs, w.r.t. the average value for each instance.
    The plots are saved to OUTFOLDER.
    """
    RawStatsLoader(infolder, name).load_random().plot(outfolder)


# preprocess features
# preprocess stats
# preprocess samples

@cli.group(short_help='subcommand to preprocess collected raw data', name='preprocess')
def preprocess():
    pass


@preprocess.command(short_help='extract features from auction instances', name='features')
@click.argument("name")
@click.argument("infolder", type=click.Path(exists=True))
@click.argument("oufolder", type=click.Path(exists=True))
def preproc_features(instance_folder, name, outfolder):
    """Extracts features of auction instances in INFOLDER comprising dataset NAME
    and saves them to OUTFOLDER using the given name.
    """
    FeatureExtractor.extract(instance_folder, name, outfolder)


@preprocess.command(short_help='extract and compute, then save algorithm stats on full instances',
                    name='stats')
@click.option("--weights", callback=validate_weights,
              default='0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.',
              help='lambda weights for the cost model (float or list of floats)')
@click.argument("name")
@click.argument("infolder", type=click.Path(exists=True))
@click.argument("oufolder", type=click.Path(exists=True))
def preproc_stats(name, weights, infolder, outfolder):
    """Processes raw stats in INFOLDER obtained by running the heuristic algorithms
    in algorithm portfolio on dataset NAME, and saves them to OUTFOLDER
    using the given name.

    The weights used in the cost model to determine the best algorithm are passed
    as a comma-separated list of positive floats (or a single value), with the
    default value: '0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.'.
    """
    DatasetCreator.create(weights, infolder, outfolder, name)


@preprocess.command(short_help='extract and compute, then save algorithm stats on instance samples',
                    name='samples')
@click.option("--weights", callback=validate_weights,
              default='0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.',
              help='lambda weights for the cost model (float or list of floats)')
@click.argument("name")
@click.argument("infolder", type=click.Path(exists=True))
@click.argument("oufolder", type=click.Path(exists=True))
def preproc_samples(name, weights, infolder, outfolder):
    """Processes raw stats in INFOLDER obtained by running the heuristic algorithms
    in algorithm portfolio on samples of each instance in the dataset NAME,
    and saves them to OUTFOLDER using the given name.

    The weights used in the cost model to determine the best algorithm are passed
    as a comma-separated list of positive floats (or a single value), with the
    default value: '0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.'.
    """
    SamplesDatasetCreator.create(weights, infolder, outfolder, name)


# run malaise
# run praise
# run fitting

@cli.group(short_help='subcommand to run algorithm selection', name='run')
def run():
    pass


@run.command(short_help='run MALAISE', name='malaise')
def run_malaise():
    pass


@run.command(short_help='run PRAISE', name='praise')
def run_praise():
    pass


@run.command(short_help='curve fitting for scaling behavior over problem size',
             name='fitting')
def run_fitting():
    pass


# postprocess breakdown
# postprocess features
# postprocess malaise
# postprocess praise

@cli.group(short_help='subcommand to postprocess algorithm selection results',
           name='postprocess')
def postprocess():
    pass


@postprocess.command(short_help='get dataset breakdown by best algorithm', name='breakdown')
def postprocess_breakdown(name, outfolder):
    # load processed dataset
    ds = ProcessedDataset.load("%s/%s.yaml" % (outfolder, name))
    ## get breakdown by algorithms and weights
    breakdown = Postprocessor(ds).breakdown()
    ## save to file for latex table
    breakdown.save_to_latex(outfolder)
    ## plot breakdown as heatmap
    breakdown.plot(outfolder)


@postprocess.command(short_help='get feature importances', name='features')
def postprocess_features(name, outfolder):
    # load processed features
    feats = Features.load("%s/%s_features.yaml" % (outfolder, name))
    # load processed stats
    ds = ProcessedDataset.load("%s/%s.yaml" % (outfolder, name))
    ## postprocessing: feature importances
    fpostp = FeatsPostprocessor(ds, feats)
    fpostp.save_feature_importances(outfolder)
    for weight in ds.weights:
        fpostp.save_feature_importances_by_weight(outfolder, weight)
    ## plot features as heatmap
    feats.plot(outfolder)


@postprocess.command(short_help='postprocess MALAISE results', name='malaise')
def postprocess_malaise(name, outfolder):
    MALAISEPostprocessor("%s/%s_stats" % (outfolder, name)).save(outfolder)


@postprocess.command(short_help='postprocess PRAISE results', name='praise')
def postprocess_praise(name, outfolder):
    PRAISEPostprocessor("%s/%s_stats" % (outfolder, name)).save(outfolder)


if __name__ == '__main__':
    cli()
