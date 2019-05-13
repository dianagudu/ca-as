# ca-as
Algorithm selection for combinatorial auctions

### description

This repository implements two approaches for algorithm selection:

* MALAISE (MAchine Learning-based AlgorIthm SElection), presented at Euro-Par 2018<sup>1</sup>
* PRAISE (PRobing-based AlgorIthm SElection)

<sup>1</sup> Gudu, Diana, Marcus Hardt, and Achim Streit. "Combinatorial Auction Algorithm Selection for Cloud Resource Allocation Using Machine Learning." In European Conference on Parallel Processing, pp. 378-391. Springer, Cham, 2018. [doi:10.1007/978-3-319-96983-1_27](https://doi.org/10.1007/978-3-319-96983-1_27)

### prerequisites

* all libs in [requirements.txt](requirements.txt)
* [auto-sklearn](https://automl.github.io/auto-sklearn/stable/) v0.5.0
* [CAGE](https://github.com/dianagudu/ca-ingen): input generator for combinatorial auctions
    * download and add to PYTHONPATH
* data for training and testing the two approaches (see [ca-eval](https://github.com/dianagudu/ca-eval))

Collecting data for training and testing the two approaches:

* create dataset ``malaise`` (comprising auction instances) with gendataset.sh in [ca-eval](https://github.com/dianagudu/ca-eval)
* collect runtime stats of the algorithms in [ca-portfolio](https://github.com/dianagudu/ca-portfolio) on this dataset, using splittasks.sh in [ca-eval](https://github.com/dianagudu/ca-eval)
* collect runtime stats of the algorithms on samples of each auction instance in the dataset

### usage


A command line tool is provided, with help and description of parameters included. Check out all options with:

    python causecli.py --help

    Usage: causecli.py [OPTIONS] COMMAND [ARGS]...

    Options:
      --help  Show this message and exit.

    Commands:
      compare      subcommand to compare algorithms
      postprocess  subcommand to postprocess algorithm selection results
      preprocess   subcommand to preprocess collected raw data
      run          subcommand to run algorithm selection

Help is also available for each command:

    python causecli.py run malaise --help

    Usage: causecli.py run malaise [OPTIONS] NAME INFOLDER OUTFOLDER

      Runs MALAISE algorithm selection on dataset NAME, with processed stats and
      extracted features located in INFOLDER. The trained models and evaluation
      results (accuracy and MSE error) are saved to OUTFOLDER, as well as
      results for random selection and best algorithm selection, to be used for
      comparison.

      The training uses auto-ml to find the best hyperparameters, and it can be
      done in parallel if nthreads are given.

      The weights used in the cost model to determine the best algorithm are
      passed as a comma-separated list of positive floats (or a single value),
      with the default value: '0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.'.

    Options:
      --weights TEXT      lambda weights for the cost model (float or list of
                          floats)
      --nthreads INTEGER  number of threads for parallel training with auto-
                          sklearn
      --help              Show this message and exit.


The examples below assume the following folder structure, with auction datasets stored in ``datasets`` and raw stats stored in ``stats``. The intermediate data obtained in the preprocessing step is stored in ``processed``, while the results (which can be plots or text files) are stored in ``output``.

    ~/ca/
        |_ datasets/
        |       |_ ca-compare-3dims/
        |       |_ malaise/
        |_ stats/
        |       |_ ca-compare-3dims/
        |       |_ malaise/
        |       |_ praise/
        |_ processed/
        |       |_ ca-compare-3dims/
        |       |_ malaise/
        |       |_ praise/
        |_ output/
                |_ ca-compare-3dims/
                |_ malaise/
                |_ praise/
    

#### preprocessing

Preprocessing consists of mining the raw stats to extract relevant stats (time and welfare for each algorithm on each auction instance), and compute costs and winning algorithms for any given lambda weights. The same instances are used for MALAISE and PRAISE.

    python causecli.py preprocess stats malaise ~/ca/stats/malaise ~/ca/processed/malaise
    python causecli.py preprocess stats praise ~/ca/stats/malaise ~/ca/processed/praise

For MALAISE, features should be extracted from each auction instance in the dataset:

    python causecli.py preprocess features malaise ~/ca/datasets/malaise ~/ca/processed/malaise

For PRAISE, the raw stats of the algorithms on samples of each instance are processed to extrapolate time and welfare from sample data, ad compute corresponding time overhead:

    python causecli.py preprocess samples praise ~/ca/stats/praise ~/ca/processed/praise

#### selection

MALAISE algorithm selection:

    python causecli.py run malaise malaise ~/ca/processed/malaise ~/ca/output/malaise

PRAISE algorithm selection:

    python causecli.py run praise praise ~/ca/processed/praise ~/ca/output/praise

#### postprocessing

Processing the prediction results to compute per-lambda accuracy and RMSE (relative mean squared error) compared to random selection and single best algorithm, for each approach:

    python causecli.py postprocess malaise malaise ~/ca/output/malaise ~/ca/output/malaise
    python causecli.py postprocess praise praise ~/ca/output/praise ~/ca/output/praise

Feature importances and correlations can be explored through:

    python causecli.py postprocess features malaise ~/ca/processed/malaise ~/ca/output/malaise

Breaking down the dataset into classes based on the best algorithm:

    python causecli.py postprocess breakdown malaise ~/ca/processed/malaise ~/ca/output/malaise

#### misc

Scaling behavior of each algorithm on can be investigated by:

    python causecli.py run fitting praise ~/ca/processed/praise

The tool also supports algorithm comparison w.r.t. the optimal algorithm, as well as effect of randomness on welfare for stochastic algorithms (we assume raw stats of portfolio on dataset ``ca-compare-3dims`` are stored in ``~/ca/stats/ca-compare-3dims``):

    python causecli.py compare all ca-compare-3dims ~/ca/stats/ca-compare-3dims ~/ca/output/ca-compare-3dims
    python causecli.py compare stochastic ca-compare-3dims ~/ca/stats/ca-compare-3dims ~/ca/output/ca-compare-3dims
