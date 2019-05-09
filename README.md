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

### usage

First, collecting data for training and testing the two approaches:

* create dataset ``malaise`` with gendataset.sh in [ca-eval](https://github.com/dianagudu/ca-eval)
* collect runtime stats of the algorithms in [ca-portfolio](https://github.com/dianagudu/ca-portfolio) on this dataset, using splittasks.sh in [ca-eval](https://github.com/dianagudu/ca-eval)
* collect runtime stats of the algorithms on samples of each auction instance in the dataset

Running the algorithm selection workflows:

    python -m cause