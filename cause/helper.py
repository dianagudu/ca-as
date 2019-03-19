from enum import Enum


class Algorithm_Names(Enum):
    GREEDY1 = 0
    GREEDY2 = 1
    GREEDY3 = 2
    GREEDY1S = 3
    HILL1 = 4
    HILL1S = 5
    HILL2 = 6
    HILL2S = 7
    SA = 8
    SAS = 9
    CASANOVA = 10
    CASANOVAS = 11
    CPLEX = 12
    RLPS = 13


class Heuristic_Algorithm_Names(Enum):
    GREEDY1 = 0
    GREEDY2 = 1
    GREEDY3 = 2
    GREEDY1S = 3
    HILL1 = 4
    HILL1s = 5
    HILL2 = 6
    HILL2s = 7
    SA = 8
    SAS = 9
    CASANOVA = 10
    CASANOVAS = 11


class Stochastic_Algorithm_Names(Enum):
    HILL2 = 6
    HILL2s = 7
    SA = 8
    SAS = 9
    CASANOVA = 10
    CASANOVAS = 11
