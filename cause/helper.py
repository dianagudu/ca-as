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
    HILL1S = 5
    HILL2 = 6
    HILL2S = 7
    SA = 8
    SAS = 9
    CASANOVA = 10
    CASANOVAS = 11


class Stochastic_Algorithm_Names(Enum):
    HILL2 = 6
    HILL2S = 7
    SA = 8
    SAS = 9
    CASANOVA = 10
    CASANOVAS = 11


feature_names = [
            ### group 1: instance - price related
              'average_bid_price_mean'                                    # 0
            , 'average_bid_price_stddev'                                  # 1
            , 'average_bid_price_skewness'                                # 2
            , 'average_bid_price_kurtosis'                                # 3
            , 'average_ask_price_mean'                                    # 4
            , 'average_ask_price_stddev'                                  # 5
            , 'average_ask_price_skewness'                                # 6
            , 'average_ask_price_kurtosis'                                # 7
            , 'average_bid_price_max'                                     # 8
            , 'average_ask_price_min'                                     # 9
            , 'mid_price'                                                 # 10
            , 'bid_ask_spread'                                            # 11
            , 'bid_ask_spread_over_mid_price'                             # 12
            ### group 2: instance - quantity related
            , 'bid_bundle_size_mean'                                      # 13
            , 'bid_bundle_size_stddev'                                    # 14
            , 'bid_bundle_size_skewness'                                  # 15
            , 'bid_bundle_size_kurtosis'                                  # 16
            , 'ask_bundle_size_mean'                                      # 17
            , 'ask_bundle_size_stddev'                                    # 18
            , 'ask_bundle_size_skewness'                                  # 19
            , 'ask_bundle_size_kurtosis'                                  # 20
            ### group 3: instance - quantity per resource related (measure of heterogeneity)
            , 'total_demand_per_resource_mean'                            # 21
            , 'total_demand_per_resource_stddev'                          # 22
            , 'total_demand_per_resource_skewness'                        # 23
            , 'total_demand_per_resource_kurtosis'                        # 24
            , 'average_demand_per_resource_mean'                          # 25
            , 'average_demand_per_resource_stddev'                        # 26
            , 'average_demand_per_resource_skewness'                      # 27
            , 'average_demand_per_resource_kurtosis'                      # 28
            , 'minimum_demand_per_resource_mean'                          # 29
            , 'minimum_demand_per_resource_stddev'                        # 30
            , 'minimum_demand_per_resource_skewness'                      # 31
            , 'minimum_demand_per_resource_kurtosis'                      # 32
            , 'maximum_demand_per_resource_mean'                          # 33
            , 'maximum_demand_per_resource_stddev'                        # 34
            , 'maximum_demand_per_resource_skewness'                      # 35
            , 'maximum_demand_per_resource_kurtosis'                      # 36
            # --> supply side
            , 'total_supply_per_resource_mean'                            # 37
            , 'total_supply_per_resource_stddev'                          # 38
            , 'total_supply_per_resource_skewness'                        # 39
            , 'total_supply_per_resource_kurtosis'                        # 40
            , 'average_supply_per_resource_mean'                          # 41
            , 'average_supply_per_resource_stddev'                        # 42
            , 'average_supply_per_resource_skewness'                      # 43
            , 'average_supply_per_resource_kurtosis'                      # 44
            , 'minimum_supply_per_resource_mean'                          # 45
            , 'minimum_supply_per_resource_stddev'                        # 46
            , 'minimum_supply_per_resource_skewness'                      # 47
            , 'minimum_supply_per_resource_kurtosis'                      # 48
            , 'maximum_supply_per_resource_mean'                          # 49
            , 'maximum_supply_per_resource_stddev'                        # 50
            , 'maximum_supply_per_resource_skewness'                      # 51
            , 'maximum_supply_per_resource_kurtosis'                      # 52
            ### group 4: instance - demand-supply balance related
            , 'surplus_value_per_surplus_unit'                            # 53
            , 'demand_supply_ratio_value'                                 # 54
            , 'demand_supply_ratio_quantity'                              # 55
            , 'demand_supply_ratio_total_quantity_per_resource_mean'      # 56
            , 'demand_supply_ratio_total_quantity_per_resource_stddev'    # 57
            , 'demand_supply_ratio_total_quantity_per_resource_skewness'  # 58
            , 'demand_supply_ratio_total_quantity_per_resource_kurtosis'  # 59
            , 'demand_supply_ratio_mean_quantity_per_resource_mean'       # 60
            , 'demand_supply_ratio_mean_quantity_per_resource_stddev'     # 61
            , 'demand_supply_ratio_mean_quantity_per_resource_skewness'   # 62
            , 'demand_supply_ratio_mean_quantity_per_resource_kurtosis'   # 63
            , 'surplus_quantity'                                          # 64
            , 'surplus_total_quantity_per_resource_mean'                  # 65
            , 'surplus_total_quantity_per_resource_stddev'                # 66
            , 'surplus_total_quantity_per_resource_skewness'              # 67
            , 'surplus_total_quantity_per_resource_kurtosis'              # 68
            , 'quantity_spread_per_resource_mean'                         # 69
            , 'quantity_spread_per_resource_stddev'                       # 70
            , 'quantity_spread_per_resource_skewness'                     # 71
            , 'quantity_spread_per_resource_kurtosis'                     # 72
            , 'ratio_average_price_bid_to_ask'                            # 73
            , 'ratio_bundle_size_bid_to_ask'                              # 74
            ### group 4: instance - critical values related
            #, 'critical_density_bids'  # 43
            #, 'critical_density_asks'  # 44
            #, 'critical_price_bids'    # 45
            #, 'critical_price_asks'    # 46
            ]
