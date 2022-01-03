import pandas as pd
import numpy as np
import datetime as dt
import itertools


if __name__ == '__main__':
    # import data
    data = get_dataframe()
    # check types of data
    idx = data.index.get_level_values
    # print(idx(0), idx(1), idx(2))

    # data cleaning
    times_of_expriment = custome_days(idx(0))
    data['locomotion'] = data['xbreak'] + data['ybreak']
    data['kcal_mean'] = data['kcal_hr']
    data.rename(columns={'kcal_hr': 'kcal'}, inplace=True)
    # ANALYSIS:

    # hourly avg and sums - for all AGGS KEYS
    design_expriment = [
        ['Control', [1, 4, 7, 10, 13]],
        ['Six_Meal', [3, 5, 9, 12, 16]],
        ['Three_Meal',	[2, 6, 8, 11, 14, 15]]]

    data_with_times_and_groups_names = make_hourly_avg_and_sums(
        data, times_of_expriment)
    data_with_times_and_groups_names.to_csv(
        'csvs/shani/hourly/hourly_avg_and_sums.csv')
