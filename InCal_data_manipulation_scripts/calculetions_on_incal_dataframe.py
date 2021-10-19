import numpy as np
import pandas as pd


def multiple(dataframe, value, *args):
    return dataframe[[*args]].mul(value)


def substruct_two_features(dataframe, feature1_name, feature2_name):
    return dataframe[feature1_name].values - dataframe[feature2_name].values


if __name__ == '__main__':

    path = "csvs/all_weeks_shani_exp.csv"
    # DataFrmae
    df = pd.read_csv(path)

    # CHANGE UNITS
    # Add feature -> actual_foodupa_kcal (kcal)
    df[['actual_foodupa_kcal']] = multiple(df, 3.56, 'actual_foodupa')
    # 'vco2', 'vh2o', 'vo2': from ml/min => ml/hr
    df[['vco2', 'vh2o', 'vo2']] = multiple(df, 60, 'vco2', 'vh2o', 'vo2')

    # CALCULATING LOCOMOTORE
    df['locomotor_activity'] = df[['xbreak', 'ybreak']].sum(axis=1)

    # ENERGY BALNCE CALCULETION
    df['Energy_Balance'] = substruct_two_features(df, 'actual_foodupa_kcal',
                                                  'kcal_hr')
