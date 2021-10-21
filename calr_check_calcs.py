import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import itertools


# from wide to long and concat files
def select_columns_by_metebolic_parm(df, param_name, exclude=False):
    if exclude == True:
        mask = ~df.columns.str.contains(pat=param_name)
        return df.loc[:, mask]
    mask = df.columns.str.contains(pat=param_name)
    return df.loc[:, mask]


def _get_columns_names_list(df):
    return df.columns.values.tolist()


def _make_dict_to_replace_names(columns_names_list, pattern_addition_to_parms):
    leng = len(columns_names_list)
    return {
        columns_names_list[i]:
        pattern_addition_to_parms + columns_names_list[i]
        for i in range(leng)
    }


def _get_actuals_values(df):
    df_actuals_features_calculeted = df.diff()
    first_row_df_cumuletive = df.iloc[0:1]
    return df_actuals_features_calculeted.fillna(first_row_df_cumuletive)


def pandas_dataframe_from_path(path, datetime_column_name):
    return pd.read_csv(path, date_parser=datetime_column_name)


def incal_get_actuals_from_cumuletive(df, columns_pattern,
                                      pattern_addition_to_parms):
    # get just the cumuletive columns from the original df

    df_cumuletive_culumns = select_columns_by_metebolic_parm(
        df, columns_pattern)
    # get the columns names of the cumuletive columns
    columns_names = _get_columns_names_list(df_cumuletive_culumns)
    # dict to replace names
    dict_new_names = _make_dict_to_replace_names(columns_names,
                                                 pattern_addition_to_parms)
    # replace the columns names of the actuals culumns
    df_actuals_features = df_cumuletive_culumns.rename(columns=dict_new_names)
    df_actuals = _get_actuals_values(df_actuals_features)
    return pd.concat([df, df_actuals], axis=1).drop(columns_names, axis=1)


def _right_sepert_first_underscore(string):
    return tuple(string.rsplit("_", 1))


def _assemble_multi_index_axis_1_df(df, d_list, axis_1_names=["", ""]):
    # make a multi index
    mul_i_columns = pd.MultiIndex.from_tuples(d_list, names=axis_1_names)
    # assemble new dataframe with multi index columns
    return pd.DataFrame(df.values, index=df.index, columns=mul_i_columns)
    # then stack level 1 to the columns (level 1 -> subjects names e.g. 1 2 3...)


def incal_wide_to_long_df(wide_df, col_subj_name='subjectID'):
    cols_names = _get_columns_names_list(wide_df)
    # sepert feature name from cage number and put it in a tuple together ('allmeters', '1')
    l_micolumns = [_right_sepert_first_underscore(col) for col in cols_names]
    multi_index_axis_1_df = _assemble_multi_index_axis_1_df(
        wide_df, l_micolumns, ['', col_subj_name])
    # https://pandas.pydata.org/docs/user_guide/reshaping.html
    return multi_index_axis_1_df.stack(level=1)


def flat_list(d_list):
    '''
    dependencies: itertools
    '''
    return list(itertools.chain.from_iterable(d_list))


def replace_ids_to_group_id(ndarray_ids, groups_names, subjects_within_group):
    conditiones = [
        ndarray_ids.astype('int64') == n for n in subjects_within_group
    ]
    choices = groups_names
    return np.select(conditiones, choices, ndarray_ids)


def incal_create_group_column_from_ids(ids_column, dict_groups):
    n_ids_multiple_name = lambda name, n: [name] * len(n)
    subjects_vlaues = ids_column.values
    items = dict_groups.items()
    groups_names = flat_list(
        [n_ids_multiple_name(group, ids) for group, ids in items])
    subjects_within_groups = flat_list([ids for ids in dict_groups.values()])
    return replace_ids_to_group_id(subjects_vlaues, groups_names,
                                   subjects_within_groups)


def start_incal_formatter(path, datetime_name, cumulative_parm,
                          pattern_addition_to_parms):
    df = pandas_dataframe_from_path(path, datetime_name)
    return incal_get_actuals_from_cumuletive(df, cumulative_parm,
                                             pattern_addition_to_parms)


#

if __name__ == '__main__':
    experiment_name = "maital"
    # 1. specific the location .csv file or fils in the list
    dataframes = [
        '_calr_Energy_Balance_check\csv\hebrew_2021-10-10_07_36_ido_accessdoors_1021_script1_w1_m_calr.csv',
    ]
    # 2. Specific the pattern of cumuletive column names
    cumulative_parm = "|".join(
        ['food', 'water', 'allmeters', 'wheelmeters', 'pedmeters'])
    date_time_column_name = 'Date_Time_1'
    # 3. Specific the prefix for the new actuals columns
    pattern_addition_to_parms = 'actual_'
    # 4. Specific the design experiment groups and subjects
    dict_groups = OrderedDict(Control=[1, 2, 3],
                              Group_2=[4, 5, 6],
                              Group_3=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    # dict_groups = OrderedDict(Control=[1, 4, 7, 10, 13],
    #                         Group_2=[3, 5, 9, 12, 16],
    #                         Group_3=[2, 6, 8, 11, 14, 15])
    # 5. run the script
    # --------
    # Creating the format
    df_groups = pd.DataFrame(dict_groups.values(), index=dict_groups.keys())

    dfs = [
        start_incal_formatter(df, date_time_column_name, cumulative_parm,
                              pattern_addition_to_parms) for df in dataframes
    ]
    dfs_concated = pd.concat(dfs).set_index(date_time_column_name)
    df = incal_wide_to_long_df(dfs_concated)
    index_datetime_subjects, df = df.index.to_frame().reset_index(
        drop=True), df.reset_index(drop=True)

    groupid = incal_create_group_column_from_ids(
        index_datetime_subjects['subjectID'], dict_groups)
    multi_index_df = pd.concat(
        [index_datetime_subjects,
         pd.Series(groupid, name='Group')], axis=1)
    df = pd.concat([multi_index_df, df], axis=1)

    columns_names_for_timeseries = {
        'actual_foodupa':
        'Hourly_Food_Intake_(kcal/hr)',
        'actual_waterupa':
        'Hourly_Water_Intake_(ml/hr)',
        'kcal_hr':
        'Energy_Expenditure_(kcal/hr)',
        'kcal_hr_cumu':
        'Cumulative_Energy_Expenditure_(kcal)',
        'actual_foodupa_cumu':
        'Cumulative_Food_Intake_(kcal)',
        'actual_pedmeters':
        'Pedestrian_Locomotion_(m)',
        'actual_allmeters':
        'Total_Distance_in_Cage_(m)',
        'vo2':
        'Oxygen_Consumption_(ml/hr)',
        'vco2':
        'Carbon_Dioxide_Production_(ml/hr)',
        'rq':
        'Respiratory_Exchange_Ratio',
        'bodymass':
        'Total_Mass_(g)',
        'Locomotor_Activity_(beam_breaks/hr)':
        'Locomotor_Activity_(beam_breaks/hr)'
    }
    {'Energy_Expenditure_(kcal)'}
    columns_names_for_averages = {
        'actual_pedmeters': 'Pedestrian_Locomotion_(m)',
        'actual_foodupa_kcal': 'Total_Food_(kcal)',
        'actual_waterupa': 'Total_Drink_(ml)',
        'kcal_hr': 'Energy_Expenditure_(kcal/hr)',
        'Energy_Balance_(kcal)': 'Energy_Balance_(kcal)',
        'Distance_in_Cage_Locomotion_(m)': 'Distance_in_Cage_Locomotion_(m)',
        'vo2': 'Oxygen_Consumption_(ml/hr)',
        'vco2': 'Carbon_Dioxide_Production_(ml/hr)',
        'rq': 'Respiratory_Exchange_Ratio',
        'bodymass': 'Total_Mass_(g)',
        'Locomotor_Activity_(beam_breaks/hr)':
        'Locomotor_Activity_(beam_breaks/hr)',
        'Average_Daily_Food_Intake_(kcal/period)':
        'Average_Daily_Food_Intake_(kcal/period)',
        'Average_Daily_Water_Intake_(kcal/period)':
        'Average_Daily_Water_Intake_(kcal/period)',
        'Average_Daily_Energy_Expenditure_(kcal/period)':
        'Average_Daily_Energy_Expenditure_(kcal/period)',
        'Average_Daily_Energy_Balance_(kcal/period)':
        'Average_Daily_Energy_Balance_(kcal/period)',
        'Average_Daily_Pedestrain_Locomotion_(m/period)':
        'Average_Daily_Pedestrain_Locomotion_(m/period)',
        'actual_allmeters': 'Average_Daily_Distance_in_Cage_(m/period)',
        'Wheel_Counts': 'Wheel_Counts'
    }

    aggs = {
        'Pedestrian_Locomotion_(m)': 'sum',
        'Total_Mass_(g)': 'mean',
        'Energy_Expenditure_(kcal)': 'sum',
        'Respiratory_Exchange_Ratio': 'mean',
        'Carbon_Dioxide_Production_(ml/hr)': 'sum',
        'Oxygen_Consumption_(ml/hr)': 'sum',
        'Average_Daily_Food_Intake_(kcal/period)': 'mean',
        'Total_Drink_(ml)': 'sum',
        'Average_Daily_Pedestrain_Locomotion_(m/period)': 'mean',
        'Energy_Balance_(kcal)': 'sum',
        'Average_Daily_Water_Intake_(kcal/period)': 'mean',
        'Average_Daily_Energy_Balance_(kcal/period)': 'mean',
        'Average_Daily_Distance_in_Cage_(m/period)': 'mean',
        'Total_Food_(kcal)': 'sum',
        'Average_Daily_Energy_Expenditure_(kcal/period)': 'mean',
        'Locomotor_Activity_(beam_breaks/hr)': 'sum',
        'Distance_in_Cage_Locomotion_(m)': 'sum',
        'Wheel_Counts': 'sum',
    }
    # add and calcs columns
    df[['vo2', 'vco2']] = df[['vo2', 'vco2']] * 60
    df['actual_foodupa_kcal'] = df['actual_foodupa'].mul(3.56)
    df['Average_Daily_Energy_Balance_(kcal/period)'] = \
      df['kcal_hr'] - df['actual_foodupa_kcal']
    df['Average_Daily_Energy_Expenditure_(kcal/period)'] = df['kcal_hr']
    df['Locomotor_Activity_(beam_breaks/hr)'] = \
      df['xbreak'].values + df['ybreak'].values
    df['Average_Daily_Pedestrain_Locomotion_(m/period)'] = \
      df['xbreak'].values + df['ybreak'].values + df['actual_pedmeters']
    df['Energy_Balance_(kcal)'] = df['kcal_hr'] - df['actual_foodupa_kcal']
    df['Average_Daily_Food_Intake_(kcal/period)'] = df['actual_foodupa_kcal']
    df['Average_Daily_Water_Intake_(kcal/period)'] = \
      df['actual_foodupa']  # kcal? in water
    df['Distance_in_Cage_Locomotion_(m)'] = df['actual_pedmeters']
    df['Wheel_Counts'] = df['actual_pedmeters']
    df['Energy_Expenditure_(kcal)'] = df['kcal_hr'] * 60
    df = df.drop(columns=[
        'xbreak',
        'ybreak',
        'vh2o',
        'actual_foodupa',
    ])
    averages = df.rename(columns=columns_names_for_averages)
    subjectID_order = flat_list(dict_groups.values())
    string_subjectID_order = [str(i) for i in subjectID_order]
    averages = averages.groupby('subjectID').agg(aggs)
    averages = averages.reindex(string_subjectID_order)
    # upload calr file to comper my calcs
    calr = pd.read_csv(
        '_calr_Energy_Balance_check\calr_output\overall_average_big_axample.csv'
    )
    mask = calr['Time of Day'] == 'Full day'

    calr_full_day = calr[mask]
    calr_full_day = calr_full_day.drop(
        columns=['Unnamed: 0', 'Time of Day', 'Group', 'NA'])
    calr_full_day.columns = [
        name.replace(' ', '_') for name in calr_full_day.columns
    ]
    calr_full_day = calr_full_day.set_index('Subject_ID')
    calr_full_day = calr_full_day.reindex(subjectID_order)

    calr_columns_names = set(calr_full_day.columns)
    incal_columns_names = set(averages.columns)
    all_columns_are_same = not calr_columns_names - incal_columns_names

    # reorder the columns position
    calr_full_day = calr_full_day[averages.columns]
    comper = pd.DataFrame(averages.values / calr_full_day.values,
                          columns=calr_full_day.columns,
                          index=calr_full_day.index)
    comper.to_csv('_calr_Energy_Balance_check/comper.csv')

    # print(incal_columns_names - calr_columns_names)
