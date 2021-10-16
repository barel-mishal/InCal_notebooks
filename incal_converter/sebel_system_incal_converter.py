import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import itertools


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


if __name__ == '__main__':
    experiment_name = "_shani"
    dataframes = [
        pd.read_csv(
            "../../csvs/all_weeks\hebrew_2021-07-28_16_33_hebrew16_shani_w2_acdoors_pt1_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks\hebrew_2021-08-01_13_49_hebrew16_shani_w1_pt2b_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks/hebrew_2021-08-04_11_45_hebrew16_shani_acdoors_w2p1_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks/hebrew_2021-08-10_16_15_hebrew16_shani_w2p2.1_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks/hebrew_2021-08-11_16_24_hebrew16_shani_acdoors_w3_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks/hebrew_2021-08-19_16_17_hebrew16_shani_acdoors_w4_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks/hebrew_2021-08-15_16_24_hebrew16_sahni_acdoors_w3p2_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks/hebrew_2021-08-26_16_12_hebrew16_shani_acdoors_w5_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks/hebrew_2021-08-29_08_41_hebrew16_shani_acdoors_w5_dd_m_calr.csv",
            parse_dates=['Date_Time_1']),
        pd.read_csv(
            "../../csvs/all_weeks/hebrew_2021-09-02_07_54_hebrew16_dark dark week2_m_calr.csv",
            parse_dates=['Date_Time_1'])
    ]
    dict_groups = OrderedDict(Control=[1, 4, 7, 10, 13],
                              Group_2=[3, 5, 9, 12, 16],
                              Group_3=[2, 6, 8, 11, 14, 15])

    is_one_file = len(dataframes) == 1
    cumulative_parm = "|".join(
        ['food', 'water', 'allmeters', 'wheelmeters', 'pedmeters'])
    pattern_addition_to_parms = 'actual_'

    df_or_dfs_in_list = dataframes
    dfs = [
        incal_get_actuals_from_cumuletive(df, cumulative_parm,
                                          pattern_addition_to_parms)
        for df in df_or_dfs_in_list
    ]
    dfs_concated = pd.concat(dfs).set_index('Date_Time_1')
    df = incal_wide_to_long_df(dfs_concated)
    index_datetime_subjects, df = df.index.to_frame().reset_index(
        drop=True), df.reset_index(drop=True)

    groupid = incal_create_group_column_from_ids(
        index_datetime_subjects['subjectID'], dict_groups)
    multi_index_df = pd.concat(
        [index_datetime_subjects,
         pd.Series(groupid, name='Group')], axis=1)
    df = pd.concat([multi_index_df, df], axis=1)

    print()
    print(
        f'Check for file name (InCal_format_{experiment_name}.csv) at the root of your local env'
    )
    print()
    df.to_csv(f'InCal_format_{experiment_name}.csv', index=False)
