import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import itertools


def flat_list(d_list):
    '''
    dependencies: itertools
    '''
    return list(itertools.chain.from_iterable(d_list))


def incal_create_df_incal_format(df_data_experiment, df_design_expriment):
    df = df_data_experiment.copy()
    groups_names = df_design_expriment.values[:, 0]
    subjects_ids = list(
        filter(lambda x: ~np.isnan(x),
               (flat_list(df_design_expriment.values[:, 1:]))))

    date_time_level = pd.Series((pd.DatetimeIndex(df['Date_Time_1'])),
                                name='Date_Time_1')
    subjects_level = pd.Series(pd.Categorical(df['subjectID'],
                                              categories=categories_subjects,
                                              ordered=True),
                               name='subjectsID')
    group_level = pd.Series(pd.Categorical(df['Group'],
                                           categories=categories_groups,
                                           ordered=True),
                            name='Group')

    df = df.drop(columns=['Date_Time_1', 'subjectID', 'Group'])

    multi_index_dataframe = pd.concat(
        [date_time_level, subjects_level, group_level], axis=1)

    return pd.DataFrame(df.values,
                        index=pd.MultiIndex.from_frame(multi_index_dataframe),
                        columns=df.columns.values.tolist())