import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import itertools
from IPython import display
import functools
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import itertools
import datetime as dt
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import itertools


def plotly_graph_how_to_create_a_graph_that_has_black_and_wight():
    average_Group_analysis_df = analysis_df.groupby(level=[0, 2]).mean()
    date_time_frame = average_Group_analysis_df.index.get_level_values(
        0).to_frame().reset_index(drop=True)
    list_of_start_and_end_days = make_lists_start_and_end_to_day_night_time(
        date_time_frame)
    print(len(list_of_start_and_end_days)/2)

    fig = px.scatter(
        x=average_Group_analysis_df.index.get_level_values(0),
        y=average_Group_analysis_df.actual_foodupa,
        color=average_Group_analysis_df.index.get_level_values(1),
        labels={
            'x': average_Group_analysis_df.index.get_level_values(0).name,
            'y': average_Group_analysis_df.actual_foodupa.name,
            'color': average_Group_analysis_df.index.get_level_values(1).name
        },
    )

    fig.update_traces(mode='lines+markers')

    for i in range(1, len(list_of_start_and_end_days), 2):
        fig.add_vrect(
            x0=list_of_start_and_end_days[i - 1], x1=list_of_start_and_end_days[i], fillcolor="rgba(200, 200, 200, 0.3)")
        fig.update_layout()
        fig.show()


def multiple(dataframe, value, *args):
    return dataframe[[*args]].mul(value)


def substruct_two_features(dataframe, feature1_name, feature2_name):
    return dataframe[feature1_name].values - dataframe[feature2_name].values


def flat_list(d_list):
    '''
    dependencies: itertools
    '''
    return list(itertools.chain.from_iterable(d_list))


def incal_create_df_incal_format(df_data_experiment, df_design_expriment):
    df = df_data_experiment.copy()
    categories_groups = df_design_expriment.values[:, 0]
    categories_subjects = list(
        filter(lambda x: ~np.isnan(x),
               (flat_list(df_design_expriment.values[:, 1:]))))
    date_time_level = pd.Series((pd.DatetimeIndex(df['Date_Time_1'])),
                                name='Date_Time_1')
    subjects_level = pd.Series(pd.Categorical(df['subjectsID'],
                                              categories=categories_subjects,
                                              ordered=True),
                               name='subjectsID')
    group_level = pd.Series(pd.Categorical(df['Group'],
                                           categories=categories_groups,
                                           ordered=True),
                            name='Group')
    df = df.drop(columns=['Date_Time_1', 'subjectsID', 'Group'])
    multi_index_dataframe = pd.concat(
        [date_time_level, subjects_level, group_level], axis=1)
    return pd.DataFrame(df.values,
                        index=pd.MultiIndex.from_frame(multi_index_dataframe),
                        columns=df.columns.values.tolist())


def count_custome_day_start_and_end(pd_index_datetime, shift_time='8:30am'):
    dates = sorted(set(pd_index_datetime.date))
    times = np.array([time.time() for time in pd_index_datetime
                      ])  # there maybe a better method
    shift_time = pd.to_datetime(shift_time).time()
    conditiones = [
        logic for date in dates for logic in [
            ((times < shift_time)
             & (pd_index_datetime.date == date)),  # day before
            ((times >= shift_time)
             & (pd_index_datetime.date == date))
        ]  # day after
    ]
    langth = (len(conditiones) // 2) + 1
    choices = [0, *np.repeat(range(1, langth), 2)]
    choices.pop()
    return np.select(conditiones, choices, pd_index_datetime.factorize()[0])


def day_and_night(pd_index_datetime, start='08:30', end='16:30'):
    times = np.array([time.time() for time in pd_index_datetime])

    greater = pd.to_datetime(start).time() <= times
    stricly_less = pd.to_datetime(end).time() > times
    return np.where((greater & stricly_less), 'Day', 'Night')


# calc agg
def get_wide_format(data, levels=[1, 2]):
    return data.unstack(levels)


def get_groupby_each_day(data, level=0):
    return data.groupby(level=level)


def get_agg_for_each(data, categories_levels=[1, 2], level=0, aggs='mean'):
    return data.pipe(get_wide_format,
                     categories_levels).pipe(get_groupby_each_day,
                                             level).agg(aggs)


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
    def n_ids_multiple_name(name, n): return [name] * len(n)
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


def flat_list(d_list):
    '''
    dependencies: itertools
    '''
    return list(itertools.chain.from_iterable(d_list))


def incal_create_df_incal_format(df_data_experiment, df_design_expriment):
    df = df_data_experiment.copy()
    categories_groups = df_design_expriment.values[:, 0]
    categories_subjects = list(
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


def each_meal_analysis(df):
    three_meals = df.loc[:, :, ['Group_3']]
    pd_index_datetime = three_meals.index.get_level_values(0)
    meals_times_indexs_three_meal = mark_between_times(pd_index_datetime,
                                                       meals_3meals_buffer)
    get_level = three_meals.index.get_level_values
    agg_three_meals = three_meals['actual_foodupa'].groupby([
        meals_times_indexs_three_meal,
        get_level(1),
        pd.Grouper(level=3, freq='D')
    ]).sum()

    group3 = agg_three_meals.loc[:, [2, 6, 8, 11, 14, 15]].copy()
    #
    six_meals = df.loc[:, :, ['Group_2']]

    pd_index_datetime = six_meals.index.get_level_values(0)
    meals_times_indexs_six_meals = mark_between_times(pd_index_datetime,
                                                      meals_6meals_buffer)
    get_level = six_meals.index.get_level_values

    agg_six_meals = six_meals['actual_foodupa'].groupby([
        meals_times_indexs_six_meals,
        get_level(1),
        pd.Grouper(level=3, freq='D')
    ]).sum()

    group2 = agg_six_meals.loc[:, [3, 5, 9, 12, 16]]
    eating_df = pd.concat([group2, group3])
    level = eating_df.index.get_level_values
    # eating_df.unstack([0, 1]).to_csv('meals_time_sum_for_each_mice.csv')

    three_meals['meals_times_indexs'] = meals_times_indexs_three_meal
    three_meals.set_index('meals_times_indexs', append=True, inplace=True)
    datapoint = three_meals['actual_foodupa'].unstack([1, 2])
    # datapoint.to_csv('eating_times_three_meals.csv')

    six_meals['meals_times_indexs'] = meals_times_indexs_six_meals
    six_meals.set_index('meals_times_indexs', append=True, inplace=True)
    datapoint = six_meals['actual_foodupa'].unstack([1, 2])
    # datapoint.to_csv('eating_times_six_meals.csv')


def get_dataframe():
    df_data_experiment = pd.read_csv(
        'csvs\shani\modiInCal_format_SHANI_EXP_MODI_RESTRIC_PLUS_DD.csv')
    df_design_expriment = pd.read_csv(
        'csvs\shani\modiInCal_format_your_Design_SHANI_EXP_MODI_RESTRIC_PLUS_DD.csv')
    df = incal_create_df_incal_format(df_data_experiment, df_design_expriment)
    return df.sort_index(level=['Date_Time_1', 'subjectsID'])


def set_and_fuse_exp_days(df):
    datetime = df.index.get_level_values(0)
    df['exp_days'] = custome_days(datetime, 8, 30)
    return df.set_index('exp_days', append=True)


def agg_catgory_and_datetime(df, AGGS, freq='D', level_name=0, ):
    group_by_datetime = pd.Grouper(level=3, freq=freq)
    level = df.index.get_level_values(level_name)
    return df.groupby([group_by_datetime, level]).agg(AGGS)


def agg_catgory_and_datetime_dark_and_light(df, AGGS, dark_light, freq='D', level_name=0):
    group_by_datetime = pd.Grouper(level=3, freq=freq)
    level = df.index.get_level_values(level_name)
    return df.groupby([group_by_datetime, dark_light, level]).agg(AGGS)


def agg_catgory(df, AGGS, level_name=0):
    level = df.index.get_level_values(level_name)
    return df.groupby([level]).agg(AGGS)


def agg_catgory_dark_light(df, dark_light, AGGS, level_name=0):
    level = df.index.get_level_values(level_name)
    return df.groupby([level, dark_light]).agg(AGGS)


def get_delta_values_for_each_row(df_or_series, unstacks=[1, 2]):
    delta_unstack = df_or_series.unstack(unstacks)
    first_row = delta_unstack.iloc[0]
    return delta_unstack - first_row


def make_bar_plot_for_each_week(x, y, facet_col):
    fig = px.bar(x=subjs, y=food, color=subjs, template='simple_white',
                 facet_col=weeks, width=5000, height=500, labels={'x': 'Subjects', 'y': 'Food Intake (gram)', 'color': 'Subjects'})
    fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=20),
    )
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1])).show()


def export_aggs_analysis(agg_subjectsID_and_each_week, agg_subjectsID_and_each_day, agg_subjectsID, paths):
    for column, calc in AGGS.items():
        agg_subjectsID_and_each_week[column].unstack().to_csv(
            f'{paths[0]}{column}_{calc}_{weeks_avg}')
        agg_subjectsID_and_each_day[column].unstack().to_csv(
            f'{paths[1]}{column}_{calc}_{days_avg}')

    columns = {key: f'{key}_{val}' for key, val in AGGS.items()}
    agg_subjectsID = agg_subjectsID.rename(columns=columns)
    agg_subjectsID.to_csv(
        f'{paths[2]}{days_avg}')


def day_and_night(pd_index_datetime, start='08:30', end='16:30'):
    times = np.array([time.time() for time in pd_index_datetime])
    greater = pd.to_datetime(start).time() <= times
    stricly_less = pd.to_datetime(end).time() > times
    return np.where((greater & stricly_less), 'Day', 'Night')


def mask_starts_end_ends(marks):
    def is_starts_mark(mark, temp): return int(mark) != 0 and temp != mark
    starts_ends = []
    temp = None
    for mark in marks:
        starts = is_starts_mark(mark, temp)
        temp = mark if starts else None
        starts_ends.append(starts)
    return starts_ends


def mark_between_times(pd_index_datetime, meals_times):
    time_series = pd_index_datetime.to_series()
    meals = list(meals_times.items())
    gen_datetime_indexs = (time_series.between_time(times[1][0], times[1][1])
                           for times in meals)
    cond = [
        time_series.index.isin(times_indexs)
        for times_indexs in gen_datetime_indexs
    ]
    choice = [meal_name[0] for meal_name in meals]
    return np.select(cond, choice)


def custome_days(pd_DateTime64, hour=00, minute=00, sec=00):
    time_1 = dt.timedelta(hours=00, minutes=00, seconds=00)
    time_2 = dt.timedelta(hours=hour, minutes=minute, seconds=sec)
    return pd_DateTime64 - (time_2 - time_1)


def flat_list(d_list):
    return list(itertools.chain.from_iterable(d_list))


def incal_create_df_incal_format(
        df_data_experiment,
        df_design_expriment):
    df = df_data_experiment.copy()
    categories_groups = df_design_expriment.values[:, 0]
    categories_subjects = list(
        filter(lambda x: ~np.isnan(x),
               (flat_list(df_design_expriment.values[:, 1:]))))
    date_time_level = pd.Series(
        pd.DatetimeIndex(df['Date_Time_1']),
        name='Date_Time_1')
    subjects_level = pd.Series(
        pd.Categorical(df['subjectID'],
                       categories=categories_subjects,
                       ordered=True),
        name='subjectsID')
    group_level = pd.Series(pd.Categorical(
        df['Group'],
        categories=categories_groups,
        ordered=True),
        name='Group')

    df = df.drop(columns=['Date_Time_1', 'subjectID', 'Group'])

    multi_index_dataframe = pd.concat(
        [date_time_level, subjects_level, group_level], axis=1)

    return pd.DataFrame(
        df.values,
        index=pd.MultiIndex.from_frame(multi_index_dataframe),
        columns=df.columns.values.tolist())


def get_dataframe():
    df_data_experiment = pd.read_csv(FILE_DATA_EXPRIMENTS)
    df_design_expriment = pd.read_csv(FILE_DESIGN_EXPRIMENT)
    df = incal_create_df_incal_format(df_data_experiment, df_design_expriment)
    return df.sort_index(level=['Date_Time_1', 'subjectsID'])


def make_hourly_avg_and_sums(df, times_of_expriment, level=3, freq='H'):
    data_with_expriment_itmes = df.set_index(times_of_expriment, append=True)
    pd_grouper = pd.Grouper(level=level, freq=freq)
    object_groupby = data_with_expriment_itmes.groupby([pd_grouper, idx(1)])
    hourly_avg_and_sums = object_groupby.agg(AGGS)
    subjects = hourly_avg_and_sums.index.get_level_values(1)
    groups_names = assemble_groups(subjects, design_expriment)
    return hourly_avg_and_sums.set_index(
        groups_names, append=True)


def custome_days(pd_DateTime64, hour=00, minute=00, sec=00):
    time_1 = dt.timedelta(hours=00, minutes=00, seconds=00)
    time_2 = dt.timedelta(hours=hour, minutes=minute, seconds=sec)
    return (pd_DateTime64 - (time_2 - time_1))


def assemble_groups(data_subjects, design_expriment):
    cond = [data_subjects.isin(subjects[1]) for subjects in design_expriment]
    choice = [group[0] for group in design_expriment]
    return np.select(cond, choice)
