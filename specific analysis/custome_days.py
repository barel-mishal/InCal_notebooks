import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import itertools
import plotly.express as px
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def custome_days(pd_DateTime64, hour=00, minute=00, sec=00):
    time_1 = dt.timedelta(hours=00, minutes=00, seconds=00)
    time_2 = dt.timedelta(hours=hour, minutes=minute, seconds=sec)
    return (pd_DateTime64 - (time_2 - time_1))


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


def box_plot_by_time(data, X, COLOR):
    return px.box(x=data.index.get_level_values(0),
                  y=data[X],
                  template='simple_white',
                  color=COLOR,
                  labels={
        'y': X,
        'x': 'Days of expriment'
    })


def get_eat_and_fest_list(all_days, uniq_days):
    is_eat_day = []
    for day in all_days:
        if day != uniq_days[0] and day in uniq_days[1::2]:
            is_eat_day.append('#FF851B')
        if day != uniq_days[0] and day not in uniq_days[1::2]:
            is_eat_day.append('rgb(233,233,233)')
    return is_eat_day


def make_graphs():
    # plotly setup
    plot_rows = 2
    plot_cols = 2
    fig = make_subplots(rows=plot_rows, cols=plot_cols)
    # add traces

    x = 0
    for i in range(1, plot_rows + 1):
        for j in range(1, plot_cols + 1):
            fig.add_trace(go.Box(
                x=df_agg_each_subjects.index.get_level_values(0),
                y=df_agg_each_subjects[df_agg_each_subjects.columns[x]].values,
                name=df_agg_each_subjects.columns[x], marker_color='#3D9970'),
                row=i,
                col=j)

            x += 1
    # Format and show fig
    fig.update_layout(height=1200, width=3000)
    fig.show()


if __name__ == '__main__':
    experiment_name = "maital"
    path = "csvs\meital\w5_end\w5_endInCal_format_maital_w5_end.csv"
    experiment_design_path = "csvs\meital\w5_end\w5_endInCal_format_your_Design_maital_w5_end.csv"
    datetime_col_name = 'Date_Time_1'
    pattern_addition_to_parms = 'actual_'

    df = pd.read_csv(path, parse_dates=[datetime_col_name])
    exp_df = pd.read_csv(experiment_design_path)
    df = incal_create_df_incal_format(df, exp_df)
    df['locomotion'] = df['xbreak'].add(df['ybreak'])
    df = df.drop(columns=['vh2o', 'xbreak', 'ybreak'])
    df.sort_index(level=['Date_Time_1', 'subjectsID'], inplace=True)
    index_datetime = df.index.get_level_values(0)
    df_agg = df.copy()
    df_agg['exp_days'] = custome_days(index_datetime, 8)
    df_agg.set_index('exp_days', append=True, inplace=True)
    for name in df_agg.columns:
        df_agg.unstack([1, 2])[name].to_csv(
            f'csvs/meital/raw data with counted custome days/{name}.csv')
    df_agg['kcal_mean'] = df_agg['kcal_hr']
    columns_aggs = {
        'actual_foodupa': 'sum',
        'kcal_hr': 'sum',
        'kcal_mean': 'mean',
        'actual_allmeters': 'mean',
        "bodymass": 'mean',
        'actual_wheelmeters': 'mean',
        'rq': 'mean',
        'actual_pedmeters': 'mean',
        'vco2': 'mean',
        'vo2': 'mean',
        'locomotion': 'mean',
        'actual_waterupa': 'sum'
    }
    df_agg_each_subjects = df_agg.groupby(
        [pd.Grouper(level=3, freq='D'),
         df_agg.index.get_level_values(1)]).agg(columns_aggs)
    df_agg_group = df_agg.groupby(
        [pd.Grouper(level=3, freq='D'),
         df_agg.index.get_level_values(2)]).agg(columns_aggs)

    for name, agg in columns_aggs.items():
        df_agg_each_subjects.unstack()[name].to_csv(
            f'csvs/meital/agg/subjects {name} {agg}.csv')
        df_agg_group.unstack()[name].to_csv(
            f'csvs/meital/agg/group {name} {agg}.csv')
    all_days = df_agg_each_subjects.index.get_level_values(0)
    uniq_days = all_days.unique()
    df_agg_each_subjects = df_agg_each_subjects.loc[uniq_days[1:]]
    X = 'actual_foodupa'
    is_eat_day = get_eat_and_fest_list(all_days, uniq_days)
    fig = box_plot_by_time(df_agg_each_subjects, X, is_eat_day)
    fig.show()
