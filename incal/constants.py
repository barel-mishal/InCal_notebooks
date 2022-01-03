meals_3meals = {
    '1': ('16:30', '19:30'),
    '2': ('22:30', '00:30'),
    '3': ('4:30', '5:30')
}
meals_6meals = {
    '1': ('16:30', '18:00'),
    '2': ('20:00', '20:30'),
    '3': ('22:30', '00:00'),
    '4': ('2:00', '2:30'),
    '5': ('4:30', '6:00'),
    '6': ('8:00', '8:30')
}

meals_3meals_buffer = {
    '1': ('16:15', '19:58'),
    '2': ('22:15', '00:58'),
    '3': ('4:15', '5:58')
}
meals_6meals_buffer = {
    '1': ('16:15', '18:30'),
    '2': ('19:45', '20:57'),
    '3': ('22:15', '00:30'),
    '4': ('1:45', '2:57'),
    '5': ('4:15', '6:30'),
    '6': ('7:45', '8:57')
}
# glob.glob(pathname)
AGGS = {
    'actual_foodupa': 'sum',
    'kcal_hr': 'sum',
    'kcal_mean': 'mean',
    'actual_allmeters': 'mean',
    "bodymass": 'mean',
    'rq': 'mean',
    'actual_pedmeters': 'mean',
    'vco2': 'mean',
    'vo2': 'mean',
    'locomotion': 'mean',
    'actual_waterupa': 'sum'
}

period = 'dd_period'
save_path_light_dark = [f'csvs\shani\summary\\{period}\light_dark_agg\\avg_weeks\\',
                        f'csvs\shani\summary\\{period}\light_dark_agg\\avg_days\\',
                        f'csvs\shani\summary\\{period}\light_dark_agg\\avg_all_exp\\']
save_path_all_exp = [f'csvs\shani\summary\\{period}\\all_exp_agg\\avg_weeks\\',
                     f'csvs\shani\summary\\{period}\\all_exp_agg\\avg_days\\',
                     f'csvs\shani\summary\\{period}\\all_exp_agg\\avg_all_exp\\']


FILE_DATA_EXPRIMENTS = 'csvs\shani\modiInCal_format_SHANI_EXP_MODI_RESTRIC_PLUS_DD.csv'
FILE_DESIGN_EXPRIMENT = 'csvs\shani\modiInCal_format_your_Design_SHANI_EXP_MODI_RESTRIC_PLUS_DD.csv'
AGGS = {
    'actual_foodupa': 'sum',
    'kcal': 'sum',
    'kcal_mean': 'mean',
    'actual_allmeters': 'mean',
    "bodymass": 'mean',
    'rq': 'mean',
    'actual_pedmeters': 'mean',
    'vco2': 'mean',
    'vo2': 'mean',
    'locomotion': 'mean',
    'actual_waterupa': 'sum'
}
