from incal_lib.readers import read_CalR_sable_file
from collections import OrderedDict

PATH, PATHS = '', [
    "csvs\meital\hebrew_2021-10-10_07_36_ido_accessdoors_1021_script1_w1_m_calr.csv",
    "csvs\meital\hebrew_2021-10-17_10_05_ido_accessdoors_1021_script2_w2_m_calr.csv",
    "csvs\meital\hebrew_2021-10-25_16_03_ido_accessdoors_1021_script2_w3_m_calr.csv",
    "csvs\meital\hebrew_2021-10-31_09_58_ido_accessdoors_1021_script2_w4_m_calr.csv",
    "csvs\meital\hebrew_2021-11-07_10_52_ido_accessdoors_1021_script1_w5_m_calr.csv",
]
GROUPS_EXP_DESIGN_STRACTURE = list(
    OrderedDict(Group=[i for i in range(1, 17)]).items())
DATETIME = 'Date_Time_1'
CUMULETIVE_COLUMNS_NAMES = "|".join(
    ['food', 'water', 'allmeters', 'wheelmeters', 'pedmeters'])
PREFIX_CUMUL = 'actual_'
# gives a df with calr foramat
df = read_CalR_sable_file(
    PATHS,
    GROUPS_EXP_DESIGN_STRACTURE,
    DATETIME,
    CUMULETIVE_COLUMNS_NAMES,
    PREFIX_CUMUL)
    
print(df)

print(df.Date_Time_1)
