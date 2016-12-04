import pandas as pd
import numpy as np

global COLUMNS_NAME,SELECTED_COLUMNS

# Columns of the data
COLUMNS_NAME = ["DATE","DAY_OFF","DAY_DS","WEEK_END","DAY_WE_DS","TPER_TEAM","TPER_HOUR",
                        "SPLIT_COD","ACD_COD","ACD_LIB","ASS_SOC_MERE","ASS_DIRECTORSHIP","ASS_ASSIGNMENT",
                        "ASS_PARTNER","ASS_POLE","ASS_BEGIN","ASS_END","ASS_COMENT","CSPL_I_STAFFTIME","CSPL_I_AVAILTIME",
                        "CSPL_I_ACDTIME","CSPL_I_ACWTIME","CSPL_I_ACWOUTTIME","CSPL_I_ACWINTIME","CSPL_I_AUXOUTTIME",
                        "CSPL_I_AUXINTIME","CSPL_I_OTHERTIME","CSPL_ACWINCALLS","CSPL_ACWINTIME","CSPL_AUXINCALLS",
                        "CSPL_AUXINTIME","CSPL_ACWOUTCALLS","CSPL_ACWOUTIME","CSPL_ACWOUTOFFCALLS","CSPL_ACWOUTOFFTIME",
                        "CSPL_AUXOUTCALLS","CSPL_AUXOUTTIME","CSPL_AUXOUTOFFCALLS","CSPL_AUXOUTOFFTIME","CSPL_INFLOWCALLS",
                        "CSPL_ACDCALLS","CSPL_ANSTIME","CSPL_HOLDCALLS","CSPL_HOLDTIME","CSPL_HOLDABNCALLS","CSPL_TRANSFERED",
                        "CSPL_CONFERENCE","CSPL_ABNCALLS","CSPL_ABNTIME","CSPL_ABNCALLS1","CSPL_ABNCALLS2","CSPL_ABNCALLS3",
                        "CSPL_ABNCALLS4","CSPL_ABNCALLS5","CSPL_ABNCALLS6","CSPL_ABNCALLS7","CSPL_ABNCALLS8","CSPL_ABNCALLS9",
                        "CSPL_ABNCALLS10","CSPL_OUTFLOWCALLS","CSPL_OUTFLOWTIME","CSPL_MAXINQUEUE","CSPL_CALLSOFFERED",
                        "CSPL_I_RINGTIME","CSPL_RINGTIME","CSPL_RINGCALLS","CSPL_NOANSREDIR","CSPL_MAXSTAFFED",
                        "CSPL_ACWOUTADJCALLS","CSPL_AUXOUTADJCALLS","CSPL_DEQUECALLS","CSPL_DEQUETIME","CSPL_DISCCALLS",
                        "CSPL_DISCTIME","CSPL_INTRVL","CSPL_INCOMPLETE","CSPL_ACCEPTABLE","CSPL_SERVICELEVEL","CSPL_ACDAUXOUTCALLS",
                        "CSPL_SLVLABNS","CSPL_SLVLOUTFLOWS","CSPL_RECEIVED_CALLS","CSPL_ABANDONNED_CALLS","CSPL_CALLS",
                        "CSPL_ACWTIME","CSPL_ACDTIME"]

# Columns selected for now
SELECTED_COLUMNS = ["DATE","DAY_OFF","WEEK_END","ASS_ASSIGNMENT","CSPL_RECEIVED_CALLS"]


def extract_useful_columns(filename, output_filename):
    # Reading the data file
    calls = pd.read_csv(filename, parse_dates=True, names = COLUMNS_NAME, delimiter=";")
    print("End reading")
    # Removing columns we won't use from dataframe
    selected_calls = calls[SELECTED_COLUMNS]
    # Saving the new dataframe into new file, to avoid loading full file next time
    selected_calls.to_csv(output_filename)
    print("End writing")
    return selected_calls

if __name__=="__main__":
    extract_useful_columns("data/train.csv", "data/train_ultra_light.csv")
