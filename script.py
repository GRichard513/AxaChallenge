# coding: utf8

# All library imports
import math
import pandas as pd
import numpy as np
import error_functions as ef
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
print("Imports ok.")

# Few functions to add features to the dataframe.

def is_working_day(d):
    return d.isoweekday()<6

def trimester(d):
    return (d.month-1)//3

def is_day_shift(d):
    min_hour = {
        'hour':7,
        'minute':30
    }
    max_hour = {
        'hour':23,
        'minute':30
    }
    hour_day = d.hour*60+d.minute
    return (hour_day > min_hour['hour']*60+min_hour['minute']) & (hour_day < max_hour['hour']*60+max_hour['minute'])

def evolution_over_years(d):
    min_date = {
        "year": 2011,
        "month": 1,
        "day": 1
    }
    date_trimester = trimester(d)
    year = d.year
    return (year-min_date['year'])*4+date_trimester

def previous_days(df, nb_days, year, month, day):
    difference_days = day-nb_days
    if (difference_days < 0):
        if (month==0):
            if (year==2011):
                df_output = df[(df.year == year) & (df.month == 0) & (df.day < day)]
            else:
                df_output = df[((df.year == year) & (df.month == 0) & (df.day < day)) | ((df.year == year-1) & (df.month >= 11) & (df.day > 31-abs(difference_days)))]
        else:
            df_output = df[(df.year == year) & (((df.month == month) & (df.day <= day)) | ((df.month == month-1) & (df.day >= 31-abs(difference_days))))]
    else:
        df_output = df[(df.year==year) & (df.month == month) & (df.day<=day) & (df.day > day-nb_days)]

    return df_output

def predict_line(line_to_predict, map_basic, map_big, map_huge, number_fitting, min_fitting_value):
    assign = line_to_predict.assignment
    prediction = map_basic[assign].predict(line_to_predict.values.reshape(1, -1))
    prediction = np.round(prediction)

    is_big_regressor_fitted = number_fitting[assign]['big']>min_fitting_value
    is_huge_regressor_fitted = number_fitting[assign]['huge']>min_fitting_value

    if (prediction > 19) and is_big_regressor_fitted:
        prediction = np.round(map_big[assign].predict(line_to_predict))
    if (prediction > 49) and is_huge_regressor_fitted:
        prediction = np.round(map_huge[assign].predict(line_to_predict))

    val = int(math.floor(prediction[0]))
    if val < 0:
        val = 0
    return val

print("Definition of functions ok.")

# Map to convert ass_assignments to integer values
assignments = ['CAT', 'CMS', 'Crises', 'Domicile',
       'Evenements', 'Gestion', 'Gestion - Accueil Telephonique',
       'Gestion Amex', 'Gestion Assurances', 'Gestion Clients', 'Gestion DZ',
       'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Manager',
       'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC',
       'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter',
       'Tech. Total', 'Téléphonie']

map_assignment = {}
for i in range(len(assignments)):
    map_assignment[assignments[i]]=i

map_assignment_inverse = {}
for i in range(len(assignments)):
    map_assignment_inverse[i] = assignments[i]

print("Importing train values...")
calls = pd.read_csv('data/train.csv', delimiter=';', nrows=1000000)
print("Reading train file ok.")
y = calls['CSPL_CALLS']
print("Extracting results ok.")
print("Importing submission values...")
submission = pd.read_csv('data/submission.txt', sep="\t")
print("Reading submission file ok.")
submission = submission.drop('prediction', axis=1)
print("Dropping prediction from submission ok.")

calls['DATE'] = pd.to_datetime(calls['DATE'],infer_datetime_format=True)
print("Train date to timetable ok.")
submission['DATE'] = pd.to_datetime(submission['DATE'],infer_datetime_format=True)
print("Submission date to timetable ok.")

calls = calls[["DATE", "ASS_ASSIGNMENT"]]
print("Extraction of columns date and ass_assignment ok.")
print("")
print("Adding new features to train...")
calls['year'] = [dd.year for dd in calls['DATE']]
calls['month'] = [dd.month for dd in calls['DATE']]
calls['day'] = [dd.day for dd in calls['DATE']]
print("Year, month, day ok.")
calls['hour'] = [dd.hour for dd in calls['DATE']]
calls['minute'] = [dd.minute for dd in calls['DATE']]
#calls['second'] = [dd.second for dd in calls['DATE']]
print("Hour, minute ok.")
calls['working'] = [is_working_day(dd) for dd in calls['DATE']]
print("Working day ok.")
calls['shift'] = [is_day_shift(dd) for dd in calls['DATE']]
print("Shift ok.")
calls['trimester'] = [trimester(dd) for dd in calls['DATE']]
print("Trimester ok.")
calls['evolution'] = [evolution_over_years(dd) for dd in calls['DATE']]
print("Evolution ok.")
calls['assignment'] = calls['ASS_ASSIGNMENT'].map(map_assignment)
print("Creating dummies ok.")
calls = calls.drop(['DATE','ASS_ASSIGNMENT'], axis=1)
print("Dropping string columns ok.")
calls['day_number'] = [dd.isoweekday() for dd in calls['DATE']]
print("Day number ok.")
print("")
print("Adding new features to submission...")
submission['year'] = [dd.year for dd in submission['DATE']]
submission['month'] = [dd.month for dd in submission['DATE']]
submission['day'] = [dd.day for dd in submission['DATE']]
print("Year, month, day ok.")
submission['hour'] = [dd.hour for dd in submission['DATE']]
submission['minute'] = [dd.minute for dd in submission['DATE']]
#submission['second'] = [dd.second for dd in submission['DATE']]
print("Hour, minute ok.")
submission['working'] = [is_working_day(dd) for dd in submission['DATE']]
print("Working day ok.")
submission['day_number'] = [dd.isoweekday() for dd in submission['DATE']]
print("Day number ok.")
submission['shift'] = [is_day_shift(dd) for dd in submission['DATE']]
print("Shift ok.")
submission['trimester'] = [trimester(dd) for dd in submission['DATE']]
print("Trimester ok.")
submission['evolution'] = [evolution_over_years(dd) for dd in submission['DATE']]
print("Evolution ok.")
submission['assignment'] = submission['ASS_ASSIGNMENT'].map(map_assignment)
print("Creating dummies ok.")
save_date = submission['DATE']
print("Saving date column ok.")
save_ass_assignment = submission['ASS_ASSIGNMENT']
print("Saving ass_assignment ok.")
submission = submission.drop(['DATE','ASS_ASSIGNMENT'], axis=1)
print("Dropping string columns ok.")


map_regressor = {}
big_values_regressor = {}
huge_values_regressor = {}
number_values = {}
print("Initialize maps ok.")

for i in range(len(assignments)):
    number_values[i] = {
        "big":0,
        "huge":0
    }
    map_regressor[i] = Ridge()
    big_values_regressor[i] = Ridge()
    huge_values_regressor[i] = Ridge()
print("Initialize regressors ok.")

print("--- TRAINING ---")
X_train, X_test, y_train, y_test = train_test_split(calls, y, test_size=10000)
X_train_full = X_train
X_train_full['pred'] = y_train
X_test_full = X_test
X_test_full['pred'] = y_test
print("Separation of train test ok.")
for i in range(len(assignments)):
    el_x = X_train_full[X_train_full['assignment']==i]
    big_x = el_x[el_x['pred'] > 19]
    huge_x = el_x[el_x['pred'] > 49]
    big_elements = big_x.shape[0]
    huge_elements = huge_x.shape[0]
    number_values[i] = {
        "big":big_elements,
        "huge":huge_elements
    }
    big_y = big_x['pred']
    big_x = big_x.drop('pred', axis=1)
    el_y = el_x['pred']
    el_x = el_x.drop("pred", axis=1)
    map_regressor[i].fit(el_x, el_y)
    if big_elements>0:
        big_values_regressor[i].fit(big_x, big_y)

    huge_y = huge_x['pred']
    huge_x = huge_x.drop('pred', axis = 1)
    if huge_elements>0:
        huge_values_regressor[i].fit(huge_x, huge_y)

    print("Training ", i, " ok.")

if True: # If testing on test sample
    error = 0
    size = 0

    min_values_of_fitting = 20

    for i in range(len(assignments)):
        el_x = X_test_full[X_test_full['assignment']==i]
        size += len(el_x)
        el_y = el_x['pred']
        el_x = el_x.drop('pred', axis=1)
        y2 = map_regressor[i].predict(el_x)
        el_y = np.array(el_y)
        y2 = np.round(np.array(y2))
        is_big_regressor_fitted = number_values[i]['big']>min_values_of_fitting
        is_huge_regressor_fitted = number_values[i]['huge']>min_values_of_fitting
        for i, pred in np.ndenumerate(y2):
            if (pred > 19) and is_big_regressor_fitted :
                tmp = el_x.iloc[i]
                assign = tmp.assignment
                y2[i] = big_values_regressor[assign].predict(tmp)
                y2[i] = np.round(y2[i])

            if (y2[i] > 49) and is_huge_regressor_fitted:
                tmp = el_x.iloc[i]
                assign = tmp.assignment
                y2[i] = huge_values_regressor[assign].predict(tmp)
                y2[i] = np.round(y2[i])

        error_tmp = ef.linex_loss(np.array(y2), np.array(el_y))
        imax = np.argmax(np.abs(np.array(y2)-np.array(el_y)))
        error += error_tmp

    print("size ", size)
    print("loss ", error)


if False: # If creating a new submission.txt
    final_result = submission.copy()
    print("Initialize second copy ok.")

    min_values_of_fitting = 10
    predictions = []
    for index, row in final_result.iterrows():
        predictions.append(predict_line(row, map_regressor, big_values_regressor, huge_values_regressor, number_values, min_values_of_fitting))
        if index % 500 == 0:
            print(index)
    print("Prediction ok.")
    final_result['prediction'] = predictions
    print(final_result.columns)

    print("--- Writing to file ---")
    final_result['DATE'] = save_date
    final_result['ASS_ASSIGNMENT'] = save_ass_assignment
    print("Adding date, ass_assignment, prediction to dataframe ok.")
    final_result = final_result[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
    print("Dropping all other columns ok.")
    final_result.to_csv("submission_test.txt", sep="\t", index=False)
    print("Writing to file ok.")
