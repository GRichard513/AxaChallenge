import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import error_functions as ef
import math

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

if __name__ == "__main__":
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

    print("Start...")
    calls = pd.read_csv('data/train.csv', delimiter=';', nrows=1000000)
    print("Reading csv ok.")
    y = calls['CSPL_CALLS']
    print("Extracting results ok.")

    calls['DATE'] = pd.to_datetime(calls['DATE'],infer_datetime_format=True)
    print("Date to timetable ok.")
    calls = calls[['DATE', 'ASS_ASSIGNMENT']]
    print("Extraction of columns date and ass_assignment ok.")
    print("")
    print("Adding new features...")
    calls['year'] = [dd.year for dd in calls['DATE']]
    calls['month'] = [dd.month for dd in calls['DATE']]
    calls['day'] = [dd.day for dd in calls['DATE']]
    print("Year, month, day ok.")
    calls['hour'] = [dd.hour for dd in calls['DATE']]
    calls['minute'] = [dd.minute for dd in calls['DATE']]
    calls['second'] = [dd.second for dd in calls['DATE']]
    print("Hour, minute, second ok.")
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
    print("")
    print("Starting submission extraction...")
    submission = pd.read_csv('data/submission.txt', sep="\t", nrows=1000)
    print("Reading csv ok.")
    submission = submission.drop('prediction', axis=1)
    print("Dropping prediction ok.")
    submission['DATE'] = pd.to_datetime(submission['DATE'],infer_datetime_format=True)
    print("Date to timetable ok.")
    print("")
    print("Adding new features...")
    submission['year'] = [dd.year for dd in submission['DATE']]
    submission['month'] = [dd.month for dd in submission['DATE']]
    submission['day'] = [dd.day for dd in submission['DATE']]
    print("Year, month, day ok.")
    submission['hour'] = [dd.hour for dd in submission['DATE']]
    submission['minute'] = [dd.minute for dd in submission['DATE']]
    submission['second'] = [dd.second for dd in submission['DATE']]
    print("Hour, minute, second ok.")
    submission['working'] = [is_working_day(dd) for dd in submission['DATE']]
    print("Working day ok.")
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
    print("")
    print("------------------------")
    print("--- CROSS VALIDATION ---")
    print("")
    print("Creating regressor...")
    X_train, X_test, y_train, y_test = train_test_split(calls, y, test_size=0.2)
    print("Splitting train from test ok.")
    S=Ridge()
    print("Instanciating regressor ok.")
    S.fit(X_train, y_train)
    print("Training phase ok.")
    y2=S.predict(X_test)
    for i, el in np.ndenumerate(y2):
        if el < 2:
            y2[i] = 0
        else:
            y2[i] = int(math.floor(y2[i]))
    print("Prediction ok.")
    y_test = np.array(y_test)
    print("Pandas series to numpy array ok.")
    loss = ef.linex_loss(y2, y_test)
    print("Compute of loss ok : ", loss)
    print("")
    print("------------------")
    print("--- SUBMISSION ---")
    print("")
    S=Ridge()
    print("Instanciating regressor ok.")
    S.fit(calls, y)
    print("Training phase ok.")
    y2=S.predict(submission)
    print("Prediction ok.")
    for i, el in np.ndenumerate(y2):
        if el < 4:
            y2[i] = 0
        else:
            y2[i] = int(math.floor(y2[i]))
    print("Taking floor value of prediction")
    submission['DATE'] = save_date
    submission['ASS_ASSIGNMENT'] = save_ass_assignment
    submission['prediction'] = y2
    print("Adding date, ass_assignment, prediction to dataframe ok.")
    submission = submission[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
    print("Dropping all other columns ok.")
    submission.to_csv("submission_test.txt", sep="\t", index=False)
    print("Writing to file ok.")
