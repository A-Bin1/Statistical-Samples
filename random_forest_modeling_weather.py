#random forest modeling for weather dataset
#Station SPW00013025 data download from https://www.ncei.noaa.gov/cdo-web/
# Note: WDFG (direction of wind gust)
#       WSFG (peak gust wind speed)

import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/A-Bin1/Statistical-Samples/main/random_forest_modeling_dataset_weather.csv'
weather_df = pd.read_csv(url)


#weather_df = pd.read_csv('C:/Users/alexa/Statistical_Samples/random_forest_modeling_dataset_weather.csv')
weather_df.head(5)
#        STATION          NAME      DATE  TMAX  TMIN  WDFG  WSFG
# 0  SPW00013025  ROTA NAS, SP  1/1/1970    56    43   270  11.4
# 1  SPW00013025  ROTA NAS, SP  1/2/1970    62    48   135  34.4
# 2  SPW00013025  ROTA NAS, SP  1/3/1970    63    58   248  21.9
# 3  SPW00013025  ROTA NAS, SP  1/4/1970    64    58   180  34.4
# 4  SPW00013025  ROTA NAS, SP  1/5/1970    60    53   225  48.3

weather_df["DATE"] = pd.to_datetime(weather_df["DATE"])

#list comp alternative to df[col].dt.day_name to avoid bound method errors from removing DATE column
#from dataset later for random forest model
import calendar
cl = list(calendar.day_name)
weather_df['Weekdays'] = [cl[a.weekday()] for a in weather_df['DATE']]

#create separate columns from datetime DATE column
weather_df["Day"] = weather_df['DATE'].map(lambda x: x.day)
weather_df["Month"] = weather_df['DATE'].map(lambda x: x.month)
weather_df["Year"] = weather_df['DATE'].map(lambda x: x.year)
weather_df['TAVG'] = (weather_df['TMAX']+ weather_df['TMIN'])/2


weather_df.head()
#        STATION          NAME       DATE  TMAX  TMIN  WDFG  WSFG  Weekdays  Day  Month  Year  TAVG
# 0  SPW00013025  ROTA NAS, SP 1970-01-01    56    43   270  11.4  Thursday    1      1  1970  49.5
# 1  SPW00013025  ROTA NAS, SP 1970-01-02    62    48   135  34.4    Friday    2      1  1970  55.0
# 2  SPW00013025  ROTA NAS, SP 1970-01-03    63    58   248  21.9  Saturday    3      1  1970  60.5
# 3  SPW00013025  ROTA NAS, SP 1970-01-04    64    58   180  34.4    Sunday    4      1  1970  61.0
# 4  SPW00013025  ROTA NAS, SP 1970-01-05    60    53   225  48.3    Monday    5      1  1970  56.5

#remove redundant columns into revised dataset
weather_df_copy = weather_df.copy()
weather_df_rev = weather_df_copy.drop(['STATION', 'NAME'], axis =1)

#check for null values in case of need for imputing data
weather_df_rev.isnull().sum().sum()
#0

#remove timestamp column prior to creating features data
weather_df_rev = weather_df_rev.drop('DATE', axis = 1)

# One-hot encode the data
features = pd.get_dummies(weather_df_rev)

#convert to 0,1 boolean encoding columns
features_ec = features.replace({True: 1, False: 0})


#predict for WSFG (peak wind gust speed)

labels = np.array(features_ec['WSFG'])
features_df = features_ec.drop(['WSFG'], axis = 1)
feature_list = list(features_df.columns)
features_arr = np.array(features_df)
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features_arr, labels, test_size = 0.25, random_state = 42)


# Baseline predictions: experiement with given features
def calc_baseline_error(predictor):
    baseline_preds = test_features[:, feature_list.index(predictor)]
    baseline_errors = abs(baseline_preds - test_labels)
    rounded = round(np.mean(baseline_errors), 2)
    return predictor , rounded

for i in feature_list:
    calc_baseline_error(i)

# ('TMAX', 50.09)
# ('TMIN', 31.02)
# ('WDFG', 174.13)
# ('Day', 10.3)
# ('Month', 14.81)
# ('Year', 1950.12)
# ('TAVG', 40.55)
# ('Weekdays_Friday', 21.26)
# ('Weekdays_Monday', 21.24)
# ('Weekdays_Saturday', 21.25)
# ('Weekdays_Sunday', 21.27)
# ('Weekdays_Thursday', 21.26)
# ('Weekdays_Tuesday', 21.25)
# ('Weekdays_Wednesday', 21.29)


#model training data
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)

# Get numerical feature importances by rank
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
feature_importlst = ['Variable: {:1} Importance: {}'.format(*pair) for pair in feature_importances]
for i in feature_importlst:
    print(i)

# Variable: WDFG Importance: 0.21
# Variable: TMIN Importance: 0.18
# Variable: TMAX Importance: 0.17
# Variable: Day Importance: 0.11
# Variable: Month Importance: 0.09
# Variable: TAVG Importance: 0.09
# Variable: Year Importance: 0.05
# Variable: Weekdays_Friday Importance: 0.02
# Variable: Weekdays_Thursday Importance: 0.02
# Variable: Weekdays_Monday Importance: 0.01
# Variable: Weekdays_Saturday Importance: 0.01
# Variable: Weekdays_Sunday Importance: 0.01
# Variable: Weekdays_Tuesday Importance: 0.01
# Variable: Weekdays_Wednesday Importance: 0.01


predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees')
#Mean Absolute Error: 4.63 degrees

#Compare to the list of baseline errors.
baseline_feature_list = []
for i in feature_list:
    baseline_feature_list.append(calc_baseline_error(i))

baseline_values = [x[1] for x in baseline_feature_list if type(x[0]) == str]
baseline_names = [x[0] for x in baseline_feature_list if type(x[0]) == str]

MAE = round(np.mean(errors), 2)

degree_diff = [round((x - MAE),2) for x in baseline_values]

degree_diff_df = pd.DataFrame(
    {'Feature': baseline_names,
     'Degree_Diff': degree_diff
    })

degree_diff_df.sort_values(by='Degree_Diff')
#                Feature  Degree_Diff
# 3                  Day         5.67
# 4                Month        10.18
# 8      Weekdays_Monday        16.61
# 9    Weekdays_Saturday        16.62
# 12    Weekdays_Tuesday        16.62
# 7      Weekdays_Friday        16.63
# 11   Weekdays_Thursday        16.63
# 10     Weekdays_Sunday        16.64
# 13  Weekdays_Wednesday        16.66
# 1                 TMIN        26.39
# 6                 TAVG        35.92
# 0                 TMAX        45.46
# 2                 WDFG       169.50
# 5                 Year      1945.49

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%')
#Accuracy: 74.93 %
#on the lower range of accuracy with the given data set.

