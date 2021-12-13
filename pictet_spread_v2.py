#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:44:57 2021

@author: benoit
"""

import pandas as pd
import numpy as np
import wbdata
import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, max_error, accuracy_score

def add_delta_spreads(data, country, month):
    delta_spreads = data.iloc[month:-1,:].copy()
    delta_spreads.index = data.Date[month + 1:].values
    old_spreads = data.iloc[:-month-1,:].copy()
    old_spreads.index = data.Date[month + 1:].values
    delta_spreads["previous_delta_" + str(month)] = delta_spreads[country] - old_spreads[country]
    delta_spreads["Date"] = delta_spreads.index
    delta_spreads["country"] = country
    delta_spreads = delta_spreads[["Date", "country", "previous_delta_" + str(month)]]
    delta_spreads = delta_spreads.set_index(["Date", "country"])
    return delta_spreads 

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n       
    
def add_mean_spreads(data, country, month):
    mean_spreads = data.iloc[:-1,:].copy()
    mean_spreads["Date"] = data.Date[1:].values
    mean_spreads["country"] = country
    mean_spreads.columns = ["Date", "mean_spread_" + str(month), "country"]
    mean_spreads = mean_spreads.set_index(["Date", "country"])
    mean_spreads = mean_spreads.dropna(subset=["mean_spread_" + str(month)])
    
    mean_spreads["mean_spread_" + str(month)] = moving_average(mean_spreads["mean_spread_" + str(month)].values, month)
    
    #keep only the rows for which there were enough months to compute the mean
    mean_spreads = mean_spreads.iloc[month - 1:,:]
    
    return mean_spreads

def prepare_all_data(spreads, all_data, normalized_data_from_countries, ESG_Scores, month_spread_prediction, prediction):
    Y = pd.DataFrame({})
    for country in spreads.columns[1:]:
        if prediction == "value":
            if month_spread_prediction != 1:
                Y = pd.concat([Y, pd.DataFrame({"country":[country] * (len(spreads)-month_spread_prediction + 1), "Date":spreads.Date[:-month_spread_prediction+1].values, "spread":spreads[country][month_spread_prediction-1:].values})])
            else:
                Y = pd.concat([Y, pd.DataFrame({"country":[country] * (len(spreads)), "Date":spreads.Date.values, "spread":spreads[country].values})])
        elif prediction == "direction":
            Y = pd.concat([Y, pd.DataFrame({"country":[country] * (len(spreads) - month_spread_prediction), "Date":spreads.Date.values[month_spread_prediction:], "spread":spreads[country].values[month_spread_prediction:] - spreads[country].values[:-month_spread_prediction]})])
    
    all_data = all_data.merge(Y.set_index(["Date", "country"]), left_index = True, right_index=True, how="left")
    all_data = all_data.sort_values(["country", "Date"])
    
    all_data["year"] = all_data.Date.dt.year
    normalized_data_from_countries["year"] = normalized_data_from_countries.Date.dt.year
    ESG_Scores["year"] = ESG_Scores.date.dt.year
    normalized_data_from_countries = normalized_data_from_countries.drop(columns = ["Date"])
    ESG_Scores = ESG_Scores.drop(columns = ["date"])
    
    result = all_data.merge(normalized_data_from_countries, on=["year", "country"], how="left") 
    result = result.merge(ESG_Scores, on=["year", "country"], how="left") 

    return result 

path_file = "/Users/benoit/Downloads/Documents/Cover Letter-Resume/case_study_pictet/spreads/"
path_ESG = "/Users/benoit/Downloads/Documents/Cover Letter-Resume/case_study_pictet/"

spreads = pd.read_excel(path_file + "2021_ESG_datascientist_casestudy_v2_data.xlsx")    

spreads = spreads.replace(0., np.nan)


spreads_without_covid = spreads[(spreads.Date < "2020-02-28") | (spreads.Date > "2020-08-31")]


spreads_before_covid = spreads[spreads.Date < "2020-02-28"]


## Lets drop the time dimension: so all are instances
# but for now we predict at 1 month. we should potentially change that

#list_months = [1, 3, 6, 12, 18, 24]

list_months = [1, 3, 6]

df_mean_spreads = pd.DataFrame({})
df_delta_spreads = pd.DataFrame({})

for country in spreads.columns[1:]:
    spreads_country = spreads[["Date", country]]
        
    df_add_to_mean_spreads = pd.DataFrame({})
    for month in list_months:
        df_add_to_mean_spreads["mean_spread_" + str(month)] = add_mean_spreads(spreads_country, country, month)["mean_spread_" + str(month)]
    df_mean_spreads = pd.concat([df_mean_spreads, df_add_to_mean_spreads])
    
    df_add_to_previous_delta = pd.DataFrame({})
    for month in list_months:
        df_add_to_previous_delta["previous_delta_" + str(month)] = add_delta_spreads(spreads_country, country, month)["previous_delta_" + str(month)]
    df_delta_spreads = pd.concat([df_delta_spreads, df_add_to_previous_delta])
    

all_data = df_mean_spreads.merge(df_delta_spreads, left_index = True, right_index=True, how="outer")
all_data["Date"] = [x[0] for x in all_data.index.values]
all_data["country"] = [x[1] for x in all_data.index.values]
all_data = all_data.sort_values(["country", "Date"])

##### chose the prediciton we want to make: 1 months 3 months? 6 months? 1 year?
normalized_data_from_countries = pd.read_csv(path_file + "normalized_data_recent.csv")
normalized_data_from_countries = normalized_data_from_countries.iloc[:,1:]
normalized_data_from_countries.Date = pd.to_datetime(normalized_data_from_countries.Date)

ESG_Scores = pd.read_csv(path_ESG + "ESG_score.csv")[["country", "date", "E", "S", "G", "ESG"]]
ESG_Scores.date = pd.to_datetime(ESG_Scores.date)
df_id_country = pd.read_csv(path_file + "country_mapping.csv")
for country in ESG_Scores.country.unique():
    if country in df_id_country.country_list.values:
        ESG_Scores.loc[ESG_Scores.country == country, "country"] = df_id_country.loc[df_id_country.country_list == country, "iso3code"].values[0]    

month_spread_prediction = 12
# prediction either direction or value. 
all_data = prepare_all_data(spreads, all_data, normalized_data_from_countries, ESG_Scores, month_spread_prediction, prediction = "value")
all_data = all_data.drop(columns = ["year"])

################## drop na and 0 spread  ############
#WARNING: this means that we filter data even if we only use the 1 months or 3 months. 
#we filter even fro 12 months  
len(all_data) #7260
all_data["spread"] = all_data["spread"].replace(0., np.nan)
all_data = all_data[~all_data.isnull().any(axis=1)]
len(all_data) #3590 if we go until 12 month for mean and delta #3167 if we choose 24 
########################

#all_data_train = all_data[(all_data.Date < "2020-02-28")]
#all_data_test = all_data[(all_data.Date > "2020-08-31")]

#knowing that we have a at least 6 month lookback period we predict over 12 months, we cannpt have post covid as
# test set as it is too short.
# and we cannot predict 12 month as of 2019/2020 as it would be during the covid which is a big outlier!!

#all_data_train = all_data[(all_data.Date < "2017-10-01")]
#
#all_data_test = all_data[(all_data.Date > "2018-04-01") & (all_data.Date < "2019-02-28")]
#
#all_data_rest = all_data[(all_data.Date > "2019-02-28")]

all_data_jan = all_data[(all_data.Date.dt.month == 1)] #REMARK: could add june also??
all_data_train = all_data_jan[(all_data_jan.Date < "2017-10-01")]
all_data_test = all_data_jan[(all_data_jan.Date > "2018-04-01") & (all_data.Date < "2019-02-28")]
all_data_rest = all_data_jan[(all_data_jan.Date > "2019-02-28")]

X_train = all_data_train.drop(columns=["Date", "country", "spread"])
Y_train = all_data_train[["spread"]]

X_test = all_data_test.drop(columns=["Date", "country", "spread"])
Y_test = all_data_test[["spread"]]


len(X_train) #2403
len(X_test) #380

#################################### TRAIN MODELS - REGRESSIONS ##############################
#what would be the error if we assumed the spread constant?
mean_absolute_percentage_error(X_train.mean_spread_1, Y_train.spread) #0.26
mean_absolute_percentage_error(X_train.previous_delta_1, Y_train.spread) #0.26

mean_absolute_percentage_error(X_train.mean_spread_1 + X_train.previous_delta_3 , Y_train.spread) #0.26


mean_absolute_percentage_error([0] * len(Y_test), Y_test.spread) #0.21

#0.3 if 12 month 
result_regression = pd.DataFrame({})
for month_chosen in list_months:
    list_result = []
    to_remove = [x for x in list_months if x != month_chosen]
    list_drop = ["mean_spread_" + str(i) for i in to_remove]
    list_drop.extend(["previous_delta_" + str(i) for i in to_remove])
    
    X_train_regr = X_train.drop(columns = list_drop)
    
    scores = cross_validate(estimator=linear_model.LinearRegression(), X=X_train_regr.as_matrix(), y=Y_train.as_matrix(), cv=5, scoring='neg_mean_absolute_percentage_error', 
                            return_train_score=True)
    list_result.append(-np.mean(scores["test_score"]))
    
    scores = cross_validate(estimator=linear_model.Lasso(), X=X_train.as_matrix(), y=Y_train.as_matrix(), cv=5, scoring='neg_mean_absolute_percentage_error', 
                        return_train_score=True)
    list_result.append(-np.mean(scores["test_score"]))
    
    scores = cross_validate(estimator=linear_model.Ridge(), X=X_train.as_matrix(), y=Y_train.as_matrix(), cv=5, scoring='neg_mean_absolute_percentage_error', 
                        return_train_score=True)
    list_result.append(-np.mean(scores["test_score"]))
    
    result_regression["month_" + str(month_chosen)] = list_result

result_regression.index = ["Linear", "Lasso", "Ridge"]

########################### SELECT THE MOST IMPORTANT FEATURES ###################
list_drop = ["mean_spread_" + i for i in ["1", "3", "6"]]
list_drop.extend(["previous_delta_" + i for i in ["1", "3", "6"]])
#X_train = X_train.drop(columns = list_drop)

scores = cross_validate(estimator=linear_model.LinearRegression(), X=X_train.as_matrix(), y=Y_train.as_matrix(), cv=5, scoring='neg_mean_absolute_percentage_error', 
                return_train_score=True)
print(np.mean(-scores["test_score"]))
        
##############################################################################
from xgboost import XGBRFRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
#min_child_weight
#max_depth


columns_chosen = ["mean_spread_1", "Life expectancy at birth, total (years)", "E", "S", "G"]

X_train_test = X_train[columns_chosen]

model = XGBRFRegressor(n_estimators=10, max_depth=20, min_child_weight=2)#, eval_metric=mean_absolute_percentage_error) #max_depth
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1)       
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

model.fit(X_train, Y_train)

mean_absolute_percentage_error(Y_train, model.predict(X_train))
mean_absolute_percentage_error(Y_test, model.predict(X_test[columns_chosen]))

print(model.feature_importances_)

from xgboost import plot_importance
plot_importance(model)

plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()


###################################### GRID SEARCH     ########################
parameters = {"scoring":['neg_mean_absolute_percentage_error'],
              'learning_rate': [0.5, 0.8, 1], #so called `eta` value
              'max_depth': [10, 15, 40],
#              'min_child_weight': [4],
#              'silent': [1],
              'subsample': [0.9],
              'colsample_bynode': [0.1, 0.15],
              'n_estimators': [100]}

from sklearn.model_selection import GridSearchCV


xgb_grid = GridSearchCV(XGBRFRegressor(), parameters, cv = 5, n_jobs = 5, verbose=True)

xgb_grid.fit(X_train,Y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
##############################################################################
##############################  sign   #####################################
##############################################################################
Y_train_test = np.sign(Y_train)
Y_test_test = np.sign(Y_test)

model = XGBRFRegressor(n_estimators=1, max_depth=1, min_child_weight=20)#, eval_metric=mean_absolute_percentage_error) #max_depth
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, Y_train_test, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)       
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

mean_squared_error([1] * len(Y_train_test), Y_train_test)
mean_squared_error([-1] * len(Y_train_test), Y_train_test)


model.fit(X_train, Y_train_test)

mean_squared_error(Y_train_test,np.sign(model.predict(X_train)))
np.sum(Y_train_test.spread==np.sign(model.predict(X_train))) / len(Y_train_test)

mean_squared_error(Y_train_test, model.predict(X_train))
mean_squared_error(Y_test_test, model.predict(X_test))

##############################################################################

parameters = {"scoring":['neg_mean_squared_error'],
              'learning_rate': [0.5, 0.8, 1], #so called `eta` value
              'max_depth': [3, 6, 9],
             'min_child_weight': [3, 6, 10],
             'silent': [1],
              'subsample': [0.9],
              'colsample_bynode': [0.2],
              'n_estimators': [100]}

from sklearn.model_selection import GridSearchCV


xgb_grid = GridSearchCV(XGBRFRegressor(), parameters, cv = 5, n_jobs = 5, verbose=True)

xgb_grid.fit(X_train,Y_train_test)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


##############################################################################
##############################  classifier   #####################################
##############################################################################
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBRFClassifier



model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, Y_train_test, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))



parameters = {"scoring":['accuracy'],
              'learning_rate': [1e-4], #so called `eta` value
              'max_depth': [3],
             'min_child_weight': [2, 3],
             'silent': [1],
              'subsample': [0.9],
              'colsample_bynode': [0.2],
              'n_estimators': [100]}

from sklearn.model_selection import GridSearchCV


xgb_grid = GridSearchCV(XGBRFClassifier(), parameters, cv = 5, n_jobs = 5, verbose=True)

xgb_grid.fit(X_train,Y_train_test)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

accuracy_score(Y_train_test, [1] * len(Y_train_test))
accuracy_score(Y_train_test, np.sign(X_train.previous_delta_1))
accuracy_score(Y_train_test, np.sign(X_train.previous_delta_6))

##############################################################################
