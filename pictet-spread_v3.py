#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:02:15 2021

@author: benoit
"""

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

month_spread_prediction = 6
# prediction either direction or value. 
all_data = prepare_all_data(spreads, all_data, normalized_data_from_countries, ESG_Scores, month_spread_prediction, prediction = "direction")
all_data = all_data.drop(columns = ["year"])


all_data_jan = all_data[all_data.Date.dt.month.isin([3, 9])] #REMARK: could add june also??
all_data_train = all_data_jan[(all_data_jan.Date < "2017-10-01")]
all_data_test = all_data_jan[(all_data_jan.Date > "2018-04-01") & (all_data.Date < "2019-02-28")]
all_data_rest = all_data_jan[(all_data_jan.Date > "2019-02-28")]

X_train = all_data_train.drop(columns=["Date", "country", "spread"])
Y_train = np.sign(all_data_train[["spread"]])

X_test = all_data_test.drop(columns=["Date", "country", "spread"])
Y_test = all_data_test[["spread"]]

from xgboost import XGBRFRegressor, XGBRFClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
result = pd.DataFrame({})
for country in all_data_train.country.unique():
    all_data_train_country = all_data_train[all_data_train.country == country]
    all_data_train_country["spread"] = all_data_train_country["spread"].replace(0., np.nan)
    all_data_train_country = all_data_train_country[~all_data_train_country.isnull().any(axis=1)]
    
    X_train = all_data_train_country.drop(columns=["Date", "country", "spread"])[["mean_spread_1", "previous_delta_6", "S", "G", 'CPI Price,not seas.adj,,,', 'Consumer price index (2010 = 100)',
       'Inflation, consumer prices (annual %)',
       'Life expectancy at birth, total (years)', 'GDP growth (annual %)',
       'Unemployment, total (% of total labor force) (modeled ILO estimate)',
       'International debt issues to GDP (%)']]
    Y_train = np.sign(all_data_train_country[["spread"]])
    
    if len(X_train) > 5:
        model = XGBRFClassifier(n_estimators=10, max_depth=5, min_child_weight=2)#, eval_metric=mean_absolute_percentage_error) #max_depth
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores_RF = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1) 
        
        model = linear_model.RidgeClassifier()#, eval_metric=mean_absolute_percentage_error) #max_depth
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores_LR = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)       

        result = pd.concat([result, pd.DataFrame({"country":[country],"nb_instances":[len(X_train)], 
                                                  "ref_6_month": [np.round(accuracy_score(np.sign(X_train.previous_delta_6), Y_train.spread), 2)], 
#                                                 "ref_3_month": [np.round(accuracy_score(np.sign(X_train.previous_delta_3), Y_train.spread), 2)], 
#                                                  "ref_1_month": [np.round(accuracy_score(np.sign(X_train.previous_delta_1), Y_train.spread), 2)], 
                                                    "model_RF": [np.mean(n_scores_RF)],
                                                  "model_Ridge_classifier": [np.mean(n_scores_LR)]})])

            
result.to_csv(path_file + "month_production_" + str(month_spread_prediction) + "_less_predictors.csv")  
    

for country in spreads.columns[1:]: 
    print(country)
    plt.figure(figsize=(20,10))
    plt.plot(spreads.Date, spreads[country])
    plt.plot(spreads.Date, spreads[country].rolling(window=3).mean(), label="3 month MA")
    plt.plot(spreads.Date, spreads[country].rolling(window=6).mean(), label="6 month MA")
    plt.legend()
    plt.grid()
    plt.ylabel("spread")
    plt.title("Evol spread of " + country)
    plt.savefig(path_file + "graphs/evol_spreads_per_country/" + country + ".png")
    plt.close()
    
    

#########################     ARIMA MODEL         ##############################
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
for country in spreads.columns[1:]:
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(spreads[country]); axes[0, 0].set_title('Original Series')
    plot_acf(spreads[country], ax=axes[0, 1])
    
    # 1st Differencing
    axes[1, 0].plot(spreads[country].diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(spreads[country].diff().dropna(), ax=axes[1, 1])
    
    # 2nd Differencing
    axes[2, 0].plot(spreads[country].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(spreads[country].diff().diff().dropna(), ax=axes[2, 1])
    
    plt.savefig(path_file + "graphs/ARIMA/auto_correl" + country + ".png")
    plt.close()
-> d = 1


# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(spreads.BRA.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(spreads.BRA.diff().dropna(), ax=axes[1])

plt.show()

-> p = 1 


import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(spreads.ARG.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(spreads.ARG.diff().dropna(), ax=axes[1])

plt.show()

-> q = 1 
