#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 16:36:29 2021

@author: benoit
"""

import pandas as pd
import numpy as np
import wbdata
import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

path_file = "/Users/benoit/Downloads/Documents/Cover Letter-Resume/case_study_pictet/spreads/"

def plot_evol_quantiles_na(data, title, with_maximum):
    plt.figure(figsize=(20,10))
    if with_maximum:
        plt.plot(pd.to_datetime(data.Date), np.max(data, axis=1), label="maximum")
        
    for quantile in [0.1, 0.25, 0.5, 0.75, 0.9][::-1]:
        plt.plot(pd.to_datetime(data.Date), (data.quantile(quantile, axis=1)), label="q=" + str(quantile))
        
    plt.plot(pd.to_datetime(data.Date), np.min(data, axis=1), label="minimum")  
    plt.axvline(datetime.date(day = 28, month=2, year = 2020), color="black", linestyle = "--", label="2020-02-28", alpha=0.5)
    plt.axvline(datetime.date(day = 30, month=8, year = 2020), color="brown", linestyle = "--", label="2020-08-30")
    plt.legend()
    plt.grid()
    plt.ylabel("Spread")
    plt.xlabel("Date")
    plt.title("Evolution of quantiles of the spreads")
    plt.savefig(path_file + "graphs/" + title+".png")
    plt.close()
    return

def plt_evol_all_spreads(data, title):
    list_linestyle = ["solid", "dashed", "dashdot", "dotted"]
    plt.figure(figsize=(20,10))
    n = 14 # number of different colors
    list_colors = plt.cm.jet(np.linspace(0, 1, n))
    index = 0
    index_n = 0
    for country in data.columns[1:]:
        if index % n == 0:
            linestyle = list_linestyle[index_n]
            index_n = index_n + 1 
        plt.plot(pd.to_datetime(data.Date), data[country], label=country, linestyle = linestyle, color=list_colors[index % n])
        index = index + 1 
     
    plt.axvline(datetime.date(day = 28, month=2, year = 2020), color="black", linestyle = "--", label="2020-02-28", alpha=0.8, linewidth=5)
    plt.axvline(datetime.date(day = 31, month=8, year = 2020), color="brown", linestyle = "--", label="2020-08-31", alpha=0.8, linewidth=5)
    plt.legend(ncol=4, fontsize=10)
    plt.grid()
    plt.ylabel("Spread")
    plt.xlabel("Date")
    plt.title("Evolution of the spreads per country")
    plt.savefig(path_file + "graphs/" + title+".png")
    plt.close()
    return

spreads = pd.read_excel(path_file + "2021_ESG_datascientist_casestudy_v2_data.xlsx")

spreads.Date.describe()
# date of spreads approx every 2 months frm 2010-02-26 to 2021-08-31


np.sum(spreads.isna(), axis=1) / spreads.shape[1]

plt.figure(figsize=(20,10))
plt.plot(spreads.Date, np.sum(spreads.isna(), axis=1) / spreads.shape[1])
plt.grid()
plt.ylabel("Proportion of spreads not available")
plt.title("Evolution of the proportion of spreads not available")
plt.savefig(path_file + "/graphs/evolution_na_spreads.png")
plt.close()

countries = spreads.columns[1:].values.tolist()

#########    proportion of na spreads   #########
prop_na_spread_per_country = np.sum(spreads.isna(), axis=0) / spreads.shape[0]
prop_na_spread_per_country = prop_na_spread_per_country.sort_values(ascending=False)
prop_na_spread_per_country.to_csv(path_file + "proportion_na_spreads_per_country.csv")


#########    evolution of the spreads  #########
title = "evolution_quantiles_spreads_without_maximum"
plot_evol_quantiles_na(spreads, title, with_maximum = False)
# we clearly see that over the covid period the spreads increased drastically. We will filter out this part

plt_evol_all_spreads(spreads, "evolution_spreads_per_country")

spreads_without_covid = spreads[(spreads.Date < "2020-02-28") | (spreads.Date > "2020-08-31")]

spreads_pre_covid = spreads[spreads.Date < "2020-02-28"]
spreads_post_covid = spreads[spreads.Date > "2020-08-31"]

####### CHECK 0 valuesfor the metric #######
check = spreads[spreads ==  0]
check = check.dropna(axis=0, how="all")
check = check.dropna(axis=1, how="all")
# we suggest to replace these entries by na as it is common and may seem weird and won't have an impact 

spreads = spreads.replace(0., np.nan)


#########     CROSS VALIDATION     #########

sets = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, "pre_covid", "post_covid"]
histo_error_one_month = []
histo_error_three_month = []
histo_error_six_month = []

for index_test_set in sets:
    spreads_first = pd.DataFrame({})
    spreads_second = pd.DataFrame({})
    spreads_third = pd.DataFrame({})
    prediction = pd.DataFrame({})
    
    if index_test_set == "pre_covid":
        spreads_test = spreads_without_covid[(spreads_without_covid.Date >= "2019-01-01") & (spreads_without_covid.Date < "2020-02-28")]
        spreads_first = spreads_without_covid[(spreads_without_covid.Date < "2019-01-01")]
        spreads_second = spreads_post_covid.copy()
    elif index_test_set == "post_covid":
        spreads_test = spreads_without_covid[(spreads_without_covid.Date > "2020-08-31")]
        spreads_first = spreads_pre_covid.copy()
    else:
        spreads_test = spreads_without_covid[spreads_without_covid.Date.dt.year == index_test_set]
        spreads_first = spreads_without_covid[spreads_without_covid.Date.dt.year < index_test_set]
        spreads_second = spreads_without_covid[(spreads_without_covid.Date.dt.year > index_test_set) & (spreads_without_covid.Date.dt.year < 2020)]
        spreads_third = spreads_post_covid.copy()

    # CONSTANT PREDICTION
    prediction = spreads_test.iloc[:-1,:]
    prediction.index = spreads_test.Date[1:].values
    spreads_test = spreads_test.iloc[1:,:]
    spreads_test.index = spreads_test.Date
    histo_error_one_month.append(np.mean(np.mean(np.abs(prediction[countries] - spreads_test[countries])/spreads_test[countries], axis=0)))
    
    prediction = spreads_test.iloc[:-3,:]
    prediction.index = spreads_test.Date[3:].values
    spreads_test = spreads_test.iloc[3:,:]
    spreads_test.index = spreads_test.Date
    histo_error_three_month.append(np.mean(np.mean(np.abs(prediction[countries] - spreads_test[countries])/spreads_test[countries], axis=0)))

    prediction = spreads_test.iloc[:-6,:]
    prediction.index = spreads_test.Date[6:].values
    spreads_test = spreads_test.iloc[6:,:]
    spreads_test.index = spreads_test.Date
    histo_error_six_month.append(np.mean(np.mean(np.abs(prediction[countries] - spreads_test[countries])/spreads_test[countries], axis=0)))

result = pd.DataFrame([histo_error_one_month], columns = sets)

result = pd.concat([result, pd.DataFrame([histo_error_three_month], columns = sets)])
result = pd.concat([result, pd.DataFrame([histo_error_six_month], columns = sets)])
result.index = ["one month", "three month", "six month"]





#    print(len(spreads_test))   12 for the years, 13 for the pre and post covid
    
    


train_set = spreads[spreads.Date < "2020-02-28"]
test_set = spreads[spreads.Date > "2020-08-31"]

#########       CONSTANT prediction     #########
#1 month
prediction = test_set.olic[:-1,:]
test_set["next_date"] = test_set.Date[1:]

