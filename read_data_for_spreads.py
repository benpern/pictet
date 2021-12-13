#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 12:26:59 2021

@author: benoit
"""

import pandas as pd
import numpy as np
import wbdata
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams.update({'font.size': 20})

wbdata.get_source() 
# Data that could be of interest:
# 6  International Debt Statistics
# 20  Quarterly Public Sector Debt
# 22  Quarterly External Debt Statistics SDDS
# 23  Quarterly External Debt Statistics GDDS
# 46  Sustainable Development Goals
# 75  Environment, Social and Governance (ESG) Data
# 81  International Debt Statistics: DSSI

def get_data(list_id_cat, countries):
    data = pd.DataFrame({})
    name_list = []   
    id_list = []
    for x in list_id_cat:
        if len(data) == 0:
            name = wbdata.get_data(x, country="USA", data_date=datetime.datetime(2019,1,1))[0]["indicator"]["value"]
            data = wbdata.get_dataframe({x:name}, country=countries, convert_date=True)
            name_list.append(name)
            id_list.append(x)
        else:
            try:
                name = wbdata.get_data(x, country="USA", data_date=datetime.datetime(2019,1,1))[0]["indicator"]["value"]
                data = data.join(wbdata.get_dataframe({x:name}, country=countries,convert_date=True))
                name_list.append(name)
                id_list.append(x)
            except:
                print(x)
                continue
            # indicators that failed:
            # Methane emissions (kt of CO2 equivalent per capita)
            # Nitrous oxide emissions (metric tons of CO2 equivalent per capita)
            # Government expenditure on education, total (% of government expenditure)
    df_id_name = pd.DataFrame({"id":id_list, "name":name_list})
    return data, df_id_name

def get_mapping_country_code(country_list):
    id_list = []
    list_of_countries = []
    country_iso3code = []
    
    for identifier in country_list:
        print(identifier)
        try:
            result = wbdata.get_data("IC.BUS.EASE.XQ", country=identifier, data_date=datetime.datetime(2019,1,1))
            id_list.append(result[0]["country"]["id"])
            list_of_countries.append(result[0]["country"]["value"])
            country_iso3code.append(result[0]["countryiso3code"])
        except:
            print("DID not work")
            continue
    
    return pd.DataFrame({"id": id_list, "country_list": list_of_countries, "iso3code":country_iso3code})

def get_variation_coefficients(data):
    data = data
    nb_countries = []
    mean_CV = []
    median_CV = []
    max_CV = []
    for category in data.columns[2:]:
        data_cat = data[["country", category]] 
        
        data_cat = data_cat.dropna(axis=0)
            
        data_cat_grpby_std = data_cat.groupby("country").std()
        
        data_cat_grpby_mean = data_cat.groupby("country").mean()
        
        data_cat_grpby_std[category] = data_cat_grpby_std[category] /  (data_cat_grpby_mean[category] + 1e-10)
        
        data_cat["nb_years"] = 1
        data_cat_grpby_sum = data_cat[["country", "nb_years"]].groupby("country").sum()
        
        # if there is only one year then we would get a volatility of 0
        data_cat_grpby_std = data_cat_grpby_std[data_cat_grpby_sum.nb_years>=2]
        nb_countries.append(len(data_cat_grpby_std))
    
        if len(data_cat_grpby_std) > 0: 
            mean_CV.append(np.mean(data_cat_grpby_std.values))
            median_CV.append(np.median(data_cat_grpby_std))
            max_CV.append(np.max(data_cat_grpby_std.values))
        else:
            mean_CV.append(np.nan)
            median_CV.append(np.nan)
            max_CV.append(np.nan)
    
    
    df_CV = pd.DataFrame({"category":data.columns[2:], "nb_countries":nb_countries, "mean_CV":mean_CV, "median_CV":median_CV, "max_CV":max_CV})    
    return df_CV

def fill_na_with_older_values(data, df_CV, lookback_general, lookback_low_CV):
    categories = data.columns.tolist()
    categories.remove("country")
    categories.remove("Date")
    for category in categories:
        if category not in df_CV.category.values:
            lookback = lookback_general
        elif df_CV.loc[df_CV.category == category, "max_CV"].values[0] < 1:
            lookback = lookback_low_CV
        else:
            lookback = lookback_general
        
        for country in data.country.unique():
            counter = 0
            previous = np.nan
            for date in np.sort(data.Date.unique()):
                if np.isnan(data.loc[(data.country == country) & (data.Date == date), category].values[0]):
                    data.loc[(data.country == country) & (data.Date == date), category] = previous
                else:
                    previous = data.loc[(data.country == country) & (data.Date == date), category].values[0]
                    counter = 0
                counter = counter + 1
                if counter > lookback:
                    previous = np.nan
                    counter = 0
    return data

def plot_nas(df_recent): 
    df_na = df_recent.isnull().sum(axis=1)
    df_na_gr = df_na.groupby("date").sum()
    plt.figure(figsize=(20,10))
    plt.plot(df_na_gr.index, (len(df_recent.country.unique()) * (df_recent.shape[1] - 2)  - df_na_gr)/(len(df_recent.country.unique()) * (df_recent.shape[1] - 2)))
    plt.grid()
    plt.ylabel("Proportion of available data")
    plt.title("Evolution of the proportion of available data per date")
    plt.savefig(path_file + "graphs/evolution_na_categories.png")
    plt.close()
    
    plt.figure(figsize=(20,10))
    for category in df_recent.columns[:-2]:
        df_cat = df_recent[[category, "Date"]]
        df_cat_na = df_cat.isnull().sum(axis=1)
        df_cat_na_gr = df_cat_na.groupby("date").sum()
        plt.plot(df_cat_na_gr.index, (len(df_recent.country.unique())  - df_cat_na_gr)/(len(df_recent.country.unique())), label=category)
    plt.grid()
    plt.legend(fontsize = 15)
    plt.ylabel("Proportion of available data")
    plt.title("Evolution of the proportion of available data per category per date")
    plt.savefig(path_file + "graphs/evolution_na_categories_per_cat.png")
    plt.close()
    
    
    df_na = df_recent.isnull().sum(axis=1)
    df_na_gr = df_na.groupby("country").sum()
    df_na_gr = df_na_gr/(len(df_recent.Date.unique()) * (df_recent.shape[1] - 2))
    df_na_gr = df_na_gr[df_na_gr>0.3]
    plt.figure(figsize=(20,10))
    plt.bar(df_na_gr.index, df_na_gr)
    plt.xticks(rotation=30, fontsize=15)
    plt.grid()
    plt.ylabel("Proportion of NA")
    plt.title("proportion of NA per country - for countries with prop above 30%")
    plt.savefig(path_file + "graphs/proportion_NA_per_country.png")
    plt.close()
    
    return

def plt_histogram(data, variable, folder_save):
    data = data.dropna(axis=0, subset=[variable])
    if len(data) != 0:
        plt.figure(figsize=(20,10))
        plt.hist(data[variable])
        plt.axvline(np.quantile(data[variable], 0.75), color="orange", label="q75%=" + str(np.round(np.quantile(data[variable], 0.75), 2)), linewidth = 5)
        plt.axvline(np.median(data[variable]), color="red", label="median=" + str(np.round(np.median(data[variable]), 2)), linewidth = 5)
        plt.axvline(np.quantile(data[variable], 0.25), color="green", label="q25%=" + str(np.round(np.quantile(data[variable], 0.25), 2)), linewidth = 5)
    
        plt.grid()
        plt.legend()
        plt.xlabel(variable)
        plt.title("Histogram of " + variable)
        plt.savefig(path_file + "graphs/" + folder_save + "/histo_" + variable.replace("/", " per ") + ".png")
        plt.close()
    return

path_file = "/Users/benoit/Downloads/Documents/Cover Letter-Resume/case_study_pictet/spreads/"

spreads = pd.read_excel(path_file + "2021_ESG_datascientist_casestudy_v2_data.xlsx")

country_list = list(spreads.columns[1:])
  
##########################       READ DATA      ##########################

df_id_country = get_mapping_country_code(country_list) # there might be some missing values
# in case it did not work. We won't consider the few ones that failed  (only DID)??
df_id_country.to_csv(path_file + "country_mapping.csv")

#CPI, inflation, unemployment, life expectancy, GDP growth, debt
list_id_cat = ["CPTOTNSXN", "FP.CPI.TOTL", "FP.CPI.TOTL.ZG", "NY.GDP.DEFL.87.ZG", 
               "JI.UEM.1524.FE.ZS", "JI.UEM.1564.OL.ZS", "SP.DYN.LE00.IN", "NY.GDP.MKTP.KD.ZG", "SL.UEM.TOTL.ZS", "GFDD.DM.07", "GC.DOD.TOTL.GD.ZS"]

df, df_id_name = get_data(list_id_cat, country_list)
df["country"] = [x[0] for x in df.index.values]
df["Date"] = [x[1] for x in df.index.values]

df_recent = df[df.Date >= datetime.datetime(2010,1,1)]

df = df.drop(columns=["Youth unemployment rate, aged 15-24, female (% of female youth labor force)", 
                      "Unemployment rate, aged 25-64 (% of labor force aged 25-64)",
                      "Central government debt, total (% of GDP)"])

df = df.sort_values(["country", "Date"])
##########################       PROCESS THE DATA      ##########################

#df.isnull().sum()/len(df) -> unemployment rate, youth unemployment rate and central governemnt debt 
#are around 75% of the time NA. Let's drop them
df_recent.isnull().sum()/len(df_recent)
df_recent = df_recent.drop(columns=["Youth unemployment rate, aged 15-24, female (% of female youth labor force)", 
                      "Unemployment rate, aged 25-64 (% of labor force aged 25-64)",
                      "Central government debt, total (% of GDP)"])

#plot_nas(df_recent)

#extend life of variables (obliged as we only have data until 2019) and we will want to predict until 2022...
df_CV = pd.DataFrame({"category":[""]})
df_recent = fill_na_with_older_values(df_recent, df_CV, 1, 1)

df_na = df_recent.isnull().sum(axis=1)
df_na_gr = df_na.groupby("date").sum()
plt.figure(figsize=(20,10))
plt.plot(df_na_gr.index, (len(df_recent.country.unique()) * (df_recent.shape[1] - 2)  - df_na_gr)/(len(df_recent.country.unique()) * (df_recent.shape[1] - 2)))
plt.grid()
plt.ylabel("Proportion of available data")
plt.title("Evolution of the proportion of available data per date after fill 1 ")
plt.savefig(path_file + "graphs/evolution_na_categories_after_fill.png")
plt.close()

df_na = df_recent.isnull().sum(axis=0)
df_na = df_na[df_na.index!="Date"]
df_na = df_na[df_na.index!="country"]
df_na = df_na/len(df_recent)
plt.figure(figsize=(20,10))
plt.bar(df_na.index, df_na)
plt.xticks(rotation=8, fontsize=12)
plt.grid()
plt.ylabel("Proportion of NA")
plt.title("proportion of NA per category after fill")
plt.savefig(path_file + "graphs/proportion_NA_per_category_after_fill.png")
plt.close()

##### check data + normalize

for name in df_recent.columns[:-2]:
    plt_histogram(df_recent, name, folder_save="histogram_cat")
    
normalized = df_recent.copy()    

for cat in normalized.columns[:-2]:
    normalized[cat] = (normalized[cat] - np.mean(normalized[cat])) / np.std(normalized[cat])
    plt_histogram(normalized, cat, folder_save="histogram_cat_normalized")

for country in normalized.country.unique():
    if country in df_id_country.country_list.values:
        normalized.loc[normalized.country == country, "country"] = df_id_country.loc[df_id_country.country_list == country, "iso3code"].values[0]
normalized.index = np.arange(len(normalized))
normalized.to_csv(path_file + "normalized_data_recent.csv")


