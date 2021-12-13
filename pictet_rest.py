#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:09:02 2021

@author: benoit
"""
import pandas as pd
import numpy as np
import wbdata
import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

path_file = "/Users/benoit/Downloads/Documents/Cover Letter-Resume/case_study_pictet/"

####### get climate data from climate_change_performance index
##### but actually is very sparse for our se tof data
climate_change = pd.read_excel(path_file + "climate_change_performance_index_2022.xlsx")                         

climate_change["id"] = np.nan
climate_change["name"] = np.nan
for i in np.arange(len(climate_change.Country)):
    country = climate_change.loc[i, "Country"]
    result = wbdata.search_countries(country)
    if len(result) > 0:
        climate_change.loc[i, "id"] = wbdata.search_countries(country)[0]["id"]
        climate_change.loc[i, "name"] = wbdata.search_countries(country)[0]["name"]



df_countries = pd.DataFrame({"id": country_list})
test = df_countries.merge(climate_change, on="id", how="left")





############# get names of SDG categories
id_SDG = []
name_SDG = []
#for x in wbdata.search_indicators("governance"):
for x in wbdata.get_indicator(source=46) :
    id_SDG.append(x["id"])
    name_SDG.append(x["name"]) 
    
    
    
    
    
    
###### missing categories   - does not work for homicides and le troisieme biodiversity
    #### les deux biodiversité pour lesquels ça amrche il y a aucune data
wbdata.search_indicators("biodiversity") # some categories could be interresting
# CC.KBA.MRN.ZS   Proportion of freshwater key biodiversity areas (KBAs) covered by protected areas (%)
# CC.KBA.TERR.ZS  Proportion of terrestrial key biodiversity areas (KBAs) covered by protected areas (%)
# ER.BDV.TOTL.XQ  GEF benefits index for biodiversity (0 = no biodiversity potential to 100 = maximum)
wbdata.search_indicators("homicides")
# VC.IHR.ICTS.P5     Intentional homicides, UN Crime Trends Survey (CTS) source (per 100,000 people)
# VC.IHR.PSRC.P5     Intentional homicides (per 100,000 people)
wbdata.search_indicators("GNI per capita")
# NY.GNP.MKTP.PC.CD     GNI per capita (US$)


list_id = ["CC.KBA.MRN.ZS", "CC.KBA.TERR.ZS", "ER.BDV.TOTL.XQ", "VC.IHR.ICTS.P5", "VC.IHR.PSRC.P5"]
list_name = ["Proportion of freshwater key biodiversity areas (KBAs) covered by protected areas (%)", 
             "Proportion of terrestrial key biodiversity areas (KBAs) covered by protected areas (%)",
             "GEF benefits index for biodiversity (0 = no biodiversity potential to 100 = maximum)",
             "Intentional homicides, UN Crime Trends Survey (CTS) source (per 100,000 people)",
             "Intentional homicides (per 100,000 people)"]

list_id = ["NY.GNP.MKTP.PC.CD"]
list_name = ["GNI per capita (US$)"]

GNI_per_capita = pd.DataFrame({})
indicators = {}
for i in np.arange(1):
    if len(ESG_data) == 0:
        GNI_per_capita = wbdata.get_dataframe({list_id[i]:list_name[i]}, convert_date=True)
    else:
        try:
            GNI_per_capita = GNI_per_capita.join(wbdata.get_dataframe({list_id[i]:list_name[i]}, convert_date=True))
        except:
            print(list_name[i])
            continue
        # indicators that failed:
        
ESG_data.to_csv(path_file + "/GNI_per_capita.csv")        
ESG_data.isna().sum()/len(ESG_data)
