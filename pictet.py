import pandas as pd
import numpy as np
import wbdata
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams.update({'font.size': 20})

path_file = "/Users/benoit/Downloads/Documents/Cover Letter-Resume/case_study_pictet/"

def plot_evol_quantiles_na(data_na, data, name_file):
    plt.figure(figsize=(20,10))
    for quantile in [0.1, 0.25, 0.5, 0.75, 0.9][::-1]:
        plt.plot(np.sort(np.unique(pd.to_datetime(data_na.index))), (data_na.quantile(quantile, axis=1))/number_countries, label="q=" + str(quantile))
    plt.legend()
    plt.grid()
    plt.ylabel("Proportion of countries/regions with a non-na data")
    plt.xlabel("Date")
    plt.title("Evolution of quantiles of the proportion of countries with available data for the " + str(ESG_data_normalized.shape[1] - 2) + " categories")
    plt.savefig(path_file + "graphs/" + name_file + ".png")
    plt.close()
    return

def get_variation_coefficients(ESG_data, ESG_data_normalized_country):
    nb_countries = []
    mean_CV = []
    median_CV = []
    max_CV = []
    for category in ESG_data.columns[2:]:
        ESG_data_cat = ESG_data_normalized_country[["country", category]] 
        
        ESG_data_cat = ESG_data_cat.dropna(axis=0)
            
        ESG_data_cat_grpby_std = ESG_data_cat.groupby("country").std()
        
        ESG_data_cat_grpby_mean = ESG_data_cat.groupby("country").mean()
        
        ESG_data_cat_grpby_std[category] = ESG_data_cat_grpby_std[category] /  (ESG_data_cat_grpby_mean[category] + 1e-10)
        
        ESG_data_cat["nb_years"] = 1
        ESG_data_cat_grpby_sum = ESG_data_cat[["country", "nb_years"]].groupby("country").sum()
        
        # if there is only one year then we would get a volatility of 0
        ESG_data_cat_grpby_std = ESG_data_cat_grpby_std[ESG_data_cat_grpby_sum.nb_years>=2]
        nb_countries.append(len(ESG_data_cat_grpby_std))
    
        if len(ESG_data_cat_grpby_std) > 0: 
            mean_CV.append(np.mean(ESG_data_cat_grpby_std.values))
            median_CV.append(np.median(ESG_data_cat_grpby_std))
            max_CV.append(np.max(ESG_data_cat_grpby_std.values))
        else:
            mean_CV.append(np.nan)
            median_CV.append(np.nan)
            max_CV.append(np.nan)
    
    
    df_CV = pd.DataFrame({"category":ESG_data.columns[2:], "nb_countries":nb_countries, "mean_CV":mean_CV, "median_CV":median_CV, "max_CV":max_CV})    
    return df_CV

def histogram_CV(df_CV):
    df_CV = df_CV.dropna(axis=0)
    plt.figure(figsize=(20,10))
    plt.hist(df_CV.max_CV, 20)
    plt.xlabel("Maximum of the coefficients of variation per category")
    plt.grid()
    plt.axvline(np.median(df_CV.max_CV),color="red", label="median: " + str(np.round(np.median(df_CV.max_CV), 2)))
    plt.axvline(np.quantile(df_CV.max_CV, 0.75),color="purple", label="75% quantile: " + str(np.round(np.quantile(df_CV.max_CV, 0.75), 2)))
    plt.legend()
    plt.title("Histogram of the maximum of the coefficients of variation \n per category for the set of countries given")
    plt.savefig(path_file + "graphs./histogram_max_coef_variation_countries.png")
    plt.close()
    return

def fill_na_with_older_values(ESG_data_normalized_country, df_CV, lookback_general, lookback_low_CV):
    for category in ESG_data_normalized_country.columns[2:]:
        if category not in df_CV.category.values:
            lookback = lookback_general
        elif df_CV.loc[df_CV.category == category, "max_CV"].values[0] < 1:
            lookback = lookback_low_CV
        else:
            lookback = lookback_general
        
        for country in ESG_data_normalized_country.country.unique():
            counter = 0
            previous = np.nan
            for date in np.sort(ESG_data_normalized_country.date.unique()):
                if np.isnan(ESG_data_normalized_country.loc[(ESG_data_normalized_country.country == country) & (ESG_data_normalized_country.date == date), category].values[0]):
                    ESG_data_normalized_country.loc[(ESG_data_normalized_country.country == country) & (ESG_data_normalized_country.date == date), category] = previous
                else:
                    previous = ESG_data_normalized_country.loc[(ESG_data_normalized_country.country == country) & (ESG_data_normalized_country.date == date), category].values[0]
                    counter = 0
                counter = counter + 1
                if counter > lookback:
                    previous = np.nan
                    counter = 0
    return ESG_data_normalized_country


def read_all_ESG_world_bank_data():
    ESG_data = pd.DataFrame({})
    indicators = {}
    for x in wbdata.get_indicator(source=75):
        if len(ESG_data) == 0:
    #        ESG_data = wbdata.get_dataframe({x["id"]:x["name"]}, country=country_list, convert_date=True)
            ESG_data = wbdata.get_dataframe({x["id"]:x["name"]}, convert_date=True)
        else:
            try:
    #            ESG_data = ESG_data.join(wbdata.get_dataframe({x["id"]:x["name"]}, country=country_list,convert_date=True))
                ESG_data = ESG_data.join(wbdata.get_dataframe({x["id"]:x["name"]}, convert_date=True))
            except:
                print(x["name"])
                continue
            # indicators that failed:
            # Methane emissions (kt of CO2 equivalent per capita)
            # Nitrous oxide emissions (metric tons of CO2 equivalent per capita)
            # Government expenditure on education, total (% of government expenditure)
    return ESG_data
    

def proportion_na(data):
    nb_na = data.iloc[:,2:].isna().sum().sum() 
    total = data.shape[0] * (data.shape[1] - 2)
    return nb_na/total

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

def plot_score_GNI(data, score):    
    reg = LinearRegression().fit(data["GNI per capita (US$)"].values.reshape((len(data), 1)), data[score])
    min_GNI = np.min(data["GNI per capita (US$)"])
    max_GNI = np.max(data["GNI per capita (US$)"])
    
    plt.figure(figsize=(20,10))
    plt.scatter(scores_countries_2019["GNI per capita (US$)"], scores_countries_2019[score], label=score + " score")
    plt.plot(np.array([min_GNI, max_GNI]).reshape((2, 1)), reg.predict(np.array([min_GNI, max_GNI]).reshape((2, 1))), label="linear regression, corr=" + str(np.round(np.corrcoef(data["GNI per capita (US$)"], data[score])[0,1], 2)), linewidth=5)
    plt.grid()
    plt.legend()
    plt.xlabel("GNI per capita (USD)")
    plt.ylabel("ESG score")
    plt.title(score + " score per GNI per capita for all countries available")
    plt.savefig(path_file + "graphs/link_GNIpercapita_with_" + score + "score.png")
    plt.close()
    return

wbdata.get_source() 
# Data that could be of interest:
# 6  International Debt Statistics
# 20  Quarterly Public Sector Debt
# 22  Quarterly External Debt Statistics SDDS
# 23  Quarterly External Debt Statistics GDDS
# 46  Sustainable Development Goals
# 75  Environment, Social and Governance (ESG) Data
# 81  International Debt Statistics: DSSI

wbdata.get_indicator(source=46) 

list_of_all_countries = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa',
       'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia',
       'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas, The',
       'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
       'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia',
       'Bosnia and Herzegovina', 'Botswana', 'Brazil',
       'British Virgin Islands', 'Brunei Darussalam', 'Bulgaria',
       'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon',
       'Canada', 'Cayman Islands', 'Central African Republic', 'Chad',
       'Channel Islands', 'Chile', 'China', 'Colombia', 'Comoros',
       'Congo, Dem. Rep.', 'Congo, Rep.', 'Costa Rica', "Cote d'Ivoire",
       'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic',
       'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',
       'Egypt, Arab Rep.', 'El Salvador', 'Equatorial Guinea', 'Eritrea',
       'Estonia', 'Eswatini', 'Ethiopia', 'Faroe Islands', 'Fiji',
       'Finland', 'France', 'French Polynesia', 'Gabon', 'Gambia, The',
       'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland',
       'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau',
       'Guyana', 'Haiti', 'Honduras', 'Hong Kong SAR, China', 'Hungary',
       'Iceland', 'India', 'Indonesia', 'Iran, Islamic Rep.', 'Iraq',
       'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan',
       'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
       "Korea, Dem. People's Rep.", 'Korea, Rep.', 'Kosovo', 'Kuwait',
       'Kyrgyz Republic', 'Lao PDR', 'Latvia', 'Lebanon', 'Lesotho',
       'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
       'Macao SAR, China', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
       'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
       'Mexico', 'Micronesia, Fed. Sts.', 'Moldova', 'Monaco', 'Mongolia',
       'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
       'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand',
       'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia',
       'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau',
       'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
       'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania',
       'Russian Federation', 'Rwanda', 'Samoa', 'San Marino',
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore',
       'Sint Maarten (Dutch part)', 'Slovak Republic', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan',
       'Spain', 'Sri Lanka', 'St. Kitts and Nevis', 'St. Lucia',
       'St. Martin (French part)', 'St. Vincent and the Grenadines',
       'Sudan', 'Suriname', 'Sweden', 'Switzerland',
       'Syrian Arab Republic', 'Tajikistan', 'Tanzania', 'Thailand',
       'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia',
       'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu',
       'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom',
       'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu',
       'Venezuela, RB', 'Vietnam', 'Virgin Islands (U.S.)',
       'West Bank and Gaza', 'Yemen, Rep.', 'Zambia', 'Zimbabwe']



#TODO: ORGANISE THE CODE IN FONCTIONS: reading data, preprocessing,.... Make several python files if too big??

spreads = pd.read_excel(path_file + "/spreads/2021_ESG_datascientist_casestudy_v2_data.xlsx")
#df.columns 
#'Date', 'ARG', 'BHR', 'BGD', 'BRB', 'BRA', 'CHL', 'CHN', 'COL', 'CRI',
#       'CZE', 'DOM', 'EGY', 'SLV', 'GEO', 'GHA', 'GTM', 'HKG', 'HUN', 'IND',
#       'IDN', 'IRQ', 'ISR', 'JAM', 'JOR', 'KAZ', 'KWT', 'LVA', 'MAC', 'MYS',
#       'MEX', 'MAR', 'NGA', 'OMN', 'PAN', 'PRY', 'PER', 'PHL', 'POL', 'QAT',
#       'RUS', 'SAU', 'SGP', 'ZAF', 'KOR', 'TWN', 'TZA', 'THA', 'TTO', 'TUR',
#       'UKR', 'ARE', 'ZWE']

country_list = list(spreads.columns[1:])


#date_test = datetime.datetime(2010, 1,1)
#
##appears to be working only with at most 2 dates at a time     
#wbdata.get_data(indicators, country=country_list, data_date = date_test)  
#
#check = wbdata.get_dataframe({x["id"]:x["name"]}, country=country_list, convert_date=True) 

id_ESG = []
name_ESG = []
#for x in wbdata.search_indicators("governance"):
for x in wbdata.get_indicator(source=75) :
    id_ESG.append(x["id"])
    name_ESG.append(x["name"])  
    
df_name_id = pd.DataFrame({"name": name_ESG, "id": id_ESG})



# ESG_data_countries.to_csv(path_file + "ESG_data_world_bank_specific_countries.csv")
        
# ESG_data.to_csv(path_file + "ESG_data_world_bank.csv")

ESG_data_countries = pd.read_csv(path_file + "ESG_data_world_bank_specific_countries.csv")

ESG_data = pd.read_csv(path_file + "ESG_data_world_bank.csv")

ESG_data_all_countries = ESG_data[ESG_data.country.isin(list_of_all_countries)]

for variable in ESG_data_all_countries.columns[2:]:
    plt_histogram(ESG_data_all_countries, variable, "histogram_raw_data")



# the variables that are already scores we do not normalize them. Only the other ones.
estimates = ["Government Effectiveness: Estimate", "Regulatory Quality: Estimate", "Voice and Accountability: Estimate",
           "Rule of Law: Estimate", "Political Stability and Absence of Violence/Terrorism: Estimate"]

not_estimates = ESG_data_normalized.columns[2:].values.tolist()
for x in estimates:
    not_estimates.remove(x)

np.min(ESG_data[estimates])
np.max(ESG_data[estimates])
np.mean(ESG_data[estimates])
np.std(ESG_data[estimates])
# appart frm the giny index, they seem to be norla variable with mean 0 and std 1.

ESG_data_normalized.columns[2:].values.tolist() - no_norm

ESG_data_normalized = ESG_data.copy()
ESG_data_normalized[estimates] = ESG_data_normalized[estimates] / 4 + 0.5

ESG_data_robust_scaler = ESG_data.copy()

#robust scaler:
ESG_data_robust_scaler[estimates] = ESG_data_robust_scaler[estimates] / 4 + 0.5
ESG_data_robust_scaler[not_estimates] = (ESG_data_robust_scaler[not_estimates] - ESG_data_robust_scaler[not_estimates].median()) / ( 4 * (ESG_data_robust_scaler[not_estimates].quantile(0.75) - ESG_data_robust_scaler[not_estimates].quantile(0.25))) + 0.5
ESG_data_robust_scaler.iloc[:,2:] = np.minimum(ESG_data_robust_scaler.iloc[:,2:], 1)
ESG_data_robust_scaler.iloc[:,2:] = np.maximum(ESG_data_robust_scaler.iloc[:,2:], 0)

ESG_data_normalized[not_estimates] = (ESG_data_normalized[not_estimates] - np.mean(ESG_data_normalized[not_estimates])) / (4 * np.std(ESG_data_normalized[not_estimates])) + 0.5
ESG_data_normalized.iloc[:,2:] = np.minimum(ESG_data_normalized.iloc[:,2:], 1)
ESG_data_normalized.iloc[:,2:] = np.maximum(ESG_data_normalized.iloc[:,2:], 0)

#values above 1 or below -1 should be capped to -1 and 1 



# we do min max normalization in order to compute the CV. For ESG grades computation 
#we use another normalization though
# ESG_data_normalized = ESG_data.copy()
# ESG_data_normalized.iloc[:,2:] = (ESG_data_normalized.iloc[:,2:] - np.min(ESG_data_normalized.iloc[:,2:])) / (np.max(ESG_data_normalized.iloc[:,2:]) - np.min(ESG_data_normalized.iloc[:,2:]))


ESG_data_normalized_country = ESG_data_normalized[ESG_data_normalized.country.isin(ESG_data_countries.country.unique())]

ESG_data_normalized_country_recent = ESG_data_normalized_country[ESG_data_normalized_country.date >= "2010-01-01"] 


ESG_data.shape # (3111, 66)

#We only need recent ESG data. 
ESG_data = ESG_data[ESG_data.date >= "2010-01-01"] 
ESG_data.shape # (561, 66)

check = ESG_data.isna().all()
ESG_data = ESG_data.loc[:,~check]
ESG_data.shape # (561, 60)
# columns with only na entries that we filter:
# Cooling Degree Days (projected change in number of degree Celsius)',
#       'GHG net emissions/removals by LUCF (Mt of CO2 equivalent)',
#       'Heat Index 35 (projected change in days)',
#       'Droughts, floods, extreme temperatures (% of population, average 1990-2009)',
#       'Maximum 5-day Rainfall, 25-year Return Level (projected change in mm)',
#       'Mean Drought Index (projected change, unitless)'


#TODO: continue investiguating the data!!!!!!

#if a grade is missing for the current year, we can use the grade from the previous year.
# we don't accept a 2 year difference though.

#For 2020, check the proportion of countries for which we would have an available score.
ESG_data_2019_2020 = ESG_data[ESG_data.date.isin(["2019-01-01", "2020-01-01"])]
ESG_data_2019_2020_grpby = ESG_data_2019_2020.groupby("country").sum()
proportion_of_available_data_2019_2020 = np.sum(ESG_data_2019_2020_grpby!=0, axis=0)/len(ESG_data_2019_2020_grpby)
proportion_of_available_data_2019_2020.to_csv(path_file + "proportion_available_ESG_data_2019_2020.csv")

proportion_of_available_data_2019_2020.describe()
# count    58.000000
# mean      0.520622
# std       0.453437
# min       0.000000
# 25%       0.000000
# 50%       0.705882
# 75%       0.995098
# max       1.000000

np.sum(proportion_of_available_data_2019_2020==0)/len(proportion_of_available_data_2019_2020) # 33%
# for 33% of the categories no country has any grade for 2019/2020

np.sum(proportion_of_available_data_2019_2020==1)/len(proportion_of_available_data_2019_2020) # 26%
# for 26% of the categories all the countries have grades for 2019/2020


#################   evolution of the number of na to identify the lack of data
ESG_data_normalized_na = ESG_data_normalized.copy()
ESG_data_normalized_na.iloc[:,2:] = ESG_data_normalized.iloc[:,2:].isna()
ESG_data_normalized_na = ESG_data_normalized_na.iloc[:,1:].groupby("date").sum()
number_countries = len(ESG_data_normalized.country.unique())
ESG_data_normalized_na = number_countries - ESG_data_normalized_na

plot_evol_quantiles_na(ESG_data_normalized_na, ESG_data_normalized, "evolution_non_NA")

ESG_data_normalized_na_zoom = ESG_data_normalized_na[ESG_data_normalized_na.index >= "2010-01-01"]
plot_evol_quantiles_na(ESG_data_normalized_na_zoom, ESG_data_normalized, "evolution_non_NA_zoom")

#################    CV PART
#knowing that one of the issues is that we don't have up to date data available::
# for such categories can we reuse older grades? For that we need to check the volatility of the grades 
# (not between different countries but the volatility of the grade through time)
# In order to be able to compare the volatilities, we will normalise the variables.

#coefficient of variation CV=standard deviation / mean
# https://www.researchgate.net/post/What-do-you-consider-a-good-standard-deviation

# computing the stf directly using groupby does not work probably due to some results
# for which there are only nas. Let's do a loop:

df_CV = get_variation_coefficients(ESG_data, ESG_data_normalized_country)
df_CV.to_csv(path_file + "variation_coefficient_countries.csv")
df_CV = pd.read_csv(path_file + "variation_coefficient_countries.csv")

#plot histogram of the CV
histogram_CV(df_CV)

ESG_data_normalized.shape #(16226, 67)
check = ESG_data_normalized.isna().all()
ESG_data_normalized = ESG_data_normalized.loc[:,~check]
ESG_data_normalized.shape #(16226, 63) -> 4 columns with only NA? same ones as the ones above?


###################   FILL NA PART
ESG_data_normalized_recent = ESG_data_normalized[ESG_data_normalized.date > "2000-01-01"]

proportion_na(ESG_data_normalized_recent) # 42%
proportion_na(ESG_data_normalized_recent[ESG_data_normalized_recent.country.isin(ESG_data_countries.country.unique())]) # 33%

ESG_data_normalized_recent = fill_na_with_older_values(ESG_data_normalized_recent, df_CV, 1, 2)

ESG_data_normalized_recent.to_csv(path_file + "/ESG_data_normalized_recent_filled_with_previous_scores_1_2_normal_normalization.csv")
ESG_data_normalized_recent = pd.read_csv(path_file + "/ESG_data_normalized_recent_filled_with_previous_scores_1_2_normal_normalization.csv")

for variable in ESG_data_normalized_recent.columns[3:]:
    plt_histogram(ESG_data_normalized_recent, variable, "histogram_normalized_data")

for variable in ESG_data_robust_scaler.columns[2:]:
    plt_histogram(ESG_data_robust_scaler, variable, "histogram_robust_scaled_data")

proportion_data_after_filling = pd.DataFrame((ESG_data_normalized_recent.shape[0] - ESG_data_normalized_recent.iloc[:,2:].isna().sum())/ESG_data_normalized_recent.shape[0])
proportion_data_after_filling.columns = ["since_2000"]
ESG_data_normalized_since_2010 = ESG_data_normalized_recent[ESG_data_normalized_recent.date >= "2010-01-01"]
proportion_data_after_filling["since_2010"] = (ESG_data_normalized_since_2010.shape[0] - ESG_data_normalized_since_2010.iloc[:,2:].isna().sum())/ESG_data_normalized_since_2010.shape[0]

proportion_data_after_filling.to_csv(path_file + "/proportion_data_per_category_after_filling.csv")
proportion_data_after_filling.describe()
#       sicne_2000  since_2010
#count   61.000000   61.000000
#mean     0.645110    0.654914
#std      0.278563    0.267678
#min      0.033835    0.014012
#25%      0.512594    0.469583
#50%      0.718609    0.759398
#75%      0.883459    0.883459
#max      0.994173    0.995557


categories_to_use = proportion_data_after_filling[proportion_data_after_filling.since_2010 > 0.46]

to_merge = pd.read_excel(path_file + "ESG.xlsx", sheet_name="ESG data")


proportion_data_after_filling = proportion_data_after_filling.merge(df_name_id, left_index=True, right_on="name", how="left")

result = to_merge.merge(proportion_data_after_filling, on="id", how="left")

result.to_excel(path_file + "to_copy_paste.xlsx")

# with lookback 1 always
ESG_data_normalized_country.isna().sum().sum() # 8390 (25%)

# with lookback 2 if max C.V < 1 
ESG_data_normalized_country.isna().sum().sum() # 7191 (22%)

# with lookback 3 if max C.V < 1 
ESG_data_normalized_country.isna().sum().sum() # 6352 (19%)


################## SEARCH DATA FOR MISSING CATEGORIES? - CHECKED - EMPTY
wbdata.search_indicators("waste") #nothing interresting it seems
wbdata.search_indicators("biodiversity") # some categories could be interresting
wbdata.search_indicators("security") # homicides from the SDG data

################ COMPUTATION OF THE ESG GRADES

pos_or_neg = pd.read_excel(path_file + "ESG.xlsx", sheet_name="ESG data")
categories = pd.read_excel(path_file + "ESG.xlsx", sheet_name="weights")

pos_or_neg = pos_or_neg.merge(df_name_id, on ="id", how="left")
pos_or_neg["name"] = pos_or_neg["name_y"].copy()

#prepare the file ESG_data_normalized_recent:
for name in pos_or_neg.name:
    if name in ESG_data_normalized_recent.columns:
        if (pos_or_neg.loc[pos_or_neg.name == name, "Positive or negative"] == "-").values[0]:
            ESG_data_normalized_recent[name] = 1 - ESG_data_normalized_recent[name]

#compute grades per sub_pilar
scores = pd.DataFrame({})
scores["country"] = ESG_data_normalized_recent.country.copy()
scores["date"] = ESG_data_normalized_recent.date.copy()
for sub_pilar in categories["sub pilar"]:
    categories_sub_pilar = categories[categories["sub pilar"] == sub_pilar]
    categories_sub_pilar = categories_sub_pilar.dropna(axis=1)
    categories_sub_pilar_id = categories_sub_pilar.iloc[:,3:]
    names = df_name_id[df_name_id.id.isin(categories_sub_pilar_id.values.tolist()[0])]
    scores[sub_pilar] = np.mean(ESG_data_normalized_recent[pd.Series(names.name).tolist()], axis=1)
    
#compute grades per pilar
# we gave same weight to all categories. If we want to increase the weight of the 
# climate for instance, we could add a copy of that column which would double its weight
for pilar in categories["pilar"]:
    sub_pilar = categories.loc[categories.pilar == pilar, "sub pilar"].values.tolist()
    scores[pilar] = np.nan
    #if more than 50% of the grades are NA for a certain pilar the pilar will get NA
    scores.loc[np.sum(scores[sub_pilar].isna(), axis=1)/len(sub_pilar) <= 0.5, pilar] = np.mean(scores[sub_pilar], axis=1)

scores["ESG"] = 0.25 * scores["E"] + 0.25 * scores["S"] + 0.5 * scores["G"]
scores.to_csv(path_file + "ESG_score.csv")
scores_countries = scores.iloc[980:,:]
scores_countries_now = scores_countries[scores_countries.date == "2020-01-01"][["country", "E", "S", "G", "ESG"]]
scores_countries_now = scores_countries_now.sort_values("ESG", ascending = False)

scores_countries_2019 = scores_countries[scores_countries.date == "2019-01-01"]

scores_countries_now = scores_countries_now.dropna() 
    

for variable in scores_countries_now.columns[2:]:
    plt_histogram(scores_countries_now, variable, "histogram_scores_2020")
    
score_countries_climate = scores_countries.sort_values("Climate", ascending = False)

####################  COMPARED ESG GRADES WITH GNI PER CAPITA. IS SUPPOSED TO BE HIGHLY CORRELATED
# compare only the G score and then compare the ESG score. DO we loose a lot with the whole ESG?
GNI_per_capita = pd.read_csv(path_file  + "/GNI_per_capita.csv")
np.max(GNI_per_capita.date) # 2019-01-01'
GNI_per_capita = GNI_per_capita[GNI_per_capita.date == "2019-01-01"]
GNI_per_capita = GNI_per_capita[GNI_per_capita.country.isin(scores_countries_2019.country)]

scores_countries_2019 = scores_countries_2019.merge(GNI_per_capita[["country", "GNI per capita (US$)"]], on="country", how="left")

scores_countries_2019 = scores_countries_2019.dropna(axis=0, subset=["GNI per capita (US$)", "ESG", "G", "S", "E"])

scores_countries_2019 = scores_countries_2019.sort_values("ESG", ascending = False)

np.corrcoef(scores_countries_2019["GNI per capita (US$)"], scores_countries_2019["ESG"]) #72%
np.corrcoef(scores_countries_2019["GNI per capita (US$)"], scores_countries_2019["G"]) #72%
np.corrcoef(scores_countries_2019["GNI per capita (US$)"], scores_countries_2019["S"]) #30%
np.corrcoef(scores_countries_2019["GNI per capita (US$)"], scores_countries_2019["E"]) #-20%

plot_score_GNI(scores_countries_2019, "E")

environment_sub_pilares = categories.loc[categories.pilar == "E", "sub pilar"].values
environment_categories = []
for sub_pilar in environment_sub_pilares:
    categories_sub_pilar = categories[categories["sub pilar"] == sub_pilar]
    categories_sub_pilar = categories_sub_pilar.dropna(axis=1)
    categories_sub_pilar_id = categories_sub_pilar.iloc[:,3:]
    environment_categories.extend(df_name_id[df_name_id.id.isin(categories_sub_pilar_id.values.tolist()[0])]["name"].values.tolist())


check_rwanda = ESG_data_normalized_recent[ESG_data_normalized_recent.country == "Latvia"]
check_rwanda = check_rwanda[check_rwanda.date == "2019-01-01"]
check_rwanda = check_rwanda[environment_categories]

check_singap = ESG_data_normalized_recent[ESG_data_normalized_recent.country == "Switzerland"]
check_singap = check_singap[check_singap.date == "2019-01-01"]
check_singap = check_singap[environment_categories]

check_rwanda.index = ["Latvia"]
check_singap.index = ["Switzerland"]

compare = pd.concat([check_rwanda, check_singap])

####################  COMPARED CLIMATE GRADES WITH the following climate grades
climate_change = pd.read_excel(path_file + "climate_change_performance_index_2022.xlsx")                         


ESG_data_normalized_country_dates = ESG_data_normalized_country[ESG_data_normalized_country.date.isin(["2019-01-01", "2020-01-01"])]
ESG_data_2019_2020_grpby = ESG_data_2019_2020.groupby("country").sum()
proportion_of_available_data_2019_2020 = np.sum(ESG_data_2019_2020_grpby!=0, axis=0)/len(ESG_data_2019_2020_grpby)

