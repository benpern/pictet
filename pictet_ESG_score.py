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
    for category in ESG_data_normalized_country.columns:
        if not category in ["Date", "date", "country"]:
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


###########     construct mapping ESG id - ESG name     ###########
id_ESG = []
name_ESG = []
for x in wbdata.get_indicator(source=75) :
    id_ESG.append(x["id"])
    name_ESG.append(x["name"])  
    
df_name_id = pd.DataFrame({"name": name_ESG, "id": id_ESG})


###########     read all ESG data      ###########
ESG_data = read_all_ESG_world_bank_data()

country_to_add = []
date_to_add = []
for x in ESG_data.index:
    country_to_add.append(x[0])
    date_to_add.append(x[1])
ESG_data["country"] = country_to_add
ESG_data["date"] = date_to_add


#########    normalize the data and change them to scores between 0 and 1   #########
# the variables that are already scores we do not normalize them. Only the other ones.
estimates = ["Government Effectiveness: Estimate", "Regulatory Quality: Estimate", "Voice and Accountability: Estimate",
           "Rule of Law: Estimate", "Political Stability and Absence of Violence/Terrorism: Estimate"]
not_estimates = ESG_data.columns.values.tolist()
for x in estimates:
    not_estimates.remove(x)
    
not_estimates.remove("date")
not_estimates.remove("country")

    
ESG_data_normalized = ESG_data.copy()
ESG_data_normalized[estimates] = ESG_data_normalized[estimates] / 4 + 0.5
ESG_data_normalized[not_estimates] = (ESG_data_normalized[not_estimates] - np.mean(ESG_data_normalized[not_estimates])) / (4 * np.std(ESG_data_normalized[not_estimates])) + 0.5
ESG_data_normalized.iloc[:,:-2] = np.minimum(ESG_data_normalized.iloc[:,:-2], 1)
ESG_data_normalized.iloc[:,:-2] = np.maximum(ESG_data_normalized.iloc[:,:-2], 0)


#################    computation of coefficients of variation   ##############
#to compute the CV coefficients we do not use the normalization shown above.
# that's in order to not modify the std. 
df_CV = pd.read_csv(path_file + "CV.csv")
#df_CV = get_variation_coefficients(ESG_data, ESG_data_normalized_country)
#df_CV.to_csv(path_file + "CV.csv")

#plot histogram of the CV
histogram_CV(df_CV)


#############      fill the NA with previous values when possible     ############
ESG_data_normalized_recent = ESG_data_normalized[ESG_data_normalized.date > "2000-01-01"]

ESG_data_normalized_recent = fill_na_with_older_values(ESG_data_normalized_recent, df_CV, 1, 2)



################## COMPUTE ESG SCORES ####################################
#hand written ESG file
pos_or_neg = pd.read_excel(path_file + "ESG.xlsx", sheet_name="ESG data") #know wether the variable is positive or negative
categories = pd.read_excel(path_file + "ESG.xlsx", sheet_name="weights") #categories and sub pillars


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
for pilar in categories["pilar"]:
    sub_pilar = categories.loc[categories.pilar == pilar, "sub pilar"].values.tolist()
    scores[pilar] = np.nan
    #if more than 50% of the grades are NA for a certain pilar the pilar will get NA
    scores.loc[np.sum(scores[sub_pilar].isna(), axis=1)/len(sub_pilar) <= 0.5, pilar] = np.mean(scores[sub_pilar], axis=1)

scores["ESG"] = 0.25 * scores["E"] + 0.25 * scores["S"] + 0.5 * scores["G"]
#scores.to_csv(path_file + "ESG_score.csv")




####################   check ESG scores and compare to GNI ###################
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