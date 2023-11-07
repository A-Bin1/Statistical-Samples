#create Naive Bayes Model with Oregon Car Crash data by County for 2019
#data courtesy of https://www.oregon.gov/odot/Data/Pages/Traffic-Counting.aspx
#imported as flat file csv format from github

url = 'https://raw.githubusercontent.com/A-Bin1/Statistical-Samples/main/2019_OR_cty_car_crash_data.csv'

df = pd.read_csv(url)

import pandas as pd
import numpy as np
df = pd.read_csv(url)
#calculate raw probabilities for sections below:

#calculate percent of fatal crashes per county
df['Percent_Fatal'] = df['FATAL_CRASHES']/df['TOTAL_CRASHES']
#calculate percent of non-fatal crashes per county
df['Percent_Non_Fatal'] = df['NON_FATAL_CRASHES']/df['TOTAL_CRASHES']
#calculate percent of dry surface crashes
df['Percent_DrySurf'] = df['DRY_SURF']/df['TOTAL_CRASHES']
#wet surface %
df['Percent_WetSurf'] = df['WET_SURF']/df['TOTAL_CRASHES']
#day conditions %
df['Percent_DayCond'] = df['DAY_COND']/df['TOTAL_CRASHES']
#dark conditions %
df['Percent_DarkCond'] = df['DARK_COND']/df['TOTAL_CRASHES']
#offroad %
df['Percent_OffRoad'] = df['OFFROAD']/df['TOTAL_CRASHES']

#Note that the total crashes population in dataset include those not classified as fatal OR non-fatal
car_crashdf = pd.DataFrame(df['CNTY_NM'])
car_crashdf['FATAL'] = df['Percent_Fatal']
car_crashdf['NON_FATAL'] = df['Percent_Non_Fatal']
car_crashdf['DAY_COND'] = df['Percent_DayCond']
car_crashdf['DRY_COND'] = df['Percent_DrySurf']
car_crashdf['OFFROAD'] = df['Percent_OffRoad']

#original data given by county -- group counties by regions for
#naive bayes model classifiers
region_list = list(car_crashdf['CNTY_NM'])

#example 1: classify by coastal region vs non-coastal
coastal_region =['Curry', 'Coos', 'Douglas','Lane','Lincoln','Tillamook', 'Clatsop']
not_coastal_region = [county for county in region_list if county not in coastal_region]

#example 2: classify by southern region vs non-southern
southern_region = ['Lane', 'Douglas', 'Coos', 'Curry', 'Josephine', 'Jackson', 'Klamath']
not_southern_region = [county for county in region_list if county not in southern_region]

#reshape data into two new category columns to show new classifiers
#coastal vs non-coastal
car_crashdf['Coastal_YN']= np.where(car_crashdf['CNTY_NM'].isin(coastal_region), 'Coastal', 'Non-Coastal')
#southern vs non-southern
car_crashdf['Southern_Region_YN']= np.where(car_crashdf['CNTY_NM'].isin(southern_region), 'Southern', 'Non-Southern')


#Naive Bayes model for coastal region vs non-coastal data for non-fatal crashes given day conditions
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
X, y = car_crashdf[['NON_FATAL','DAY_COND']], car_crashdf['Coastal_YN']
classifier.fit(X, y)


print(classifier.predict_proba([[1,1]])[:,0])  # Pr(C='Coastal'|A=probability%, B=probability%)
#[6.05550291e-71]
print(classifier.predict_proba([[1,1]])[:,1])  # Pr(C='Non-Coastal'|A=probability%, B=probability%)
#[1.]
#note this model is most likely overfitted or heavily reliant on training data, sample size

#The colorbar shows the probability of an Oregon coast car crash
# given Non-fatal Oregon car crash in day conditions. Original data points plotted in black and white.
#plot courtesy of 
# https://stackoverflow.com/questions/52076415/calculating-conditional-probabilities-for-categorical-and-continuous-variables-i


import matplotlib.pylab as plt
X1, X2 = np.meshgrid(np.linspace(X[['NON_FATAL']].min(), X[['NON_FATAL']].max(),10), np.linspace(X[['DAY_COND']].min(), X[['DAY_COND']].max(),10))
plt.figure(figsize=(10,6))
# plot the probability surface for Oregon Coast
plt.contourf(X1, X2, classifier.predict_proba(np.c_[X1.ravel(), X2.ravel()])[:,0].reshape(X1.shape), cmap='jet', alpha=.8)
plt.colorbar()
cols = {'Coastal':'black', 'Non-Coastal':'white'}
plt.scatter(X[['NON_FATAL']], X[['DAY_COND']], c=[cols[c] for c in y.tolist()], s=50)
plt.show()

#convert to function for oregon coast car crash probability simulation
def oregon_coast_car_crash_prb(condition1, condition2):
    classifier = GaussianNB()
    X, y = car_crashdf[[str(condition1),str(condition2)]], car_crashdf['Coastal_YN']
    classifier.fit(X, y)
    X1, X2 = np.meshgrid(np.linspace(X[[str(condition1)]].min(), X[[str(condition1)]].max(),10), np.linspace(X[[str(condition2)]].min(), X[[str(condition2)]].max(),10))
    plt.figure(figsize=(10,6))
    # plot the probability surface for Oregon Coast
    plt.contourf(X1, X2, classifier.predict_proba(np.c_[X1.ravel(), X2.ravel()])[:,0].reshape(X1.shape), cmap='jet', alpha=.8)
    plt.colorbar()
    cols = {'Coastal':'black', 'Non-Coastal':'white'}
    plt.scatter(X[[str(condition1)]], X[[str(condition2)]], c=[cols[c] for c in y.tolist()], s=50)
    return plt.show()

#probability of an oregon coast car crash given dry and offroad conditions.
oregon_coast_car_crash_prb('DRY_COND', 'OFFROAD')

#probability of an oregon coast car crash given fatal and offroad conditions.
oregon_coast_car_crash_prb('FATAL', 'OFFROAD')


#create similar function for southern oregon region
def southern_oregon_car_crash_prb(condition1, condition2):
    classifier = GaussianNB()
    X, y = car_crashdf[[str(condition1),str(condition2)]], car_crashdf['Southern_Region_YN']
    classifier.fit(X, y)
    X1, X2 = np.meshgrid(np.linspace(X[[str(condition1)]].min(), X[[str(condition1)]].max(),10), np.linspace(X[[str(condition2)]].min(), X[[str(condition2)]].max(),10))
    plt.figure(figsize=(10,6))
    # plot the probability surface for Oregon Coast
    plt.contourf(X1, X2, classifier.predict_proba(np.c_[X1.ravel(), X2.ravel()])[:,0].reshape(X1.shape), cmap='jet', alpha=.8)
    plt.colorbar()
    cols = {'Southern' :'black', 'Non-Southern':'white'}
    plt.scatter(X[[str(condition1)]], X[[str(condition2)]], c=[cols[c] for c in y.tolist()], s=50)
    return plt.show()

#probability of a southern oregon car crash given dry and offroad conditions.
southern_oregon_car_crash_prb('DRY_COND', 'OFFROAD')

#probability of a southern oregon car crash given fatal and offroad conditions.
southern_oregon_car_crash_prb('FATAL', 'OFFROAD')