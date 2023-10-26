# %%
#Datasets courtesy of FRED https://data.nasdaq.com/
#Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2013), "The Next Generation of the Penn World Table"
#available for download at www.ggdc.net/pwt For more information, see http://www.rug.nl/research/ggdc/data/penn-world-table.
# Multivariate time series forecasting of NZ - annual hh consumption, inflation, gdp change, price of imports
#vector autoregression model VAR - variables have a known influence on each other
#inflation consumer prices: FRED/FPCPITOTLZGNZL
#gdp nz: FRED/MKTGDPNZA646NWDB  --change in gdp stationary time series variable
#price level of household consumption NZ final household expenditure on goods and services: FRED/PLCCPPNZA670NRUG
#price level of imports: FRED/PLMCPPNZA670NRUG

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import matplotlib.pyplot as plt
import nasdaqdatalink as nsd



nz_hhcsmp = nsd.get("FRED/PLCCPPNZA670NRUG", returns="numpy", start_date="1974-01-01", end_date="2014-01-01")
nz_gdp = nsd.get("FRED/MKTGDPNZA646NWDB", returns="numpy", start_date="1974-01-01", end_date="2014-01-01")
nz_impts = nsd.get("FRED/PLMCPPNZA670NRUG", returns="numpy", start_date="1974-01-01", end_date="2014-01-01")
nz_inflt = nsd.get("FRED/FPCPITOTLZGNZL", returns="numpy", start_date="1974-01-01", end_date="2014-01-01")


#convert to pandas from recarray
nz_infltdf = pd.DataFrame(nz_inflt)
nz_imptsdf = pd.DataFrame(nz_impts) 
nz_gdpdf = pd.DataFrame(nz_gdp)
nz_hhcsmpdf = pd.DataFrame(nz_hhcsmp)

nz_dflist = [nz_infltdf, nz_imptsdf, nz_gdpdf, nz_hhcsmpdf]
nz_colnames = ['inflt', 'impts', 'gdp', 'hhcsmp']

#create stationary variable of difference in gdp and convert sci notation to int
nz_gdpdf['Value'] = (nz_gdpdf['Value']).astype('int64')
nz_gdpdf['gdp_delta'] = nz_gdpdf['Value'].diff

#concatenated df result where each country is grouped as a key
nz_df = pd.concat(nz_dflist, keys=nz_colnames)
nz_df.fillna(0)

#create transformed dataframe
nz_df_trnsfm = pd.DataFrame({ 'inflt':nz_df.loc["inflt","Value"]
                                ,'impts': nz_df.loc["impts","Value"]
                                , 'gdp_delta':nz_df.loc["gdp","Value"]
                                , 'hhcsmp': nz_df.loc["hhcsmp","Value"]})
 


# make a VAR model
import statsmodels.api as sm

from statsmodels.tsa.api import VAR
model = VAR(nz_df_trnsfm)
results = model.fit()
results.summary()

#impulse response analysis
irf = results.irf(10)
irf.plot(orth=False)

#Establish lag variable for more accurate fit.
#function to forecast lag simulation and model fit for inflation predicted vs true values
def forecast_lag_simulation(lag):
    try:
        model.select_order(lag)
    except ValueError:
        print('lag value is too large, try again')
    else:
        results = model.fit(maxlags=lag, ic='aic')
        lag_order = results.k_ar
        forecast_results = results.forecast(nz_df_trnsfm.values[-lag_order:], 5)
        forecast_results_df = pd.DataFrame(forecast_results, columns= nz_colnames)
        
        #plot predicted vs true: inflation
        nz_inflt_true = nsd.get("FRED/FPCPITOTLZGNZL", returns="numpy", start_date="2015-01-01", end_date="2019-01-01")
        nz_inflt_true_df = pd.DataFrame(nz_inflt_true)
        nz_inflt_true_values = nz_inflt_true_df['Value']
        nz_inflt_predicted_values = forecast_results_df['inflt']
        plt.figure(figsize=(10,10))
        plt.scatter(nz_inflt_true_values, nz_inflt_predicted_values, c='red')
        plt.xlim(max(nz_inflt_true_values),0)
        plt.ylim(0, max(nz_inflt_predicted_values))

        p1 = max(max(nz_inflt_predicted_values), max(nz_inflt_true_values))
        p2 = min(min(nz_inflt_predicted_values), min(nz_inflt_true_values))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.xlim(max(nz_inflt_true_values),0)
        plt.ylim(0, max(nz_inflt_predicted_values))
        
        return(results.summary(), results.plot_forecast(10), plt.show())

#experiment with the different lag value inputs
forecast_lag_simulation(8)
forecast_lag_simulation(7)
forecast_lag_simulation(5)
forecast_lag_simulation(3)


# %%
