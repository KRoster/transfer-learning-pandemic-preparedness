
## Transfer Learning using TrAdaBoost - empirical data

import numpy as np
import pandas as pd
from adapt.instance_based import TrAdaBoostR2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



from functions import run_pipeline_tradaboost



# ## Zika & Dengue


# read data
dengue_train_horizons = pd.read_csv('dengue_train_lags_horizons.csv')
zika_test_horizons = pd.read_csv('zika_test_lags_horizons.csv')
zika_train_horizons = pd.read_csv('zika_train_lags_horizons.csv')



# randomly shuffle all the datasets
dengue_train_horizons = dengue_train_horizons.sample(frac=1, random_state=12)
zika_test_horizons = zika_test_horizons.sample(frac=1, random_state=12)
zika_train_horizons = zika_train_horizons.sample(frac=1, random_state=12)




# normalize
dengue_train_horizons_norm, zika_test_horizons_norm, mean_horizons, std_horizons  = normalize_data(dengue_train_horizons.copy(), zika_test_horizons.copy())

zika_train_horizons_norm = zika_train_horizons.copy()
zika_train_horizons_norm.iloc[:,2:] = (zika_train_horizons_norm.iloc[:,2:] - mean_horizons)/std_horizons



# run the tradaboost for zika/dengue

cutoff_dates = ['2016-05-27', '2016-06-24', '2016-07-22']

predictions = run_pipeline_tradaboost(source_data = dengue_train_horizons_norm.copy(), 
                                      target_train = zika_train_horizons_norm.copy(), 
                                      target_test = zika_test_horizons_norm.copy(), 
                                      test_identifiers = zika_test_horizons.loc[:,['date','mun_code']], 
                                      target_train_identifiers = zika_train_horizons.loc[:,['date','mun_code']],
                                      cutoff_dates = cutoff_dates, 
                                      lag=9, maxgap=9)


# unnormalize
predictions['pred_real'] = (predictions.pred * std_horizons['cases']) + mean_horizons['cases']
predictions['actual_real'] = (predictions.actual * std_horizons['cases']) + mean_horizons['cases']



# percentage predictions

### divide by the total number of cases per city
predictions['total_cases'] = predictions.groupby(['mun_code', 'gap', 'cutoff']).actual_real.transform('sum')

predictions['actual_real_percent'] = predictions.actual_real / predictions.total_cases
predictions['pred_real_percent'] = predictions.pred_real / predictions.total_cases



# compute city errors
# errors
### compute errors across cities
errors = pd.DataFrame()
          
for mun_code in predictions.mun_code.unique():
    predictions_city = predictions[predictions.mun_code==mun_code]
    
    for gap in predictions_city.gap.unique():
        predictions_gap = predictions_city[predictions_city.gap==gap]
        
        for cutoff in predictions_gap.cutoff.unique():
            predictions_cutoff = predictions_gap[predictions_gap.cutoff==cutoff]

            mae, rmse = compute_errors_pair(predictions_cutoff.pred_real, predictions_cutoff.actual_real)

            # percent errors
            if predictions_cutoff.total_cases.unique()>0:
                mae_percent, rmse_percent = compute_errors_pair(predictions_cutoff.pred_real_percent, 
                                                                predictions_cutoff.actual_real_percent)
            else:
                mae_percent = np.NaN
                rmse_percent = np.NaN
            
            temp = pd.DataFrame({'mun_code' : mun_code,
                                 'gap' : gap,
                                 'cutoff' : cutoff,
                                 'mae' : mae,
                                 'rmse' : rmse,
                                 'mae_percent' : mae_percent,
                                 'rmse_percent' : rmse_percent,
                                 }, index=[0])

            errors = errors.append(temp)



# aggregate errors
errors_agg = errors.drop(['mun_code'], axis=1).groupby(['gap','cutoff']).agg(['mean','median'])



# ## Flu & Covid


# read the data
flu_train_horizons = pd.read_csv('flu_train_lags_horizons.csv', parse_dates=['date'])
covid_test_horizons = pd.read_csv('covid_test_lags_horizons.csv', parse_dates=['date'])
covid_train_horizons = pd.read_csv('covid_train_lags_horizons.csv', parse_dates=['date'])


# randomly shuffle all the datasets
flu_train_horizons = flu_train_horizons.sample(frac=1, random_state=12)
covid_test_horizons = covid_test_horizons.sample(frac=1, random_state=12)
covid_train_horizons = covid_train_horizons.sample(frac=1, random_state=12)


# normalize
flu_train_horizons_norm, covid_test_horizons_norm, mean_horizons_covid, std_horizons_covid  = normalize_data(flu_train_horizons.copy(), covid_test_horizons.copy())

covid_train_horizons_norm = covid_train_horizons.copy()
covid_train_horizons_norm.iloc[:,2:] = (covid_train_horizons_norm.iloc[:,2:] - mean_horizons_covid)/std_horizons_covid


# run the tradaboost for covid/flu
cutoff_dates = ['2020-08-15', '2020-09-12', '2020-10-10']

predictions_covid = run_pipeline_tradaboost(source_data = flu_train_horizons_norm.copy(), 
                                          target_train = covid_train_horizons_norm.copy(), 
                                          target_test = covid_test_horizons_norm.copy(), 
                                          test_identifiers = covid_test_horizons.loc[:,['date','mun_code']], 
                                          target_train_identifiers = covid_train_horizons.loc[:,['date','mun_code']],
                                          cutoff_dates = cutoff_dates, 
                                          lag=9, maxgap=9)


# unnormalize
predictions_covid['pred_real'] = (predictions_covid.pred * std_horizons_covid['cases']) + mean_horizons_covid['cases']
predictions_covid['actual_real'] = (predictions_covid.actual * std_horizons_covid['cases']) + mean_horizons_covid['cases']


# percentage predictions

### divide by the total number of cases per city
predictions_covid['total_cases'] = predictions_covid.groupby(['mun_code', 'gap', 'cutoff']).actual_real.transform('sum')

predictions_covid['actual_real_percent'] = predictions_covid.actual_real / predictions_covid.total_cases
predictions_covid['pred_real_percent'] = predictions_covid.pred_real / predictions_covid.total_cases


