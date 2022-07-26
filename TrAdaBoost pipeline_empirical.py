
## Transfer Learning using TrAdaBoost - empirical data

import numpy as np
import pandas as pd
from adapt.instance_based import TrAdaBoostR2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



from functions import run_pipeline_tradaboost




# read data
dengue_train = pd.read_csv('dengue_train.csv')
zika_test = pd.read_csv('zika_test.csv')
zika_train = pd.read_csv('zika_train.csv')

flu_train = pd.read_csv('flu_train.csv')
covid_test = pd.read_csv('covid_test.csv')
covid_train = pd.read_csv('covid_train.csv')


# randomly shuffle all the datasets
dengue_train = dengue_train.sample(frac=1, random_state=12)
zika_test = zika_test.sample(frac=1, random_state=12)
zika_train = zika_train.sample(frac=1, random_state=12)

flu_train = flu_train.sample(frac=1, random_state=12)
covid_test = covid_test.sample(frac=1, random_state=12)
covid_train = covid_train.sample(frac=1, random_state=12)



# normalize
dengue_train_norm, zika_test_norm, mean_val, std_dev  = normalize_data(dengue_train.copy(), zika_test.copy())
zika_train_norm = zika_train.copy()
zika_train_norm.iloc[:,2:] = (zika_train_norm.iloc[:,2:] - mean_val)/std_dev


flu_train_norm, covid_test_norm, mean_covid, std_covid  = normalize_data(flu_train.copy(), covid_test.copy())
covid_train_norm = covid_train_horizons.copy()
covid_train_norm.iloc[:,2:] = (covid_train_norm.iloc[:,2:] - mean_covid)/std_covid


# run the tradaboost for zika/dengue

cutoff_dates = ['2016-05-27', '2016-06-24', '2016-07-22']

predictions = run_pipeline_tradaboost(source_data = dengue_train_norm.copy(), 
                                      target_train = zika_train_norm.copy(), 
                                      target_test = zika_test_norm.copy(), 
                                      test_identifiers = zika_test.loc[:,['date','mun_code']], 
                                      target_train_identifiers = zika_train.loc[:,['date','mun_code']],
                                      cutoff_dates = cutoff_dates, 
                                      lag=9, maxgap=9)


# unnormalize
predictions['pred_real'] = (predictions.pred * std_dev['cases']) + mean_val['cases']
predictions['actual_real'] = (predictions.actual * std_dev['cases']) + mean_val['cases']



# percentage predictions

### divide by the total number of cases per city
predictions['total_cases'] = predictions.groupby(['mun_code', 'gap', 'cutoff']).actual_real.transform('sum')

predictions['actual_real_percent'] = predictions.actual_real / predictions.total_cases
predictions['pred_real_percent'] = predictions.pred_real / predictions.total_cases




# run the tradaboost for covid/flu
cutoff_dates = ['2020-08-15', '2020-09-12', '2020-10-10']

predictions_covid = run_pipeline_tradaboost(source_data = flu_train_norm.copy(), 
                                          target_train = covid_train_norm.copy(), 
                                          target_test = covid_test_norm.copy(), 
                                          test_identifiers = covid_test.loc[:,['date','mun_code']], 
                                          target_train_identifiers = covid_train.loc[:,['date','mun_code']],
                                          cutoff_dates = cutoff_dates, 
                                          lag=9, maxgap=9)


# unnormalize
predictions_covid['pred_real'] = (predictions_covid.pred * std_covid['cases']) + mean_covid['cases']
predictions_covid['actual_real'] = (predictions_covid.actual * std_covid['cases']) + mean_covid['cases']


# percentage predictions

### divide by the total number of cases per city
predictions_covid['total_cases'] = predictions_covid.groupby(['mun_code', 'gap', 'cutoff']).actual_real.transform('sum')

predictions_covid['actual_real_percent'] = predictions_covid.actual_real / predictions_covid.total_cases
predictions_covid['pred_real_percent'] = predictions_covid.pred_real / predictions_covid.total_cases


