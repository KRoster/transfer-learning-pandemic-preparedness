# RF transfer models

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from functions import normalize_data, prediction_horizons



###############################################################################
### Data Prep 
###############################################################################

# read data
dengue_train_horizons = pd.read_csv('dengue_train_lags_horizons.csv')
zika_test_horizons = pd.read_csv('zika_test_lags_horizons.csv')
zika_train_horizons = pd.read_csv('zika_train_lags_horizons.csv')


# randomly shuffle all the datasets
dengue_train_horizons = dengue_train_horizons.sample(frac=1, random_state=12)
zika_test_horizons = zika_test_horizons.sample(frac=1, random_state=12)
zika_train_horizons = zika_train_horizons.sample(frac=1, random_state=12)



###############################################################################
### Run Models 
###############################################################################


# train on dengue, test on zika
pred_horizon1 = prediction_horizons(use_train=dengue_train.copy(), use_test=zika_test.copy(), 
                                    maxgap=10, min_date_train='2014-01-01', min_date_test='2016-01-01',
                                    lag=8)


# train on zika, test on zika
pred_horizon2 = prediction_horizons(use_train=zika_train.copy(), use_test=zika_test.copy(), 
                                    maxgap=10, min_date_train='2016-01-01', min_date_test='2016-01-01',
                                    lag=8)




# train on influenza, test on covid
pred_horizon3 = prediction_horizons(use_train=flu.copy(), use_test=covid_test.copy(), 
                                    maxgap=10, min_date_train='2013-01-01', min_date_test='2020-03-28',
                                    lag=8)


# train on covid, test on covid
pred_horizon4 = prediction_horizons(use_train=covid_train.copy(), use_test=covid_test.copy(), 
                                    maxgap=10, min_date_train='2020-03-28', min_date_test='2020-03-28',
                                    lag=10)



## combine the predictions
pred_horizon1['train_on'] = 'dengue'
pred_horizon2['train_on'] = 'zika'

pred_horizon12 = pd.concat([pred_horizon1,pred_horizon2])

pred_horizon3['train_on'] = 'flu'
pred_horizon4['train_on'] = 'covid'

pred_horizon34 = pd.concat([pred_horizon3, pred_horizon4])

pred_horizon1234 = pd.concat([pred_horizon12,pred_horizon34])



### divide by the total number of cases per city
pred_horizon1234['total_cases'] = pred_horizon1234.groupby(['mun_code', 'train_on', 'gap']).true_real.transform('sum')

pred_horizon1234['true_real_percent'] = pred_horizon1234.true_real/pred_horizon1234.total_cases
pred_horizon1234['pred_real_percent'] = pred_horizon1234.pred_real/pred_horizon1234.total_cases


