
### NN transfer models


import numpy as np
import pandas as pd


from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error


from functions import normalize_data, run_transfer_models_lowdata, run_model2



###############################################################################
### Prep data (Zika & Dengue)
###############################################################################


# read data
dengue_train_horizons = pd.read_csv('dengue_train_lags_horizons.csv')
zika_test_horizons = pd.read_csv('zika_test_lags_horizons.csv')
zika_train_horizons = pd.read_csv('zika_train_lags_horizons.csv')


# normalize
dengue_train_horizons_norm, zika_test_horizons_norm, mean_horizons, std_horizons  = normalize_data(dengue_train_horizons.copy(), zika_test_horizons.copy())
zika_train_horizons_norm, zika_test_horizons_norm2, mean_horizons2, std_horizons2  = normalize_data(zika_train_horizons.copy(), zika_test_horizons.copy())


test_identifiers = zika_test_horizons.loc[:,['date','mun_code']].copy()
train2_identifiers = zika_train_horizons.loc[:,['date','mun_code']].copy()
ckpt_name='train_on_dengue_limiteddata'

cutoff_dates = ['2016-05-27', '2016-06-24', '2016-07-22']






###############################################################################
### Run models
###############################################################################

predictions_out = run_transfer_models_lowdata(use_train = dengue_train_horizons_norm, 
                                              use_test = zika_test_horizons_norm, 
                                              use_train2 = zika_train_horizons_norm, 
                                              test_identifiers = test_identifiers, 
                                              train2_identifiers = train2_identifiers,
                                              cutoff_dates = cutoff_dates, 
                                              ckpt_name = ckpt_name, 
                                              lag=9, 
                                              maxgap=9)
   


# unnormalize the predictions
predictions_out['actual_real'] = (predictions_out.actual * std_horizons['cases']) + mean_horizons['cases']
predictions_out['pred1_real'] = (predictions_out.pred1 * std_horizons['cases']) + mean_horizons['cases']
predictions_out['pred2_real'] = (predictions_out.pred2 * std_horizons['cases']) + mean_horizons['cases']
predictions_out['pred3_real'] = (predictions_out.pred3 * std_horizons['cases']) + mean_horizons['cases']




###############################################################################
### repeat for Flu & COVID-19
###############################################################################


# read data
flu_train_horizons = pd.read_csv('flu_train_lags_horizons.csv', parse_dates=['date'])
covid_test_horizons = pd.read_csv('covid_test_lags_horizons.csv', parse_dates=['date'])
covid_train_horizons = pd.read_csv('covid_train_lags_horizons.csv', parse_dates=['date'])

# normalize
flu_train_horizons_norm, covid_test_horizons_norm, mean_horizons_covid, std_horizons_covid  = normalize_data(flu_train_horizons.copy(), covid_test_horizons.copy())
covid_train_horizons_norm, covid_test_horizons_norm2, mean_horizons2_covid, std_horizons2_covid  = normalize_data(covid_train_horizons.copy(), covid_test_horizons.copy())


test_identifiers = covid_test_horizons.loc[:,['date','mun_code']].copy()
train2_identifiers = covid_train_horizons.loc[:,['date','mun_code']].copy()
ckpt_name='train_on_flu_limiteddata'

cutoff_dates = ['2020-08-15', '2020-09-12', '2020-10-10']



predictions_out_covid = run_transfer_models_lowdata(use_train = flu_train_horizons_norm, 
                                              use_test = covid_test_horizons_norm, 
                                              use_train2 = covid_train_horizons_norm, 
                                              test_identifiers = test_identifiers, 
                                              train2_identifiers = train2_identifiers,
                                              cutoff_dates = cutoff_dates, 
                                              ckpt_name = ckpt_name, 
                                              lag=9, 
                                              maxgap=9)


# unnormalize the predictions
predictions_out_covid['actual_real'] = (predictions_out_covid.actual * std_horizons_covid['cases']) + mean_horizons_covid['cases']
predictions_out_covid['pred1_real'] = (predictions_out_covid.pred1 * std_horizons_covid['cases']) + mean_horizons_covid['cases']
predictions_out_covid['pred2_real'] = (predictions_out_covid.pred2 * std_horizons_covid['cases']) + mean_horizons_covid['cases']
predictions_out_covid['pred3_real'] = (predictions_out_covid.pred3 * std_horizons_covid['cases']) + mean_horizons_covid['cases']




# ## Relative errors (percentage of total number of cases in the city)


# combine the predictions
predictions_out_covid['train_on']='flu'
predictions_out['train_on']='dengue'

pred_combo = pd.concat([predictions_out_covid, predictions_out])




### divide by the total number of cases per city
pred_combo['total_cases'] = pred_combo.groupby(['mun_code', 'train_on', 'gap', 'cutoff']).actual_real.transform('sum')

pred_combo['actual_real_percent'] = pred_combo.actual_real / pred_combo.total_cases
pred_combo['pred1_real_percent'] = pred_combo.pred1_real / pred_combo.total_cases
pred_combo['pred2_real_percent'] = pred_combo.pred2_real / pred_combo.total_cases
pred_combo['pred3_real_percent'] = pred_combo.pred3_real / pred_combo.total_cases







###############################################################################
### NN baselines
###############################################################################

### Zika & Dengue

use_test = zika_test_horizons_norm.copy()
use_train2 = zika_train_horizons_norm.copy()

ckpt_name='train_on_zika'

predictions_allgaps_zika_model2 = run_model2(use_test, use_train2, test_identifiers, ckpt_name, lag=9, maxgap=9)




### COVID-19 & Flu

use_test = covid_test_horizons_norm.copy()
use_train2 = covid_train_horizons_norm.copy()

ckpt_name='train_on_covid'

predictions_allgaps_covid_model2 = run_model2(use_test, use_train2, test_identifiers, ckpt_name, lag=9, maxgap=9)






