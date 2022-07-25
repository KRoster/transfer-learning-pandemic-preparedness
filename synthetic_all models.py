
## Transfer models - synthetic data


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
import os


from functions import data_prep, normalize_data, run_RF_transfer, run_transfer_models_lowdata_multiplediseases, run_model2




###############################################################################
### Data prep
###############################################################################



datanames = os.listdir('simulated_data/')
# remove our main dataset (which we use for transfer)
datanames = [d for d in datanames if 'beta' in d]
# separate the test sets
testnames = [d for d in datanames if 'test' in d]
# keep the remaining as train sets
datanames = [d for d in datanames if d not in testnames]


# read train data
data0 = pd.read_csv('simulated_data/source.csv')
data1 = pd.read_csv('simulated_data/'+datanames[0])
data2 = pd.read_csv('simulated_data/'+datanames[1])
data3 = pd.read_csv('simulated_data/'+datanames[2])
data4 = pd.read_csv('simulated_data/'+datanames[3])
data5 = pd.read_csv('simulated_data/'+datanames[4])
data6 = pd.read_csv('simulated_data/'+datanames[5])
data7 = pd.read_csv('simulated_data/'+datanames[6])
data8 = pd.read_csv('simulated_data/'+datanames[7])
data9 = pd.read_csv('simulated_data/'+datanames[8])

# change the feature names to match the context
for d in [data0,data1,data2,data3,data4,data5,data6,data7,data8,data9]:
    d.rename({'I_percap': 'cases', 'time':'date', 'id':'mun_code'}, axis=1, inplace=True)
    
# -------------------------------------------------------------------------------------------   
# read test data
test0 = pd.read_csv('simulated_data/source_test.csv')
test1 = pd.read_csv('simulated_data/'+testnames[0])
test2 = pd.read_csv('simulated_data/'+testnames[1])
test3 = pd.read_csv('simulated_data/'+testnames[2])
test4 = pd.read_csv('simulated_data/'+testnames[3]).
test5 = pd.read_csv('simulated_data/'+testnames[4])
test6 = pd.read_csv('simulated_data/'+testnames[5])
test7 = pd.read_csv('simulated_data/'+testnames[6])
test8 = pd.read_csv('simulated_data/'+testnames[7])
test9 = pd.read_csv('simulated_data/'+testnames[8])


data0_lags = data_prep(data0.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data1_lags = data_prep(data1.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data2_lags = data_prep(data2.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data3_lags = data_prep(data3.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data4_lags = data_prep(data4.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data5_lags = data_prep(data5.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data6_lags = data_prep(data6.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data7_lags = data_prep(data7.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data8_lags = data_prep(data8.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
data9_lags = data_prep(data9.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)


test0_lags = data_prep(test0.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test1_lags = data_prep(test1.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test2_lags = data_prep(test2.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test3_lags = data_prep(test3.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test4_lags = data_prep(test4.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test5_lags = data_prep(test5.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test6_lags = data_prep(test6.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test7_lags = data_prep(test7.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test8_lags = data_prep(test8.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)
test9_lags = data_prep(test9.loc[:,['date', 'mun_code', 'cases']], gap=0, max_lags=18)

# normalize
data0_norm = data0_lags.copy()

mean = np.mean(data0_norm.iloc[:,2:],axis=0)
std = np.std(data0_norm.iloc[:,2:],axis=0)
data0_norm.iloc[:,2:] = (data0_norm.iloc[:,2:] - mean)/std


test1_lags.name=testnames[0][12:-9]
test2_lags.name=testnames[1][12:-9]
test3_lags.name=testnames[2][12:-9]
test4_lags.name=testnames[3][12:-9]
test5_lags.name=testnames[4][12:-9]
test6_lags.name=testnames[5][12:-9]
test7_lags.name=testnames[6][12:-9]
test8_lags.name=testnames[7][12:-9]
test9_lags.name=testnames[8][12:-9]





###############################################################################
### Random Forest
###############################################################################


use_train = data0_norm

test_datasets = [test1_lags, test2_lags, test3_lags, test4_lags, test5_lags, 
                 test6_lags, test7_lags, test8_lags, test9_lags]

predictions_all = run_RF_transfer(use_train, train_mean=mean, train_std=std, 
                                  test_datasets=test_datasets, maxgap=9, lag=10)


# unnormalize
predictions_all['true_real'] = (predictions_all.true * std.cases) + mean.cases
predictions_all['pred_real'] = (predictions_all.pred * std.cases) + mean.cases






###############################################################################
### Random Forest Baseline
###############################################################################


train_dfs = [data1_lags, data2_lags, data3_lags, data4_lags, data5_lags, 
             data6_lags, data7_lags, data8_lags, data9_lags]
test_dfs = [test1_lags, test2_lags, test3_lags, test4_lags, test5_lags, 
                 test6_lags, test7_lags, test8_lags, test9_lags]

predictions_model2 = pd.DataFrame()
for i, use_train in enumerate(train_dfs):
    # select corresponding test df
    test_dataset = [test_dfs[i]]
    # normalize the train set
    train_norm = use_train.copy()
    train_mean = np.mean(train_norm.iloc[:,2:],axis=0)
    train_std = np.std(train_norm.iloc[:,2:],axis=0)
    train_norm.iloc[:,2:] = (train_norm.iloc[:,2:] - train_mean)/train_std
    # run the model
    predictions_temp = run_RF_transfer(use_train = train_norm, 
                                      train_mean = train_mean, 
                                      train_std = train_std, 
                                      test_datasets = test_dataset, 
                                      maxgap=9, lag=10)
    
    # unnormalize the predictions
    # unnormalize
    predictions_temp['true_real'] = (predictions_temp.true * train_std.cases) + train_mean.cases
    predictions_temp['pred_real'] = (predictions_temp.pred * train_std.cases) + train_mean.cases
    
    predictions_model2 = predictions_model2.append(predictions_temp)






###############################################################################
### NN Transfer Models
###############################################################################


# normalize

# test set
test1_norm = test1_lags.copy()
test1_norm.iloc[:,2:] = (test1_norm.iloc[:,2:] - mean)/std
# finetuning train set
data1_norm = data1_lags.copy()
data1_norm.iloc[:,2:] = (data1_norm.iloc[:,2:] - mean)/std

# test set
test2_norm = test2_lags.copy()
test2_norm.iloc[:,2:] = (test2_norm.iloc[:,2:] - mean)/std
# finetuning train set
data2_norm = data2_lags.copy()
data2_norm.iloc[:,2:] = (data2_norm.iloc[:,2:] - mean)/std

# test set
test3_norm = test3_lags.copy()
test3_norm.iloc[:,2:] = (test3_norm.iloc[:,2:] - mean)/std
# finetuning train set
data3_norm = data3_lags.copy()
data3_norm.iloc[:,2:] = (data3_norm.iloc[:,2:] - mean)/std

# test set
test4_norm = test4_lags.copy()
test4_norm.iloc[:,2:] = (test4_norm.iloc[:,2:] - mean)/std
# finetuning train set
data4_norm = data4_lags.copy()
data4_norm.iloc[:,2:] = (data4_norm.iloc[:,2:] - mean)/std

# test set
test5_norm = test5_lags.copy()
test5_norm.iloc[:,2:] = (test5_norm.iloc[:,2:] - mean)/std
# finetuning train set
data5_norm = data5_lags.copy()
data5_norm.iloc[:,2:] = (data5_norm.iloc[:,2:] - mean)/std

# test set
test6_norm = test6_lags.copy()
test6_norm.iloc[:,2:] = (test6_norm.iloc[:,2:] - mean)/std
# finetuning train set
data6_norm = data6_lags.copy()
data6_norm.iloc[:,2:] = (data6_norm.iloc[:,2:] - mean)/std

# test set
test7_norm = test7_lags.copy()
test7_norm.iloc[:,2:] = (test7_norm.iloc[:,2:] - mean)/std
# finetuning train set
data7_norm = data7_lags.copy()
data7_norm.iloc[:,2:] = (data7_norm.iloc[:,2:] - mean)/std

# test set
test8_norm = test8_lags.copy()
test8_norm.iloc[:,2:] = (test8_norm.iloc[:,2:] - mean)/std
# finetuning train set
data8_norm = data8_lags.copy()
data8_norm.iloc[:,2:] = (data8_norm.iloc[:,2:] - mean)/std

# test set
test9_norm = test9_lags.copy()
test9_norm.iloc[:,2:] = (test9_norm.iloc[:,2:] - mean)/std
# finetuning train set
data9_norm = data9_lags.copy()
data9_norm.iloc[:,2:] = (data9_norm.iloc[:,2:] - mean)/std



# add names
test1_norm.name=testnames[0][12:-9]
test2_norm.name=testnames[1][12:-9]
test3_norm.name=testnames[2][12:-9]
test4_norm.name=testnames[3][12:-9]
test5_norm.name=testnames[4][12:-9]
test6_norm.name=testnames[5][12:-9]
test7_norm.name=testnames[6][12:-9]
test8_norm.name=testnames[7][12:-9]
test9_norm.name=testnames[8][12:-9]


# compute predictions

train_dfs = [data1_norm, data2_norm, data3_norm, data4_norm, data5_norm, 
             data6_norm, data7_norm, data8_norm, data9_norm]
test_dfs = [test1_norm, test2_norm, test3_norm, test4_norm, test5_norm, 
                 test6_norm, test7_norm, test8_norm, test9_norm]



ckpt_name='train_on_data0'

cutoff_dates = [20, 25, 35, 100, 1000]


predictions_nn = run_transfer_models_lowdata_multiplediseases(use_train = data0_norm, 
                                                               test_datasets = test_dfs, 
                                                               train2_datasets = train_dfs, 
                                                               cutoff_dates = cutoff_dates, 
                                                               ckpt_name = ckpt_name, 
                                                               lag=9, maxgap=9
                                                              )

# unnormalize
predictions_nn['actual_real'] = (predictions_nn.actual * std['cases']) + mean['cases']
predictions_nn['pred1_real'] = (predictions_nn.pred1 * std['cases']) + mean['cases']
predictions_nn['pred2_real'] = (predictions_nn.pred2 * std['cases']) + mean['cases']
predictions_nn['pred3_real'] = (predictions_nn.pred3 * std['cases']) + mean['cases']




###############################################################################
### NN Baseline
###############################################################################


predictions_nn_model2 = pd.DataFrame()
test_identifiers = use_test.loc[:,['date','mun_code']].copy()

for i,use_test in enumerate(test_dfs):
    predictions_temp = run_model2(use_test=use_test.iloc[:,2:], use_train2=train_dfs[i].iloc[:,2:], 
                                  test_identifiers=test_identifiers,
                          ckpt_name='model2_'+test_dfs[i].name, 
                          lag=9, maxgap=9)
    predictions_temp['dataset'] = test_dfs[i].name
    
    predictions_nn_model2 = predictions_nn_model2.append(predictions_temp)


# unnormalize
predictions_nn_model2['actual_real'] = (predictions_nn_model2.actual * std['cases']) + mean['cases']
predictions_nn_model2['pred_model2_real'] = (predictions_nn_model2.pred_model2 * std['cases']) + mean['cases']

