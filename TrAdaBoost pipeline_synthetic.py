
## Transfer Learning using TrAdaBoost - synthetic data

import numpy as np
import pandas as pd
from adapt.instance_based import TrAdaBoostR2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



from functions import run_pipeline_tradaboost


### Transfer models using TrAdaBoost for synthetic data

# read and prep the data

datanames = os.listdir('simulated_data/horizons')
# remove our main dataset (which we use for transfer)
datanames = [d for d in datanames if 'beta' in d]
# separate the test sets
testnames = [d for d in datanames if 'test' in d]
# keep the remaining as train sets
datanames = [d for d in datanames if d not in testnames]


data0 = pd.read_csv('simulated_data/horizons/horizons_sird_waning-2021-11-14.csv')
test0 = pd.read_csv('simulated_data/horizons/horizons_sird_waning-2021-11-14_test.csv')

# read train data
data1 = pd.read_csv('simulated_data/horizons/'+datanames[0])
data2 = pd.read_csv('simulated_data/horizons/'+datanames[1])
data3 = pd.read_csv('simulated_data/horizons/'+datanames[2])
data4 = pd.read_csv('simulated_data/horizons/'+datanames[3])
data5 = pd.read_csv('simulated_data/horizons/'+datanames[4])
data6 = pd.read_csv('simulated_data/horizons/'+datanames[5])
data7 = pd.read_csv('simulated_data/horizons/'+datanames[6])
data8 = pd.read_csv('simulated_data/horizons/'+datanames[7])
data9 = pd.read_csv('simulated_data/horizons/'+datanames[8])


# read test data
test1 = pd.read_csv('simulated_data/horizons/'+testnames[0])
test2 = pd.read_csv('simulated_data/horizons/'+testnames[1])
test3 = pd.read_csv('simulated_data/horizons/'+testnames[2])
test4 = pd.read_csv('simulated_data/horizons/'+testnames[3])
test5 = pd.read_csv('simulated_data/horizons/'+testnames[4])
test6 = pd.read_csv('simulated_data/horizons/'+testnames[5])
test7 = pd.read_csv('simulated_data/horizons/'+testnames[6])
test8 = pd.read_csv('simulated_data/horizons/'+testnames[7])
test9 = pd.read_csv('simulated_data/horizons/'+testnames[8])


# normalize
data0_norm = data0.copy()

mean = np.mean(data0_norm.iloc[:,2:],axis=0)
std = np.std(data0_norm.iloc[:,2:],axis=0)
data0_norm.iloc[:,2:] = (data0_norm.iloc[:,2:] - mean)/std

# test set
test1_norm = test1.copy()
test1_norm.iloc[:,2:] = (test1_norm.iloc[:,2:] - mean)/std
# finetuning train set
data1_norm = data1.copy()
data1_norm.iloc[:,2:] = (data1_norm.iloc[:,2:] - mean)/std

# test set
test2_norm = test2.copy()
test2_norm.iloc[:,2:] = (test2_norm.iloc[:,2:] - mean)/std
# finetuning train set
data2_norm = data2.copy()
data2_norm.iloc[:,2:] = (data2_norm.iloc[:,2:] - mean)/std

# test set
test3_norm = test3.copy()
test3_norm.iloc[:,2:] = (test3_norm.iloc[:,2:] - mean)/std
# finetuning train set
data3_norm = data3.copy()
data3_norm.iloc[:,2:] = (data3_norm.iloc[:,2:] - mean)/std

# test set
test4_norm = test4.copy()
test4_norm.iloc[:,2:] = (test4_norm.iloc[:,2:] - mean)/std
# finetuning train set
data4_norm = data4.copy()
data4_norm.iloc[:,2:] = (data4_norm.iloc[:,2:] - mean)/std

# test set
test5_norm = test5.copy()
test5_norm.iloc[:,2:] = (test5_norm.iloc[:,2:] - mean)/std
# finetuning train set
data5_norm = data5.copy()
data5_norm.iloc[:,2:] = (data5_norm.iloc[:,2:] - mean)/std

# test set
test6_norm = test6.copy()
test6_norm.iloc[:,2:] = (test6_norm.iloc[:,2:] - mean)/std
# finetuning train set
data6_norm = data6.copy()
data6_norm.iloc[:,2:] = (data6_norm.iloc[:,2:] - mean)/std

# test set
test7_norm = test7.copy()
test7_norm.iloc[:,2:] = (test7_norm.iloc[:,2:] - mean)/std
# finetuning train set
data7_norm = data7.copy()
data7_norm.iloc[:,2:] = (data7_norm.iloc[:,2:] - mean)/std

# test set
test8_norm = test8.copy()
test8_norm.iloc[:,2:] = (test8_norm.iloc[:,2:] - mean)/std
# finetuning train set
data8_norm = data8.copy()
data8_norm.iloc[:,2:] = (data8_norm.iloc[:,2:] - mean)/std

# test set
test9_norm = test9.copy()
test9_norm.iloc[:,2:] = (test9_norm.iloc[:,2:] - mean)/std
# finetuning train set
data9_norm = data9.copy()
data9_norm.iloc[:,2:] = (data9_norm.iloc[:,2:] - mean)/std




train_dfs = [data1_norm, data2_norm, data3_norm, data4_norm, data5_norm, 
             data6_norm, data7_norm, data8_norm, data9_norm]
test_dfs = [test1_norm, test2_norm, test3_norm, test4_norm, test5_norm, 
                 test6_norm, test7_norm, test8_norm, test9_norm]

cutoff_dates = [20, 25, 35, 100, 1000]

predictions_synthetic = pd.DataFrame()
for i,target_train in enumerate(train_dfs):
    if i < 5:
        print(str(i)+' already completed: '+datanames[i])
        continue
        
    else:
        target_test = test_dfs[i]
        predictions_synthetic_1 = run_pipeline_tradaboost(source_data = data0_norm.drop(['date','mun_code'], axis=1), 
                                                  target_train = target_train.drop(['date','mun_code'], axis=1), 
                                                  target_test = target_test.drop(['date','mun_code'], axis=1), 
                                                  test_identifiers = target_test.loc[:,['date','mun_code']], 
                                                  target_train_identifiers = target_train.loc[:,['date','mun_code']],
                                                  cutoff_dates = cutoff_dates, 
                                                  lag=9, maxgap=9)
        predictions_synthetic_1['dataset'] = datanames[i]

        predictions_synthetic = predictions_synthetic.append(predictions_synthetic_1)



# unnormalize
predictions_synthetic['pred_real'] = (predictions_synthetic.pred * std['cases']) + mean['cases']
predictions_synthetic['actual_real'] = (predictions_synthetic.actual * std['cases']) + mean['cases']


# percentage predictions

### divide by the total number of cases per city
predictions_synthetic['total_cases'] = predictions_synthetic.groupby(['mun_code', 'gap', 'cutoff', 'dataset']).actual_real.transform('sum')

predictions_synthetic['actual_real_percent'] = predictions_synthetic.actual_real / predictions_synthetic.total_cases
predictions_synthetic['pred_real_percent'] = predictions_synthetic.pred_real / predictions_synthetic.total_cases

