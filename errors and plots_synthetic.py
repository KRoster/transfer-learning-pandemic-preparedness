
# # Error Plots for all Transfer Models - Synthetic data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


from functions import compute_errors_pair



###############################################################################
### Read and combine the data
###############################################################################


## RF models predictions
rf = pd.read_csv('simulated_results1/RF_pred_different simulated data-2021-11-16.csv', 
                 index_col='Unnamed: 0')


# rf model 2
rf_model2 = pd.read_csv('simulated_results1/RF_pred_model2_different simulated data-2021-11-16.csv',
                                 index_col='Unnamed: 0')

nn = pd.read_csv('simulated_results1/NN transfer_pred_different simulated data-2021-11-17.csv', 
                            index_col='Unnamed: 0')

nn_model2 = pd.read_csv('simulated_results1/NN model2_pred_different simulated data-2021-11-18.csv',
                                   index_col='Unnamed: 0')


# TrAdaBoost
predictions_datanames = os.listdir('simulated_results1')
# remove our main dataset (which we use for transfer)
predictions_datanames = [d for d in predictions_datanames if 'tradaboost_predictions' in d]

trada = pd.DataFrame()
for d in predictions_datanames:
    temp = pd.read_csv('simulated_results1/'+d)
    trada = trada.append(temp)


# for unnormalization:
data0 = pd.read_csv('simulated_data/horizons/horizons_sird_waning-2021-11-14.csv')
mean = np.mean(data0.iloc[:,2:],axis=0)
std = np.std(data0.iloc[:,2:],axis=0)

# unnormalize
nn['actual_real'] = (nn.actual * std['cases']) + mean['cases']
nn['pred1_real'] = (nn.pred1 * std['cases']) + mean['cases']
nn['pred2_real'] = (nn.pred2 * std['cases']) + mean['cases']
nn['pred3_real'] = (nn.pred3 * std['cases']) + mean['cases']

nn_model2['actual_real'] = (nn_model2.actual * std['cases']) + mean['cases']
nn_model2['pred_model2_real'] = (nn_model2.pred_model2 * std['cases']) + mean['cases']

trada['pred_real'] = (trada.pred * std['cases']) + mean['cases']
trada['actual_real'] = (trada.actual * std['cases']) + mean['cases']


# combine the predictions

# start with RF models
rf['model_type'] = 'RF'
rf['cutoff'] = 'no cutoff'
rf_model2['model_type'] = 'RF - model 2'
rf_model2['cutoff'] = 'no cutoff'
trada['model_type'] = 'TrAdaBoost'
rf.rename({'true':'actual', 'true_real':'actual_real'},axis=1, inplace=True)
rf_model2.rename({'true':'actual', 'true_real':'actual_real'},axis=1, inplace=True)

predictions = pd.DataFrame()
predictions = predictions.append(rf.drop(['actual', 'pred'], axis=1))
predictions = predictions.append(rf_model2.drop(['actual','pred'], axis=1))
predictions = predictions.append(trada.drop(['actual','pred'], axis=1))


# add the other datasets: nn 

# pred 1 (direct transfer) 
temp = nn.loc[:,['date', 'mun_code', 'gap','cutoff', 'dataset', 'actual_real', 'pred1_real']]
temp.rename({'pred1_real':'pred_real'}, axis=1, inplace=True)
temp['model_type'] = 'NN - no transfer'
predictions = predictions.append(temp)

# pred 2 (NN transfer) 
temp = nn.loc[:,['date', 'mun_code', 'gap','cutoff', 'dataset', 'actual_real', 'pred2_real']]
temp.rename({'pred2_real':'pred_real'}, axis=1, inplace=True)
temp['model_type'] = 'NN - transfer'
predictions = predictions.append(temp)

# pred 3 (finetuned transfer) 
temp = nn.loc[:,['date', 'mun_code', 'gap','cutoff', 'dataset', 'actual_real', 'pred3_real']]
temp.rename({'pred3_real':'pred_real'}, axis=1, inplace=True)
temp['model_type'] = 'NN - finetuned'
predictions = predictions.append(temp)


# nn model 2
temp = nn_model2.drop(['actual', 'pred_model2'], axis=1)
temp.rename({'pred_model2_real':'pred_real'}, axis=1, inplace=True)
temp['model_type'] = 'NN - model 2'
temp['cutoff'] = 'no cutoff'
predictions = predictions.append(temp)


## compute percent predictions (as with empirical data)
# divide by the total number of cases per city
predictions['total_cases'] = predictions.groupby(['mun_code', 'gap', 'cutoff', 'dataset', 'model_type']).actual_real.transform('sum')

predictions['actual_real_percent'] = predictions.actual_real / predictions.total_cases
predictions['pred_real_percent'] = predictions.pred_real / predictions.total_cases




###############################################################################
### Compute city errors
###############################################################################

errors = pd.DataFrame()
          
for mun_code in predictions.mun_code.unique():
    predictions_city = predictions[predictions.mun_code==mun_code]
    
    for modeltype in predictions_city.model_type.unique():
        predictions_modeltype = predictions_city[predictions_city.model_type==modeltype]

        for gap in predictions_modeltype.gap.unique():
            predictions_gap = predictions_modeltype[predictions_modeltype.gap==gap]

            for cutoff in predictions_gap.cutoff.unique():
                predictions_cutoff = predictions_gap[predictions_gap.cutoff==cutoff]

                for dataset in predictions_cutoff.dataset.unique():
                    predictions_dataset = predictions_cutoff[predictions_cutoff.dataset==dataset]

                    mae, rmse = compute_errors_pair(predictions_dataset.pred_real, predictions_dataset.actual_real)

                    # percent errors
                    if predictions_dataset.total_cases.unique()>0:
                        mae_percent, rmse_percent = compute_errors_pair(predictions_dataset.pred_real_percent, 
                                                                        predictions_dataset.actual_real_percent)
                    else:
                        mae_percent = np.NaN
                        rmse_percent = np.NaN

                    temp = pd.DataFrame({'mun_code' : mun_code,
                                         'gap' : gap,
                                         'cutoff' : cutoff,
                                         'dataset' : dataset,
                                         'model_type':modeltype,
                                         'mae' : mae,
                                         'rmse' : rmse,
                                         'mae_percent' : mae_percent,
                                         'rmse_percent' : rmse_percent,
                                         }, index=[0])

                    errors = errors.append(temp)



# correct dataset names
errors.dataset = [x[21:-4] if 'horizons' in x else x for x in errors.dataset]

# aggregate
errors_agg = errors.drop(['mun_code'], axis=1).groupby(['gap','cutoff', 'dataset','model_type'
                                                       ]).agg(['mean','median'])
errors_agg.reset_index(inplace=True)





###############################################################################
### Plots
###############################################################################



sns.set_style('whitegrid')
sns.set_context("paper", font_scale = 1.5)


diseases_ordered = ['beta0.25gamma0.01', 'beta0.25gamma0.1', 'beta0.25gamma0.15',
                    'beta0.3gamma0.01', 'beta0.3gamma0.1', 'beta0.3gamma0.15',
                    'beta0.35gamma0.01', 'beta0.35gamma0.1', 'beta0.35gamma0.15']

colors = ['#fab384', # light orange - TrAdaBoost
          '#b3b3b3', # light gray - NN baseline
          '#c95102', # darker orange - RF
          '#585a5c', # dark gray - RF baseline      
          '#093575', # dark blue - NN finetuned
          '#8bb7d7', # light blue - NN no transfer
          '#0068b3', # medium blue - NN transfer   
         ]

nicer_disease_names = [r'$\beta$'+'=0.25, '+ r'$\gamma$'+'=0.01',
                       r'$\beta$'+'=0.25, '+ r'$\gamma$'+'=0.1',
                       r'$\beta$'+'=0.25, '+ r'$\gamma$'+'=0.15',
                       r'$\beta$'+'=0.3, '+ r'$\gamma$'+'=0.01',
                       r'$\beta$'+'=0.3, '+ r'$\gamma$'+'=0.1',
                       r'$\beta$'+'=0.3, '+ r'$\gamma$'+'=0.15',
                       r'$\beta$'+'=0.35, '+ r'$\gamma$'+'=0.01',
                       r'$\beta$'+'=0.35, '+ r'$\gamma$'+'=0.1',
                       r'$\beta$'+'=0.35, '+ r'$\gamma$'+'=0.15']


fig, axs = plt.subplots(3,3, sharey=False, figsize=(15,15))

sub = errors_agg[errors_agg.cutoff.isin(['no cutoff', 20, '20'])]
# replace "model 2" with "baseline"
sub.replace({'RF - model 2': 'RF - baseline', 'NN - model 2': 'NN - baseline'},inplace=True)

for d,dataset in enumerate(diseases_ordered):
    temp = sub[sub.dataset==dataset]
    
    sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], 
                 hue=temp.model_type, style=temp.model_type, palette=colors, linewidth=3,
                ax=axs[int(d/3),d%3],
                )
    axs[int(d/3),d%3].set_title(nicer_disease_names[d])
    axs[int(d/3),d%3].set_xlabel('Prediction horizon')
    axs[int(d/3),d%3].set_ylabel('Median MAE (%)')
    axs[int(d/3),d%3].set_ylim([0, 0.0007])

    if d!=2:
        axs[int(d/3),d%3].get_legend().remove()

plt.tight_layout()




## rank plot

colors = ['#093575', # dark blue - NN finetuned
          '#8bb7d7', # light blue - NN no transfer
          '#0068b3', # medium blue - NN transfer
          '#fab384', # light orange - TrAdaBoost
          '#b3b3b3', # light gray - NN baseline
          '#c95102', # darker orange - RF
          '#585a5c', # dark gray - RF baseline
         ]

# select the relevant data (cutoff 1 & baselines)
sub = errors_agg[errors_agg.cutoff.isin(['no cutoff', 20, '20'])]
# replace "model 2" with "baseline"
sub.replace({'RF - model 2': 'RF - baseline', 'NN - model 2': 'NN - baseline'},inplace=True)


# set up plot
fig, axs = plt.subplots(3,3, figsize=(15,11), sharey=True, sharex=True)

for d,dataset in enumerate(diseases_ordered):
    # select the given dataset and gaps >=1
    temp = sub[(sub.dataset==dataset)&(sub.gap>0)]
    # adjust columns
    temp=temp.loc[:,[(  'gap',       ''),(  'model_type',       ''),( 'mae_percent', 'median')]]
    temp.columns = temp.columns.droplevel(level=1)
    # add rank
    temp['rank'] = temp.groupby('gap').agg({'mae_percent':'rank'})
    # adjust the gap value
    temp['gap'] = temp.gap+1
    # plot
    sns.lineplot(y='rank', x='gap', hue='model_type', data=temp, 
                 marker='o', linewidth=2, palette=colors, markersize=10,
                ax=axs[int(d/3),d%3], legend=False)
    axs[int(d/3),d%3].invert_yaxis()
    axs[int(d/3),d%3].set_title(nicer_disease_names[d])
    axs[int(d/3),d%3].set_xlabel('Prediction Horizon')
    axs[int(d/3),d%3].set_ylabel('Rank')

plt.tight_layout()




