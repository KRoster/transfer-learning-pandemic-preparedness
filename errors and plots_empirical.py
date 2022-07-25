
# # Error Plots for all Transfer Models - Empirical data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from functions import compute_errors_pair





###############################################################################
### Read and combine the data
###############################################################################


# read all the relevant data (predictions)

# NN model, trained on all data, includes all 3 transfer types and model 2
pred_nn = pd.read_csv('predictions_nn_incl percent predictions_2021-11-23.csv', 
                      index_col='Unnamed: 0', parse_dates=['date'])
# NN model, trained on cutoff data, all 3 transfer types
pred_nn_limited = pd.read_csv('predictions_nn_limited data_incl percent predictions_2021-11-23.csv', 
                              index_col='Unnamed: 0', parse_dates=['date'])
# rf predictions (direct transfer, no finetuning)
pred_rf = pd.read_csv('predictions_horizons_combined_2021-11-22.csv', parse_dates=['date'])
# rf predictions with tradaboost transfer
pred_trada_zika = pd.read_csv('empirical_results/tradaboost_predictions_zika_gaps and cutoffs_2021-11-29.csv',
                             index_col='Unnamed: 0', parse_dates=['date'])
pred_trada_covid = pd.read_csv('empirical_results/tradaboost_predictions_covid_gaps and cutoffs_2021-11-29.csv',
                              index_col='Unnamed: 0', parse_dates=['date'])


test_on_dict = {'zika':'zika',
                'dengue':'zika',
                'covid':'covid',
                'flu':'covid'
               }

pred1_dict = {'zika':'dengue', 'covid':'flu'}

pred_model2_dict = {'zika':'zika', 'covid':'covid'}

#### start with first dataset
# add the required variables
pred_rf['model_type'] = 'RF'
pred_rf['cutoff'] = 'no cutoff'
pred_rf['test_on'] = [test_on_dict[x] for x in pred_rf.train_on]

# add to the output dataframe
data = pred_rf.drop('total_cases',axis=1)
# add the method and cutoff columns
data.rename({'true_real':'actual_real', 'true_real_percent':'actual_real_percent'}, axis=1, inplace=True)


#### add the other datasets: tradaboost zika
pred_trada_zika['model_type'] = 'TrAdaBoost'
pred_trada_zika['train_on'] = 'dengue'
pred_trada_zika['test_on']='zika'
# append
data = data.append(pred_trada_zika.drop(['actual', 'pred', 'total_cases'], axis=1))


#### add the other datasets: tradaboost covid
pred_trada_covid['model_type'] = 'TrAdaBoost'
pred_trada_covid['train_on'] = 'flu'
pred_trada_covid['test_on']='covid'
# append
data = data.append(pred_trada_covid.drop(['actual', 'pred', 'total_cases'], axis=1))


#### add the other datasets: nn no transfer
pred_nn.rename({'dataset':'test_on'}, axis=1, inplace=True)
pred_nn['cutoff'] = 'no cutoff'

# pred 1 (direct transfer) 
temp = pred_nn.loc[:,['date', 'mun_code', 'gap', 'actual_real', 'pred1_real', 'test_on', 
                      'actual_real_percent', 'pred1_real_percent', 'cutoff']]
temp.rename({'pred1_real':'pred_real', 'pred1_real_percent': 'pred_real_percent'}, axis=1, inplace=True)
temp['train_on'] = [pred1_dict[x] for x in temp.test_on]
temp['model_type'] = 'NN - no transfer'
data = data.append(temp)

# pred 2 (transfer) 
temp = pred_nn.loc[:,['date', 'mun_code', 'gap', 'actual_real', 'pred2_real', 'test_on', 
                      'actual_real_percent', 'pred2_real_percent', 'cutoff']]
temp.rename({'pred2_real':'pred_real', 'pred2_real_percent': 'pred_real_percent'}, axis=1, inplace=True)
temp['train_on'] = [pred1_dict[x] for x in temp.test_on]
temp['model_type'] = 'NN - transfer'
data = data.append(temp)

# pred 3 (transfer & finetune) 
temp = pred_nn.loc[:,['date', 'mun_code', 'gap', 'actual_real', 'pred3_real', 'test_on', 
                      'actual_real_percent', 'pred3_real_percent', 'cutoff']]
temp.rename({'pred3_real':'pred_real', 'pred3_real_percent': 'pred_real_percent'}, axis=1, inplace=True)
temp['train_on'] = [pred1_dict[x] for x in temp.test_on]
temp['model_type'] = 'NN - finetuned'
data = data.append(temp)

# model 2 (train and test on same disease) 
temp = pred_nn.loc[:,['date', 'mun_code', 'gap', 'actual_real', 'pred_model2_real', 'test_on', 
                      'actual_real_percent', 'pred_model2_real_percent', 'cutoff']]
temp.rename({'pred_model2_real':'pred_real', 'pred_model2_real_percent': 'pred_real_percent'}, axis=1, inplace=True)
temp['train_on'] = [pred_model2_dict[x] for x in temp.test_on]
temp['model_type'] = 'NN - model 2'
data = data.append(temp)


#### NN - with cutoffs (limited finetuning data)

# pred 1 (direct transfer) 
temp = pred_nn_limited.loc[:,['date', 'mun_code', 'gap', 'actual_real', 'pred1_real', 'train_on', 
                      'actual_real_percent', 'pred1_real_percent', 'cutoff']]
temp.rename({'pred1_real':'pred_real', 'pred1_real_percent': 'pred_real_percent'}, axis=1, inplace=True)
temp['test_on'] = [test_on_dict[x] for x in temp.train_on]
temp['model_type'] = 'NN - no transfer'
data = data.append(temp)

# pred 2 (transfer) 
temp = pred_nn_limited.loc[:,['date', 'mun_code', 'gap', 'actual_real', 'pred2_real', 'train_on', 
                      'actual_real_percent', 'pred2_real_percent', 'cutoff']]
temp.rename({'pred2_real':'pred_real', 'pred2_real_percent': 'pred_real_percent'}, axis=1, inplace=True)
temp['test_on'] = [test_on_dict[x] for x in temp.train_on]
temp['model_type'] = 'NN - transfer'
data = data.append(temp)

# pred 3 (transfer & finetune) 
temp = pred_nn_limited.loc[:,['date', 'mun_code', 'gap', 'actual_real', 'pred3_real', 'train_on', 
                      'actual_real_percent', 'pred3_real_percent', 'cutoff']]
temp.rename({'pred3_real':'pred_real', 'pred3_real_percent': 'pred_real_percent'}, axis=1, inplace=True)
temp['test_on'] = [test_on_dict[x] for x in temp.train_on]
temp['model_type'] = 'NN - finetuned'
data = data.append(temp)




###############################################################################
### Compute city errors
###############################################################################


# group by: mun_code, gap, cutoff, model_type, train_on, test_on

errors = pd.DataFrame()
count=0          
for mun_code in data.mun_code.unique():
    predictions_city = data[data.mun_code==mun_code]
    
    
    for gap in predictions_city.gap.unique():
        predictions_gap = predictions_city[predictions_city.gap==gap]
        
        for cutoff in predictions_gap.cutoff.unique():
            predictions_cutoff = predictions_gap[predictions_gap.cutoff==cutoff]
            
            for model_type in predictions_cutoff.model_type.unique():
                predictions_modeltype = predictions_cutoff[predictions_cutoff.model_type==model_type]
                
                for trainedon in predictions_modeltype.train_on.unique():
                    predictions_trainedon = predictions_modeltype[predictions_modeltype.train_on==trainedon]
                    
                    for teston in predictions_trainedon.test_on.unique():
                        predictions_teston = predictions_trainedon[predictions_trainedon.test_on==teston]



                        mae, rmse = compute_errors_pair(predictions_teston.pred_real, 
                                                        predictions_teston.actual_real)
                        
                        # percent errors
                        if np.all(np.abs(predictions_teston.pred_real_percent.unique()) != np.inf):
                            mae_percent, rmse_percent = compute_errors_pair(predictions_teston.pred_real_percent, 
                                                                            predictions_teston.actual_real_percent)
                        else:
                            mae_percent = np.NaN
                            rmse_percent = np.NaN

                        temp = pd.DataFrame({'mun_code' : mun_code,
                                             'gap' : gap,
                                             'cutoff' : cutoff,
                                             'model_type':model_type,
                                             'train_on':trainedon,
                                             'test_on':teston,
                                             'mae' : mae,
                                             'rmse' : rmse,
                                             'mae_percent' : mae_percent,
                                             'rmse_percent' : rmse_percent,
                                             }, index=[0])

                        errors = errors.append(temp)

    count+=1
    print(count)



# aggregate
errors_agg = errors.drop(['mun_code'], axis=1).groupby(['gap','cutoff', 'model_type', 
                                                        'train_on', 'test_on']).agg(['mean','median'])


errors_agg.reset_index(inplace=True)





###############################################################################
### Plots
###############################################################################

sns.set_style('whitegrid')
sns.set_context("paper", font_scale = 1.5)

colors = ['#093575', # dark blue - NN finetuned
          '#8bb7d7', # light blue - NN no transfer
          '#0068b3', # medium blue - NN transfer
          '#fab384', # light orange - TrAdaBoost
          '#b3b3b3', # light gray - NN baseline
          '#c95102', # darker orange - RF
          '#585a5c', # dark gray - RF baseline
         ]


color_dict = {'NN - model 2':'#b3b3b3', 
              'NN - finetuned':'#093575', 
              'NN - transfer':'#0068b3',
              'NN - no transfer':'#8bb7d7', 
              'RF':'#c95102', 
              'RF - model 2':'#585a5c', 
              'TrAdaBoost':'#fab384'}





### Line plots

# Zika, RF models

sub = errors_agg[errors_agg.test_on=='zika']

sns.set_style('whitegrid')
sns.set_context("paper", font_scale = 1.5)

plt.figure(figsize=(6,4))

# RF models
temp = sub[sub.model_type=='RF']
temp = temp[temp.gap<9]
temp = temp.replace({'dengue':'RF - direct (trained on dengue)', 'zika': 'RF - baseline (trained on Zika)'})
sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], 
             linewidth=3,
             hue=temp.train_on, 
             style=temp.train_on, 
             dashes=False, 
             palette=['#c95102', # darker orange - RF, 
                      '#585a5c', # dark gray - RF baseline
                     ],
             legend=None
            )

# Tradaboost different cutoffs
temp = sub[sub.model_type=='TrAdaBoost']
temp = temp[temp.gap>0]
temp = temp.replace({'2016-05-27':'TrAdaBoost cutoff 1', 
                     '2016-06-24': 'TrAdaBoost cutoff 2',
                     '2016-07-22': 'TrAdaBoost cutoff 3',
                    })

sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], 
             style=temp.cutoff.astype(str), hue=temp.train_on,
             palette=['#fab384'], linewidth=3, legend=None
            )

plt.ylabel('MAE median (%)')
plt.xlabel('prediction horizon')



# Covid, NN models


sub = errors_agg[errors_agg.test_on=='covid']


sns.set_style('whitegrid')
sns.set_context("paper", font_scale = 1.5)

fig, axs = plt.subplots(1,2, sharey=True, figsize=(13,4))

# NN baseline
temp = sub[sub.model_type=='NN - model 2']
temp = temp.replace({'NN - model 2':'NN - baseline (trained on COVID-19)'})
sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], hue=temp.model_type, 
             palette=['#b3b3b3'], linewidth=3,
             #markers=True, 
             style=temp.model_type,
             legend=None, 
             ax=axs[0])
sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], hue=temp.model_type, 
             palette=['#b3b3b3'], linewidth=3,
             #markers=True, 
             style=temp.model_type,
             legend=None, 
             ax=axs[1])

# NN no transfer
temp = sub[sub.model_type=='NN - no transfer']
temp = temp[temp.cutoff=='no cutoff']
sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], hue=temp.model_type, 
             palette=['#8bb7d7'], linewidth=3,
             #markers='v', 
             style=temp.model_type,
             legend=None, 
             ax=axs[0])
sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], hue=temp.model_type, 
             palette=['#8bb7d7'], linewidth=3,
             #markers='v', 
             style=temp.model_type,
             legend=None, 
             ax=axs[1])

# NN transfer
temp = sub[sub.model_type=='NN - transfer']
temp=temp.replace({'2020-08-15':'cutoff 1',
                  '2020-09-12':'cutoff 2',
                  '2020-10-10':'cutoff 3',
                 })
temp = temp[temp.cutoff!='no cutoff']
sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], hue=temp.model_type, 
             palette=['#0068b3'], style=temp.cutoff.astype(str), linewidth=3,
             legend=None, 
             ax=axs[0])

# NN finetuned
temp = sub[sub.model_type=='NN - finetuned']
temp=temp.replace({'2020-08-15':'cutoff 1',
                  '2020-09-12':'cutoff 2',
                  '2020-10-10':'cutoff 3',
                 })
temp = temp[temp.cutoff!='no cutoff']
sns.lineplot(x=temp.gap+1, y=temp['mae_percent']['median'], hue=temp.model_type, 
             palette=['#093575'], style=temp.cutoff.astype(str), linewidth=3,
             legend=None, 
             ax=axs[1])


axs[0].set_xlabel('prediction horizon')
axs[0].set_ylabel('median MAE (%)')

axs[0].set_title('NN - Transfer')
axs[1].set_title('NN - Fine-tuned')

plt.xlabel('prediction horizon')



# ---------



## Rank plot, zika

## rank plot
sub = errors_agg[errors_agg.test_on=='zika']
sub = sub[sub.gap>0]
sub = sub[sub.gap<9]

# rename to differentiate between "RF - direct" and "RF - baseline"
sub['model_type'] = np.where((sub.model_type=='RF')&(sub.train_on=='zika'), 
                             'RF - baseline (trained on Zika)', 
                             sub.model_type)
sub['model_type'] = np.where((sub.model_type=='RF')&(sub.train_on=='dengue'), 
                             'RF - direct (trained on dengue)', 
                             sub.model_type)
sub['model_type'] = np.where((sub.model_type=='NN - model 2'), 
                             'NN - baseline (trained on Zika)', 
                             sub.model_type)
sub['model_type'] = np.where((sub.model_type=='NN - no transfer'), 
                             'NN - no transfer (trained on dengue)', 
                             sub.model_type)

# select the relevant cutoffs
sub = sub[sub.cutoff.isin(['no cutoff', '2016-05-27'])]

# drop the transfer models with no cutoff
sub.drop(sub[(sub.cutoff=='no cutoff')&(sub.model_type.isin(['NN - finetuned', 'NN - no transfer (trained on dengue)',
                                                    'NN - transfer'
                                                   ]))].index, inplace=True)

# select columns and adjust column names
sub = sub.loc[:,[(  'gap',       ''),(  'model_type',       ''),( 'mae_percent', 'median')]]
sub.columns = sub.columns.droplevel(level=1)


# add a rank 
sub['rank'] = sub.groupby('gap').agg({'mae_percent':'rank'})

# adjust the gap value
sub['gap'] = sub.gap+1


# plot
fig, axs = plt.subplots(1,1, figsize=(6,4))
    
sns.lineplot(y='rank', x='gap', hue='model_type', data=sub, 
             marker='o', linewidth=2, palette=colors, markersize=10,
            ax=axs, #legend=False
            )
lgd=plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs.invert_yaxis()
#axs.set_title(nicer_disease_names[d])
axs.set_xlabel('Prediction horizon')
axs.set_ylabel('Rank')



## Rank plot, covid

sub = errors_agg[errors_agg.test_on=='covid']
sub = sub[sub.gap>0]
sub = sub[sub.gap<9]

# rename to differentiate between "RF - direct" and "RF - baseline"
sub['model_type'] = np.where((sub.model_type=='RF')&(sub.train_on=='covid'), 
                             'RF - baseline (trained on COVID-19)', 
                             sub.model_type)
sub['model_type'] = np.where((sub.model_type=='RF')&(sub.train_on=='flu'), 
                             'RF - direct (trained on influenza)', 
                             sub.model_type)
sub['model_type'] = np.where((sub.model_type=='NN - model 2'), 
                             'NN - baseline (trained on COVID-19)', 
                             sub.model_type)
sub['model_type'] = np.where((sub.model_type=='NN - no transfer'), 
                             'NN - no transfer (trained on influenza)', 
                             sub.model_type)

# select the relevant cutoffs
sub = sub[sub.cutoff.isin(['no cutoff', '2020-08-15'])]

# drop the transfer models with no cutoff
sub.drop(sub[(sub.cutoff=='no cutoff')&(sub.model_type.isin(['NN - finetuned', 'NN - no transfer (trained on influenza)',
                                                    'NN - transfer'
                                                   ]))].index, inplace=True)

# select columns and adjust column names
sub = sub.loc[:,[(  'gap',       ''),(  'model_type',       ''),( 'mae_percent', 'median')]]
sub.columns = sub.columns.droplevel(level=1)


# add a rank 
sub['rank'] = sub.groupby('gap').agg({'mae_percent':'rank'})

# adjust the gap value
sub['gap'] = sub.gap+1



# plot
fig, axs = plt.subplots(1,1, figsize=(6,4))
    
sns.lineplot(y='rank', x='gap', hue='model_type', data=sub, 
             marker='o', linewidth=2, palette=colors, markersize=10,
            ax=axs, #legend=False
            )
lgd=plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs.invert_yaxis()
#axs.set_title(nicer_disease_names[d])
axs.set_xlabel('Prediction horizon')
axs.set_ylabel('Rank')



# ---------




#### select best model by city

errors.reset_index(drop=True,inplace=True)

# remove na and zero errors
temp = errors[errors['mae_percent'].notna()]
temp = temp[temp.mae_percent!=0]
# remove the no cutoff & no transfer NN rows
temp = temp.drop(temp[(temp['model_type'] == 'NN - no transfer') & (temp['cutoff'] == 'no cutoff')].index)
temp.drop(temp[(temp.cutoff=='no cutoff')&(temp.model_type.isin(['NN - finetuned', 'NN - no transfer',
                                                    'NN - transfer'
                                                   ]))].index, inplace=True)

# change the name of the RF model to RF-model2 where appropriate
temp.loc[(temp.train_on=='zika')&(temp.test_on=='zika')&(temp.model_type=='RF'), 'model_type'] = 'RF - model 2'
temp.loc[(temp.train_on=='covid')&(temp.test_on=='covid')&(temp.model_type=='RF'), 'model_type'] = 'RF - model 2'

temp.sort_values('mae_percent', inplace=True)
temp.reset_index(drop=True,inplace=True)



# separate best model for each cutoff

# without baselines
temp2 = temp[temp.model_type.isin(['TrAdaBoost', 'NN - transfer', 'RF',
                                   'NN - finetuned', 'NN - no transfer'])]

best_model_ids_cutoff1 = temp2[temp2.cutoff.isin(['2016-05-27','2020-08-15', 'no cutoff'])].groupby(['mun_code', 'test_on', 'gap'])['mae_percent'].idxmin()
best_model_ids_cutoff2 = temp2[temp2.cutoff.isin(['2016-06-24','2020-09-12', 'no cutoff'])].groupby(['mun_code', 'test_on', 'gap'])['mae_percent'].idxmin()
best_model_ids_cutoff3 = temp2[temp2.cutoff.isin(['2016-07-22','2020-10-10', 'no cutoff'])].groupby(['mun_code', 'test_on', 'gap'])['mae_percent'].idxmin()

best_model_cutoff1_nobaseline = temp2.loc[best_model_ids_cutoff1]
best_model_cutoff2_nobaseline = temp2.loc[best_model_ids_cutoff2]
best_model_cutoff3_nobaseline = temp2.loc[best_model_ids_cutoff3]

# combine the datasets
  
best_model_allcutoffs_nobaseline = pd.DataFrame()
for b, best in enumerate([best_model_cutoff1_nobaseline, best_model_cutoff2_nobaseline, 
                          best_model_cutoff3_nobaseline]):
    fool = best.copy()
    fool['cutoff_level']='cutoff '+str(b+1)
    best_model_allcutoffs_nobaseline = best_model_allcutoffs_nobaseline.append(fool)




### stacked barplot (WITHOUT baseline models)

for i,dataset in enumerate(['zika', 'covid']):


    sub_best_model = best_model_allcutoffs_nobaseline[(best_model_allcutoffs_nobaseline.test_on==dataset)]

    # adjust the gap value
    sub_best_model.gap = sub_best_model.gap+1

    for g,use_gap in enumerate(np.arange(2,10)):
        # drop the rows where both 'no cutoff' and 'cutoff' exist

        fig, axs = plt.subplots(ncols=1,nrows=1, figsize=(6,4))
        
        # select only a single gap value
        use_data = sub_best_model[sub_best_model.gap==use_gap]

        # reshape
        best_model_pivot = use_data.drop(['mun_code', 'mae', 'rmse', 'rmse_percent', 'cutoff', 'train_on','test_on', 
                                                'gap'],axis=1).groupby(['cutoff_level', 'model_type']).agg('count')
        best_model_pivot = best_model_pivot.unstack(level=-1)
        best_model_pivot.columns = best_model_pivot.columns.droplevel()

        # get the list of required colors
        colors = [color_dict[x] for x in best_model_pivot.columns]
        # plot
        best_model_pivot.plot.bar(stacked=True, color=colors, ax=axs, legend=False)
        lgd=plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs.set_xlabel('Level of data availability')
        axs.set_ylabel('Frequency')
        axs.set_xticklabels(['cutoff 1', 'cutoff 2', 'cutoff 3'], rotation=0)

        plt.tight_layout()

    plt.show()



