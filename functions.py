


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import binom, nbinom
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping



# --- data prep ---
def normalize_data(data_train, data_test):
    # remove identifiers from data
    data_train_norm = data_train.drop(['mun_code','date'], axis=1).copy()
    data_test_norm = data_test.drop(['mun_code','date'], axis=1).copy()

    # normalize
    mean = np.mean(data_train_norm,axis=0)
    std = np.std(data_train_norm,axis=0)
    data_train_norm = (data_train_norm - mean)/std
    data_test_norm = (data_test_norm - mean)/std
    
    return data_train_norm, data_test_norm, mean, std


def data_prep(data, gap = 0, max_lags=12):
    """add lagged features as columns
    """
    
    # remove rows with NA date
    data.dropna(axis=0, subset=['date'], inplace=True)
    # set identifiers
    data.set_index(["date", "mun_code"], inplace=True)

    # prep output with unlagged data:
    data_lags = data.copy()

    # add all of the lags
    for l in np.arange(gap+1,max_lags):
        data_lags[data.columns+'_lag'+str(l)] = data.copy().unstack().shift(l).stack(dropna=False)
    # reset index    
    data_lags.reset_index(inplace=True)
    data_lags.sort_values("date", inplace=True)

    # remove rows with missing values
    data_lags.dropna(inplace=True)
    

    return data_lags


# --- errors ----
def compute_errors_pair(y_pred, y_true):
    
    """compute MAE and RMSE (normalized and real) 
    """
    
    # mae
    mae =  mean_absolute_error(y_true, y_pred)
    # rmse
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    
    return mae,rmse



# --- synthetic data generation ---

def simulate(T, S0, I0, R0, params, fct):
    
    S = S0
    I = I0
    R = R0

    S_list = [S]
    I_list = [I]
    R_list = [R]
    
    for t in np.arange(T):
        N = S+I+R
        S, I, R = fct(S,I,R,N,params,t)
        
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
    
    return S_list, I_list, R_list



# density dependent transmission rate
def sird_waning_immunity_step(S, I, R, N, params, t):
    
    beta, gamma, zeta, mu_i = params
    
    # stochastic events
    # infection
    dn_si = binom.rvs(n=S, p=1-np.exp(-beta*I/N))
    # recovery
    dn_ir = binom.rvs(n=I, p=1-np.exp(-gamma))
    # waning immunity
    dn_rs = binom.rvs(n=R, p=1-np.exp(-zeta))
    # death from disease + regular death rate
    i_death = binom.rvs(n=I, p=1-np.exp(-mu_i))
    
    # updates
    S += - dn_si + dn_rs 
    I += dn_si - dn_ir - i_death
    R += dn_ir - dn_rs 
    
    if S<0:
        S=0
    if I<0:
        I=0
    if R<0:
        R=0
    
    return S, I, R



# frequency dependent transmission rate
def generate_data_fixedN(n_iter, T, params, fct):
    
    out = pd.DataFrame()
    for i in np.arange(n_iter):

        N=19000   

        I0 = 1
        R0 = 0
        S0 = N-I0
        
        S_list, I_list, R_list = simulate(T, S0, I0, R0, params, fct)
        
        # combine the results
        data = pd.DataFrame({'date':np.arange(T+1),
                             'S' : S_list,
                             'I' : I_list,
                             'R' : R_list,
                            })
        data['mun_code'] = i
        data['N'] = N
        
        # append
        out = out.append(data)
                
    return out
          


def generate_data(n_iter, T, params, fct):
    
    out = pd.DataFrame()
    for i in np.arange(n_iter):

        mu=19000
        sigma = 8000
        N=np.clip(int(np.random.normal(mu, sigma)), a_min=100, a_max=1000000)    

        I0 = 1
        R0 = 0
        S0 = N-I0
        
        S_list, I_list, R_list = simulate(T, S0, I0, R0, params, fct)
        
        # combine the results
        data = pd.DataFrame({'date':np.arange(T+1),
                             'S' : S_list,
                             'I' : I_list,
                             'R' : R_list,
                            })
        data['mun_code'] = i
        data['N'] = N
        
        # append
        out = out.append(data)
                
    return out
       



# --- tradaboost ---


def run_pipeline_tradaboost(source_data, target_train, target_test, test_identifiers, target_train_identifiers, cutoff_dates, lag=9, maxgap=9):
    
    
    # prep output
    predictions = pd.DataFrame()


    for gap in np.arange(0,maxgap):

        # for each iteration, use only lag-1 number of features for training (plus the target feature)

        # drop the gap variables
        # for each iteration of the loop, drop the latest gap:
        p = 'lag'+str(gap)
        selector = [x[-len(p):]==p for x in source_data.columns]
        source_data.drop(source_data.loc[:,selector].columns, axis=1, inplace=True)

        source_lags = source_data.iloc[:,:lag+1].copy()
        
        target_lags = target_train.loc[:,source_lags.columns].copy()
        target_test_lags = target_test.loc[:,source_lags.columns].copy()
        

        
        
        # iterate over different levels of data availability
        for cutoff in cutoff_dates:
            
            # reduce the training date
            target_cutoff = target_lags[target_train_identifiers.date<=cutoff].copy()
            
            #### TrAdaBoost regressor
            
            tr = TrAdaBoostR2(n_estimators=10, 
                              estimator=RandomForestRegressor(n_estimators=50, max_depth=None))
            tr.fit(Xs = source_lags.drop('cases',axis=1), 
                      ys = source_lags.cases, 
                      Xt = target_cutoff.drop('cases',axis=1), 
                      yt = target_cutoff.cases)
            
            # predictions
            pred_tr = tr.predict(target_test_lags.drop('cases', axis=1))

    
            
            # combine predictions
            temp = pd.DataFrame({'date': test_identifiers.date,
                                'mun_code': test_identifiers.mun_code,
                                'actual':target_test_lags.cases,
                                'pred':pred_tr.flatten()
                                })

            # add column to identify the number of months skipped
            # i.e. predict gap+1 months ahead
            temp['gap'] = gap
            
            # add column to identify the cutoff date
            # i.e. predict gap+1 months ahead
            temp['cutoff'] = cutoff

            # append
            predictions = predictions.append(temp)
            
    return predictions


# --- RF ---
def prediction_horizons(use_train, use_test, maxgap, min_date_train, min_date_test, lag=10):

    # prep output
    predictions_allgaps = pd.DataFrame()

    max_lags = lag+maxgap

    train_lags = data_prep(use_train.copy(), gap=0, max_lags=max_lags, min_date=min_date_train)
    test_lags = data_prep(use_test.copy(), gap=0, max_lags=max_lags, min_date=min_date_test)


    for gap in np.arange(1,maxgap):
        ### prep the data:

        # how many lags?
        max_lags = 12+gap

        # drop the gap variables
        # for each iteration of the loop, drop the latest gap:
        p = 'lag'+str(gap)
        selector = [x[-len(p):]==p for x in train_lags.columns]
        train_lags.drop(train_lags.loc[:,selector].columns, axis=1, inplace=True)
        test_lags.drop(test_lags.loc[:,selector].columns, axis=1, inplace=True)

        # for each iteration, use only lag-1 number of features for training (plus the target feature)
        # normalize
        train_norm, test_norm, mean, std  = normalize_data(train_lags.iloc[:,:lag+2], test_lags.iloc[:,:lag+2])

        # train model 
        rf = RandomForestRegressor(n_estimators=50, max_depth=None)
        rf.fit(X=train_norm.drop(['cases'],axis=1) , y=train_norm.cases)

        # test
        pred = rf.predict(test_norm.drop(['cases'],axis=1))
        # non-normalize
        pred_real = (pred*std['cases'])+mean['cases']

        # combine predictions
        predictions = pd.DataFrame({'date': test_lags.date,
                                    'mun_code': test_lags.mun_code,
                                    'true_real':test_lags.cases,
                                    'pred_real':pred_real
                                   })

        # add column to identify the number of months skipped
        # i.e. predict gap+1 months ahead
        predictions['gap'] = gap

        # append
        predictions_allgaps = predictions_allgaps.append(predictions)


    return predictions_allgaps




# --- RF synthetic ---
def run_RF_transfer(use_train, train_mean, train_std, test_datasets, maxgap, lag=10):
    train_lags = use_train.copy()
    # prep output
    predictions_all = pd.DataFrame()
    
    # iterate over the prediction horizons
    for gap in np.arange(0,9):
        # drop the gap variables
        # for each iteration of the loop, drop the latest gap:
        p = 'lag'+str(gap)
        selector = [x[-len(p):]==p for x in train_lags.columns]
        train_lags.drop(train_lags.loc[:,selector].columns, axis=1, inplace=True)

        train_norm = train_lags.iloc[:,:lag+2].copy()
        # drop the indicator vars
        train_norm.drop(['mun_code', 'date'], axis=1, inplace=True)
        print(train_norm.columns)
        
        # train RF model 
        rf = RandomForestRegressor(n_estimators=50, max_depth=None)
        rf.fit(X=train_norm.drop(['cases'],axis=1) , y=train_norm.cases)

        # iterate over the test datasets (i.e. other diseases)
        for test_lags in test_datasets:
            
            # for each iteration, use only lag-1 number of features for training (plus the target feature)
            # normalize
            test_norm = test_lags.copy()
            test_norm.iloc[:,2:] = (test_norm.iloc[:,2:] - train_mean) / train_std
            test_norm = test_norm.loc[:,train_norm.columns]
            print(test_norm.columns)
            # test the model
            pred = rf.predict(test_norm.drop(['cases'],axis=1))

            # combine predictions
            predictions = pd.DataFrame({'date': test_lags.date,
                                        'mun_code': test_lags.mun_code,
                                        'true':test_norm.cases,
                                        'pred':pred,
                                       })

            # add identifying columns
            predictions['gap'] = gap
            predictions['dataset'] = test_lags.name # 

            # append
            predictions_all = predictions_all.append(predictions)
        
    return predictions_all




# --- NN synthetic ---
def run_transfer_models_lowdata_multiplediseases(use_train, test_datasets, train2_datasets, cutoff_dates, ckpt_name, lag=9, maxgap=9):
    """implement base and transfer models for different levels of data availability
    """
    
    # prep output
    predictions_allgaps = pd.DataFrame()
    

    for gap in np.arange(1,maxgap):

        # for each iteration, use only lag-1 number of features for training (plus the target feature)

        # drop the gap variables
        # for each iteration of the loop, drop the latest gap:
        p = 'lag'+str(gap)
        selector = [x[-len(p):]==p for x in use_train.columns]
        use_train.drop(use_train.loc[:,selector].columns, axis=1, inplace=True)
        
        train_lags = use_train.iloc[:,:lag+3].copy()
        print(train_lags.columns)
        
        # -------- train base model (MODEL 1)
        # i.e. train and test on different diseases

        # set up the optimizer learning schedule
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
        )

        base_model = keras.Sequential([
            keras.Input(shape=(9)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1),
        ])

        base_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  
            loss=keras.losses.MeanSquaredError()
        )

        # train the base model
        base_model.fit(
            train_lags.drop(['mun_code','date','cases'],axis=1).values,
            train_lags.cases.values,
            batch_size=300,
            validation_split = 0.1,
            epochs=500,
            callbacks = [EarlyStopping(monitor='val_loss', patience=10)],
            verbose=0
        )
        
        # save model 1
        base_model.save_weights(ckpt_name+'+gap'+str(gap))
        
        # iterate over the test / finetune datasets
        for i,use_test in enumerate(test_datasets):
            
            # select relevant features from test set
            test_lags = use_test.loc[:,train_lags.columns]
            # select corresponding finetune set and select features
            train_lags2 = train2_datasets[i]
            train_lags2 = train_lags2.loc[:,train_lags.columns]
            

            # predictions of the base model
            pred1 = base_model.predict(test_lags.drop(['mun_code','date','cases'],axis=1).values)

            
        
            # iterate over different levels of data availability
            for cutoff in cutoff_dates:

                # reduce the training date
                train2_cutoff = train_lags2[train_lags2.date<=cutoff].copy()

                # -------- transfer model (TRANSFER MODEL 1.1)

                # i.e. update last layer of the model using the second disease

                ### set up the transfer model with the same architecture
                transfer_model = keras.Sequential([
                    keras.Input(shape=(9)),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1),
                ])

                # load weights from base model
                transfer_model.load_weights(ckpt_name+'+gap'+str(gap)).expect_partial()

                # freeze all layers except the last one.
                for layer in transfer_model.layers[:-1]:
                    layer.trainable = False

                # recompile 
                initial_learning_rate = 0.001
                lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
                )
                transfer_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  
                    loss=keras.losses.MeanSquaredError()
                )
                # train (only update the weights of the last layer)
                transfer_model.fit(
                    train2_cutoff.drop(['mun_code','date','cases'],axis=1).values,
                    train2_cutoff.cases.values,
                    batch_size=300,
                    validation_split = 0.1,
                    epochs=500,
                    callbacks = [EarlyStopping(monitor='val_loss', patience=10)],
                    verbose=0
                )

                # predictions
                pred2 = transfer_model.predict(test_lags.drop(['mun_code','date','cases'],axis=1).values)




                # -------- finetune model (TRANSFER MODEL 1.2)
                # i.e. unfreeze and update all layers using data from second disease

                # unfreeze the transfer model
                transfer_model.trainable = True

                # recompile with low learning rate
                transfer_model.compile(optimizer=keras.optimizers.Adam(1e-5),
                                       loss=keras.losses.MeanSquaredError()
                                      )

                # continue training for small number of epochs
                transfer_model.fit(train2_cutoff.drop(['mun_code','date','cases'],axis=1).values,
                                    train2_cutoff.cases.values, 
                                    epochs=10,)

                # predictions
                pred3 = transfer_model.predict(test_lags.drop(['mun_code','date','cases'],axis=1).values)


                # combine predictions
                predictions = pd.DataFrame({'date': test_lags.date,
                                            'mun_code': test_lags.mun_code,
                                            'actual':test_lags.cases,
                                            'pred1':pred1.flatten(),
                                            'pred2':pred2.flatten(),
                                            'pred3':pred3.flatten(),
                                           })

                # add column to identify the number of months skipped
                # i.e. predict gap+1 months ahead
                predictions['gap'] = gap

                # add column to identify the cutoff date
                # i.e. predict gap+1 months ahead
                predictions['cutoff'] = cutoff
                
                # add column to identify the finetune/test dataset
                predictions['dataset'] = use_test.name

                # append
                predictions_allgaps = predictions_allgaps.append(predictions)

    return predictions_allgaps


# --- NN baseline ---
def run_model2(use_test, use_train2, test_identifiers, ckpt_name, lag=9, maxgap=9):

    # prep output
    predictions_allgaps2 = pd.DataFrame()


    for gap in np.arange(1,maxgap):
        # for each iteration, use only lag-1 number of features for training (plus the target feature)
        # drop the gap variables
        # for each iteration of the loop, drop the latest gap:
        p = 'lag'+str(gap)
        selector = [x[-len(p):]==p for x in use_train2.columns]
        use_test.drop(use_test.loc[:,selector].columns, axis=1, inplace=True)
        use_train2.drop(use_train2.loc[:,selector].columns, axis=1, inplace=True)

        test_lags = use_test.iloc[:,:lag+1].copy()
        train_lags2 = use_train2.iloc[:,:lag+1].copy()

        # -------- train base model (MODEL 1)
        # i.e. train and test on different diseases

        # set up the optimizer learning schedule
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
        )

        base_model = keras.Sequential([
            keras.Input(shape=(9)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1),
        ])

        base_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  
            loss=keras.losses.MeanSquaredError()
        )

        # train
        base_model.fit(
            train_lags2.drop(['cases'],axis=1).values,
            train_lags2.cases.values,
            batch_size=300,
            validation_split = 0.1,
            epochs=500,
            callbacks = [EarlyStopping(monitor='loss', patience=10)],
            verbose=0
        )

        # predictions
        pred_model2 = base_model.predict(test_lags.drop(['cases'],axis=1).values)

        # save model 1
        base_model.save_weights(ckpt_name+'+gap'+str(gap))


        # combine predictions
        predictions = pd.DataFrame({'date': test_identifiers.date,
                                    'mun_code': test_identifiers.mun_code,
                                    'actual':test_lags.cases,
                                    'pred_model2':pred_model2.flatten(),

                                   })

        # add column to identify the number of months skipped
        # i.e. predict gap+1 months ahead
        predictions['gap'] = gap

        # append
        predictions_allgaps2 = predictions_allgaps2.append(predictions)
    
    return predictions_allgaps2



# --- NN, empirical ---
def run_transfer_models_lowdata(use_train, use_test, use_train2, test_identifiers, train2_identifiers, cutoff_dates, ckpt_name, lag=9, maxgap=9):
    """implement base and transfer models for different levels of data availability and prediction horizons
    """
    
    # prep output
    predictions_allgaps = pd.DataFrame()


    for gap in np.arange(1,maxgap):

        # for each iteration, use only lag-1 number of features for training (plus the target feature)

        # drop the gap variables
        # for each iteration of the loop, drop the latest gap:
        p = 'lag'+str(gap)
        selector = [x[-len(p):]==p for x in use_train.columns]
        use_train.drop(use_train.loc[:,selector].columns, axis=1, inplace=True)
        use_test.drop(use_test.loc[:,selector].columns, axis=1, inplace=True)
        use_train2.drop(use_train2.loc[:,selector].columns, axis=1, inplace=True)

        train_lags = use_train.iloc[:,:lag+1].copy()
        test_lags = use_test.iloc[:,:lag+1].copy()
        train_lags2 = use_train2.iloc[:,:lag+1].copy()

        # -------- train base model (MODEL 1)
        # i.e. train and test on different diseases

        # set up the optimizer learning schedule
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
        )

        base_model = keras.Sequential([
            keras.Input(shape=(9)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1),
        ])

        base_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  
            loss=keras.losses.MeanSquaredError()
        )

        # train the base model
        base_model.fit(
            train_lags.drop(['cases'],axis=1).values,
            train_lags.cases.values,
            batch_size=300,
            validation_split = 0.1,
            epochs=500,
            callbacks = [EarlyStopping(monitor='val_loss', patience=10)],
            verbose=0
        )

        # predictions
        pred1 = base_model.predict(test_lags.drop(['cases'],axis=1).values)

        # save model 1
        base_model.save_weights(ckpt_name+'+gap'+str(gap))

        
        # iterate over different levels of data availability
        for cutoff in cutoff_dates:
            
            # reduce the training date
            train2_cutoff = train_lags2[train2_identifiers.date<=cutoff].copy()
            
            # -------- transfer model (TRANSFER MODEL 1.1)

            # i.e. update last layer of the model using the second disease

            ### set up the transfer model with the same architecture
            transfer_model = keras.Sequential([
                keras.Input(shape=(9)),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1),
            ])

            # load weights from base model
            transfer_model.load_weights(ckpt_name+'+gap'+str(gap)).expect_partial()

            # freeze all layers except the last one.
            for layer in transfer_model.layers[:-1]:
                layer.trainable = False

            # recompile 
            initial_learning_rate = 0.001
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
            )
            transfer_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  
                loss=keras.losses.MeanSquaredError()
            )
            # train (only update the weights of the last layer)
            transfer_model.fit(
                train2_cutoff.drop(['cases'],axis=1).values,
                train2_cutoff.cases.values,
                batch_size=300,
                validation_split = 0.1,
                epochs=500,
                callbacks = [EarlyStopping(monitor='val_loss', patience=10)],
                verbose=0
            )

            # predictions
            pred2 = transfer_model.predict(test_lags.drop(['cases'],axis=1).values)



            # -------- finetune model (TRANSFER MODEL 1.2)
            # i.e. unfreeze and update all layers using data from second disease

            # unfreeze the transfer model
            transfer_model.trainable = True

            # recompile with low learning rate
            transfer_model.compile(optimizer=keras.optimizers.Adam(1e-5),
                                   loss=keras.losses.MeanSquaredError()
                                  )

            # continue training for small number of epochs
            transfer_model.fit(train2_cutoff.drop(['cases'],axis=1).values,
                                train2_cutoff.cases.values, 
                                epochs=10,)

            # predictions
            pred3 = transfer_model.predict(test_lags.drop(['cases'],axis=1).values)


            # combine predictions
            predictions = pd.DataFrame({'date': test_identifiers.date,
                                        'mun_code': test_identifiers.mun_code,
                                        'actual':test_lags.cases,
                                        'pred1':pred1.flatten(),
                                        'pred2':pred2.flatten(),
                                        'pred3':pred3.flatten(),
                                       })

            # add column to identify the number of months skipped
            # i.e. predict gap+1 months ahead
            predictions['gap'] = gap
            
            # add column to identify the cutoff date
            # i.e. predict gap+1 months ahead
            predictions['cutoff'] = cutoff

            # append
            predictions_allgaps = predictions_allgaps.append(predictions)

    return predictions_allgaps


