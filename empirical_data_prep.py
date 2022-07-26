### Data Prep


import numpy as np
import pandas as pd


from functions import data_prep


## Dengue and Zika


data = pd.read_csv('data_dengue-zika.csv', parse_dates=['date'])

data.date = pd.to_datetime(data.date)

zika = data[data.dataset=='Zika (weekly)']
dengue = data[data.dataset=='Dengue (weekly)']

zika_lags = data_prep(zika.copy(), gap=0, max_lags=10, min_date='2016-01-01')
dengue_lags = data_prep(dengue.copy(), gap=0, max_lags=10, min_date='2014-01-01')

# split the zika cities into half randomly
zika_cities = zika.mun_code.unique()
zika_train_cities, zika_test_cities = train_test_split(zika_cities, test_size=0.5, random_state=12)

### dengue splits
# train set (2014-15)
# note: only use the data available before 2016
dengue_train = dengue_lags[dengue_lags.date<'2016-01-01']


### zika splits
# test set (2016, half the cities)
zika_test = zika_lags[zika_lags.date<'2017-01-01']
zika_test = zika_test[zika_test.mun_code.isin(zika_test_cities)]
# train set (2016, other half)
zika_train = zika_lags[zika_lags.date<'2017-01-01']
zika_train = zika_train[zika_train.mun_code.isin(zika_train_cities)]
# expanded train set (half cities in 2016, and all cities in 2017-20)
zika_train_expanded = zika_train.copy()
zika_train_expanded = zika_train_expanded.append(zika_lags[zika_lags.date>='2017-01-01'])


# save
zika_train.to_csv('data/zika_train.csv')
zika_test.to_csv('data/zika_test.csv')
dengue_train.to_csv('data/dengue_train.csv')






### Flu & COVID-19

# read data
flu = pd.read_csv('data_flu.csv', parse_dates=['date'])
covid = pd.read_csv('data_covid.csv', parse_dates=['data'])


# data prep
flu_lags = data_prep(flu.copy(), gap=0, max_lags=10, min_date='2013-01-01')
covid_lags = data_prep(covid.copy(), gap=0, max_lags=10, min_date='2020-03-28')


# split the covid cities into half randomly
# covid splits
covid_cities = covid.mun_code.unique()
covid_train_cities, covid_test_cities = train_test_split(covid_cities, test_size=0.5, random_state=12)


covid_test = covid[covid.mun_code.isin(covid_test_cities)]
covid_train = covid[covid.mun_code.isin(covid_train_cities)]


### flu splits
# train set (2013-19)
# note: only use the data available before 2020
flu_train = flu_lags[flu_lags.date<'2020-03-28']


### covid splits
# test set (half the cities, 2020-2021)
covid_test = covid_lags[covid_lags.mun_code.isin(covid_test_cities)]
# train set (2020-21, other half)
covid_train = covid_lags[covid_lags.mun_code.isin(covid_train_cities)]



# save
covid_train.to_csv('data/covid_train.csv')
covid_test.to_csv('data/covid_test.csv')
flu_train.to_csv('data/flu_train.csv')


