### Data Prep







def data_prep(data, gap = 0, max_lags=12, min_date='2007-01-01'):
    """add lagged features as columns

    Arguments
    data : dataframe with input data
    max_lags : int, maximum number of past/lagged values


    """
    
    # remove rows with NA date
    data.dropna(axis=0, subset=['date'], inplace=True)
    # check
    data.date = pd.to_datetime(data.date)
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

    # remove lag0 vars
    data_lags.drop(data.columns[1:], axis=1, inplace=True)


    # remove rows with missing values
    data_lags.dropna(inplace=True)
    
    # drop data before min_date
    data_lags = data_lags[data_lags.date>=min_date]

    return data_lags





## Dengue and Zika


data = pd.read_csv('/Users/kirstinroster/Documents/PhD/Data/sinan_all datasets_2021-09-17.csv', parse_dates=['date'])



zika = data[data.dataset=='Zika (weekly)'].drop(['mun_name','year', 'period','period_numeric','lag1', 'lag1_diff', 'lag1_diff_percent', 'dataset'], axis=1)
dengue = data[data.dataset=='Dengue (weekly)'].drop(['mun_name', 'year', 'period','period_numeric','lag1', 'lag1_diff', 'lag1_diff_percent', 'dataset'], axis=1)




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





### Flu & COVID-19

# read data
flu = pd.read_csv('/Users/kirstinroster/Documents/PhD/mobility and covid-19/influenza_merged_2021-06-14.csv', parse_dates=['date'])
covid = pd.read_csv('covid_merged.csv', parse_dates=['data'])


# prep flu data
flu.drop(['sem_not', 'neg','pos_rate', 'epi_year','epi_week'], axis=1, inplace=True)


# prep covid data
# use only city-level data
covid = covid[~covid.codmun.isna()]

# drop unnecessary columns
covid.drop(['regiao', 'codRegiaoSaude', 'nomeRegiaoSaude', 'estado','municipio','coduf',
            'casosAcumulado','populacaoTCU2019', 'obitosAcumulado','obitosNovos',
           'Recuperadosnovos', 'emAcompanhamentoNovos'], axis=1, inplace=True)


# covid aggregate to weekly
covid['year']=covid.data.dt.year
covid = covid.groupby(['codmun', 'semanaEpi','year']).agg({'data':'max', 'casosNovos':'mean' })
covid.reset_index(inplace=True)


# rename
flu.rename({'pos':'cases'}, axis=1, inplace=True)
covid.rename({'codmun':'mun_code',
              'data':'date',
              'casosNovos':'cases'}, axis=1, inplace=True)
covid.drop(['semanaEpi','year'], axis=1,inplace=True)


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


