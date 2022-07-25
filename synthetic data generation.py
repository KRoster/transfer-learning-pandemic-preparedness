## Generate SIR data

import numpy as np
import pandas as pd
from scipy.stats import binom, nbinom


from functions import simulate, sird_waning_immunity_step, generate_data



###############################################################################
#### Implementation example 
###############################################################################



## example for generating the data

n_iter = 100
T = 1000


beta = 0.191 # infection rate
gamma = 0.05 # recovery rate
zeta = 0.008 # waning immunity
mu_i = 0.0294 # death rate from disease

params = beta, gamma, zeta, mu_i


## Target disease 
data = generate_data(n_iter, T, params, fct=sird_waning_immunity_step)


data['beta'] = beta
data['gamma'] = gamma
data['zeta'] = zeta
data['mu_i'] = mu_i
data['cases'] = data.I / data.N




## Source diseases

for beta_new in [0.25, 0.3, 0.35]:
    for gamma_new in [0.01, 0.1, 0.15]:
        
        params = beta_new, gamma_new, zeta, mu_i
        
        data = generate_data(n_iter, T, params, fct=sird_waning_immunity_step)
        # per capita
        data['cases'] = data.I / data.N
        # add the params
        data['beta'] = beta_new
        data['gamma'] = gamma_new
        data['zeta'] = zeta
        data['mu_i'] = mu_i
        
        