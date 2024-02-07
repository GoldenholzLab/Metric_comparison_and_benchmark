#%%
import numpy as np
import pickle

#%% Parameter setting
nb_month = 36 # diary length (month)
nb_boost = 100 # number of permutation test
sim_sz_range = np.arange(0,31) # seizure frequency range to generate
winlen=90 # window length to perform seizure forecasting (days)
DAY_IN_A_MONTH = 30

# placeholder
y_score_list = []
y_true_list = []

for sz_i in sim_sz_range:
    off_period = DAY_IN_A_MONTH-sz_i
    y_true = np.tile(np.concatenate([np.ones(sz_i),np.zeros(off_period)]),nb_month)
    y_true_list.append(y_true[winlen:])

    # RMR
    y_rmr = np.zeros(len(y_true)-winlen)
    for w_i in np.arange(winlen, len(y_true)):
        y_rmr[w_i-winlen] = np.mean(y_true[(w_i-winlen):w_i])
    y_score_list.append(y_rmr)

#%% save data
with open('simulated_dataset.p','wb') as f:
    simulated_dataset = {'y_score_list':y_score_list,
                         'y_true_list':y_true_list}
    pickle.dump(simulated_dataset,f)