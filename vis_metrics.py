#%%
import os
import numpy as np
from joblib import Parallel, delayed
import pickle
# visualization
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from model_lib import *

#%% calculate metrics visualization function
def cal_metrics2plot(y_true_list, y_score_list, is_permute_truth=True, nb_boost=100):
    """Visualize metrics (Brier, AUC ROC, Calibration Curve, and AUC PR) across patients.

    Args:
        y_true_list (list): list of true labels. Each element is a sample from a patients.
        y_score_list (list): list of model prediction scores. Each element is a sample from a patients.
        is_permute_truth (bool): Return metrics from permuted truth if True.
        nb_boost (int): Number of times for permutation tests
        
    Returns:
        

    """
    plt_brier_list = np.zeros(len(y_true_list))
    plt_aucroc_list = np.zeros(len(y_true_list))
    plt_aucpr_list = np.zeros(len(y_true_list))
    permute_brier = np.zeros(len(y_true_list))
    permute_aucroc = np.zeros(len(y_true_list))
    permute_aucpr = np.zeros(len(y_true_list))
    permute_p0 = np.zeros(len(y_true_list))
    permute_p1 = np.zeros(len(y_true_list))
    pr_curve =[]
    permute_pr = []
    
    for subj_i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        if len(np.unique(y_true))>1:
            # Brier
            plt_brier_list[subj_i] = np.mean(np.square(y_true-y_score))
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_true,y_score)
            plt_aucroc_list[subj_i] = auc(fpr,tpr)
            # PR AUC
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt_aucpr_list[subj_i] = auc(recall,precision)
            pr_curve.append((recall,precision))
        else:
            plt_brier_list[subj_i] = 0
            plt_aucroc_list[subj_i] = np.nan
            plt_aucpr_list[subj_i] = np.nan
            pr_curve.append([])
        
        # Permutation
        if is_permute_truth:
            tmp_b = [] # store brier for each permutation
            tmp_a = [] # store AUCROC for each permutation
            tmp_ap = [] # store AUCPR for each permutation
            tmp_p0 = [] # store actual prob. when y_score=0 for each permutation
            tmp_p1 = [] # store actual prob. when y_score=1 for each permutation
            for boost_i in range(nb_boost):
                if len(np.unique(y_true))>1:
                    y_permuted = np.random.permutation(y_true)
                    tmp_b.append(np.mean(np.square(y_permuted.astype(int)-y_true.astype(int))))
                    fpr,tpr,_ = roc_curve(y_true,y_permuted)
                    tmp_a.append(auc(fpr,tpr))
                    precision,recall,_ = precision_recall_curve(y_true,y_permuted)
                    tmp_ap.append(auc(recall,precision))
                    permute_pr.append((recall,precision))
                    tmp_p0.append(np.mean(y_true,where=~y_permuted.astype(bool)))
                    tmp_p1.append(np.mean(y_true,where=y_permuted.astype(bool)))
                else:
                    permute_pr.append([])
            if len(tmp_b)>0:
                permute_p0[subj_i] = np.mean(tmp_p0)
                permute_p1[subj_i] = np.mean(tmp_p1)
                permute_brier[subj_i] = np.mean(tmp_b)
            else:
                permute_p0[subj_i] = 0
                permute_p1[subj_i] = 1
                permute_brier[subj_i] = 0

            if len(tmp_a)>1:
                tmp_a = np.mean(tmp_a)
                tmp_ap = np.mean(tmp_ap)
            else:
                tmp_a = np.nan
                tmp_ap = np.nan
            permute_aucroc[subj_i] = tmp_a
            permute_aucpr[subj_i] = tmp_ap
    
    return (plt_brier_list,plt_aucroc_list,plt_aucpr_list),\
           (permute_brier,permute_aucroc,permute_aucpr,permute_p0,permute_p1),\
           (y_true_list, y_score_list), (pr_curve,permute_pr)
    
#%% Calculate ST data
loadpath = './parameter/ST_data/'
savepath = './figure/ST_data/val_data2/'
if not os.path.isdir(savepath):
    os.mkdir(savepath)
model_name = 'MLP_w_trainedST_val2'
winlen = 90

with open(os.path.join(loadpath,f'vis_result_{model_name}.p'),'rb') as f:
    vis_data = pickle.load(f)
with open(os.path.join(loadpath,f'test_result_{model_name}.p'),'rb') as f:
    test_data = pickle.load(f)

diaries = test_data['diaries']
val_idx = np.array(test_data['val_idx'])
train_idx = np.array(test_data['train_idx'])
y_true_list = [x[winlen:] for i,x in enumerate(diaries) if i in np.concatenate([val_idx,train_idx])]
y_score_list = [moving_average(x,n=winlen)[:-1] for i,x in enumerate(diaries) if i in np.concatenate([val_idx,train_idx])]

(plt_brier_list,plt_aucroc_list,plt_aucpr_list),\
(permute_brier,permute_aucroc,permute_aucpr,permute_p0,permute_p1),\
(y_true_list, y_score_list),(pr_curve,permute_pr)= cal_metrics2plot(y_true_list,y_score_list)

with open(os.path.join(savepath,'plt_st_data.p'),'wb') as f:
    data = {
        'plt_brier_list':plt_brier_list,
        'plt_aucroc_list':plt_aucroc_list,
        'plt_aucpr_list':plt_aucpr_list,
        'permute_brier':permute_brier,
        'permute_aucroc':permute_aucroc,
        'permute_aucpr':permute_aucpr,
        'permute_p0':permute_p0,
        'permute_p1':permute_p1,
        'y_true_list':y_true_list,
        'y_score_list':y_score_list,
        'pr_curve':pr_curve,
        'permute_pr':permute_pr
    }
    pickle.dump(data,f)

#%% Load ST data
loadpath = './parameter/ST_data/'
savepath = './figure/ST_data/val_data2/'
if not os.path.isdir(savepath):
    os.mkdir(savepath)
model_name = 'MLP_w_trainedST_val2'
winlen = 90

with open(os.path.join(loadpath,f'vis_result_{model_name}.p'),'rb') as f:
    vis_data = pickle.load(f)
with open(os.path.join(loadpath,f'test_result_{model_name}.p'),'rb') as f:
    test_data = pickle.load(f)

diaries = test_data['diaries']
val_idx = np.array(test_data['val_idx'])
train_idx = np.array(test_data['train_idx'])
y_true_list = [x[winlen:] for i,x in enumerate(diaries) if i in np.concatenate([val_idx,train_idx])]
y_score_list = [moving_average(x,n=winlen)[:-1] for i,x in enumerate(diaries) if i in np.concatenate([val_idx,train_idx])]

with open(os.path.join(savepath,'plt_st_data.p'),'rb') as f:
    data = pickle.load(f)
    plt_brier_list=data['plt_brier_list']
    plt_aucroc_list=data['plt_aucroc_list']
    plt_aucpr_list=data['plt_aucpr_list']
    permute_brier=data['permute_brier']
    permute_aucroc=data['permute_aucroc']
    permute_aucpr=data['permute_aucpr']
    permute_p0=data['permute_p0']
    permute_p1=data['permute_p1']
    y_true_list=data['y_true_list']
    y_score_list=data['y_score_list'] 

#%% Generate Toy model
nb_day = 10000
nb_boost = 100
winlen = 90
test_range = np.arange(0,31)

# booststrapping
y_score_list = []
y_true_list = []

for sz_i in test_range:
    off_period = 30-sz_i
    y_true = np.tile(np.concatenate([np.ones(sz_i),np.zeros(off_period)]),100)
    y_true_list.append(y_true[winlen:])

    # RMR
    y_rmr = np.zeros(len(y_true)-winlen)
    for w_i in np.arange(winlen, len(y_true)):
        y_rmr[w_i-winlen] = np.mean(y_true[(w_i-winlen):w_i])
    y_score_list.append(y_rmr)
        
(plt_brier_list,plt_aucroc_list,plt_aucpr_list),\
(permute_brier,permute_aucroc,permute_aucpr,permute_p0,permute_p1),\
(y_true_list, y_score_list),(pr_curve,permute_pr) = cal_metrics2plot(y_true_list,y_score_list)

#%% Load Empatica 
with open(os.path.join(loadpath,'Empatica_plt_st_data.p'),'rb') as f:
    data = pickle.load(f)
    plt_brier_list=data['plt_brier_list']
    plt_aucroc_list=data['plt_aucroc_list']
    plt_aucpr_list=data['plt_aucpr_list']
    permute_brier=data['permute_brier']
    permute_aucroc=data['permute_aucroc']
    permute_aucpr=data['permute_aucpr']
    permute_p0=data['permute_p0']
    permute_p1=data['permute_p1']
    y_true_list=data['y_true_list']
    y_score_list=data['y_score_list']


#%% visualization
is_one_fig = True
fontsize = 25
linewidth=3
marker_size_base=100
# set maximum frequency for calculating the metrics
thres_freq=9
est_mSF = np.array([np.mean(x)*30 for x in y_true_list])
sort_idx = np.argsort(est_mSF.reshape(-1))
plt_mSF = est_mSF[sort_idx]
freq_max = len(plt_mSF)
bins = np.arange(0.5,thres_freq+1,1)
inds = np.digitize(plt_mSF,bins=bins).reshape(-1)
bins = bins[:-1]


if is_one_fig:
    fig, ax = plt.subplots(4,1,figsize=(8,30),dpi=300)
else:
    ax = []
    for p_i in range(4):
        fig = plt.figure(figsize=(8,8),dpi=300)
        ax.append(fig.add_subplot(111))
    
    
brier_b = np.zeros(len(bins))
auc_b = np.zeros(len(bins))
nb_sampl = [np.sum(inds==c_i) for c_i in np.arange(inds[0],thres_freq+1)]
marker_size = np.array(nb_sampl)/np.sum(nb_sampl)*marker_size_base

# Brier
brier_b = [plt_brier_list[sort_idx][inds==c_i] for c_i in np.arange(1,thres_freq+1)]
# ROC AUC
auc_b = [plt_aucroc_list[sort_idx][inds==c_i] for c_i in np.arange(1,thres_freq+1)]
# PR AUC
auc_p = [plt_aucpr_list[sort_idx][inds==c_i] for c_i in np.arange(1,thres_freq+1)]

# line
ax[1].plot(bins+0.5,[np.mean(x) for x in brier_b],'k-',label='Moving Average',linewidth=linewidth)
ax[1].plot(bins+0.5,[np.mean(x)+np.std(x)/len(x) for x in brier_b],'k--',linewidth=linewidth)
ax[1].plot(bins+0.5,[np.mean(x)-np.std(x)/len(x) for x in brier_b],'k--',linewidth=linewidth)
ax[2].plot(bins+0.5,[np.nanmean(x) for x in auc_b],'k-',label='Moving Average',linewidth=linewidth)
ax[2].plot(bins+0.5,[np.nanmean(x)+np.std(x)/len(x) for x in auc_b],'k--',linewidth=linewidth)
ax[2].plot(bins+0.5,[np.nanmean(x)-np.std(x)/len(x) for x in auc_b],'k--',linewidth=linewidth)
ax[3].plot(bins+0.5,[np.nanmean(x) for x in auc_p],'k-',label='Moving Average',linewidth=linewidth)
ax[3].plot(bins+0.5,[np.nanmean(x)+np.std(x)/len(x) for x in auc_p],'k--',linewidth=linewidth)
ax[3].plot(bins+0.5,[np.nanmean(x)-np.std(x)/len(x) for x in auc_p],'k--',linewidth=linewidth)
# marker
for b_i in range(len(brier_b)):
    ax[1].plot(bins[b_i]+0.5,np.mean(brier_b[b_i]),'ko',markersize=marker_size[b_i])
    ax[2].plot(bins[b_i]+0.5,np.nanmean(auc_b[b_i]),'ko',markersize=marker_size[b_i])
    ax[3].plot(bins[b_i]+0.5,np.nanmean(auc_p[b_i]),'ko',markersize=marker_size[b_i])

# Calibration curve
is_mksize = True
cmap = plt.get_cmap('jet')
num_colors = thres_freq
# colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
colors = ['b','','','','g','','','','r']
local_color = ['b','g','r']
lc_i = 0
# treat all patients in the same bins as the same patients and plot one calibration curve
# for cal_i in np.arange(1,thres_freq+1):
for cal_i in [1,5,9]:
    bin_idx = np.arange(len(y_true_list))[sort_idx][inds==cal_i]
    y_true_b = []
    y_score_b = []
    for idx in bin_idx:
        y_true_b.append(y_true_list[idx])
        y_score_b.append(y_score_list[idx])
    y_true_b = np.concatenate(y_true_b)
    y_score_b = np.concatenate(y_score_b)
    prob_true, prob_pred = calibration_curve(y_true_b, y_score_b, n_bins=10)
    cali_bins = np.arange(0,1.1,0.1)
    mk_size_bin,_=np.histogram(y_score_b,bins=cali_bins)
    ax[0].plot(prob_pred,prob_true,'-',color=local_color[lc_i],label=f'Moving Average (SF={cal_i})',linewidth=linewidth)
    if is_mksize:
        # marker size
        mk_size_bin = np.array([np.min([x,20]) for x in mk_size_bin/np.sum(mk_size_bin)*marker_size_base])
        for b_i in range(len(prob_true)):
            if mk_size_bin[0]!=0:
                ax[0].plot(prob_pred[b_i],prob_true[b_i],'o',color=local_color[lc_i],markersize=mk_size_bin[b_i])
            else:
                ax[0].plot(prob_pred[b_i],prob_true[b_i],'o',color=local_color[lc_i],markersize=20)
    else:
        # Alpha
        mk_ratio = 0.1
        mk_size_bin = mk_size_bin/np.sum(mk_size_bin)*(1-mk_ratio)+mk_ratio
        for b_i in range(len(prob_true)):
            ax[0].plot(prob_pred[b_i],prob_true[b_i],'o',color=colors[cal_i-1],markersize=15, alpha=mk_size_bin[b_i])
    lc_i+=1
    
# Permute model        
ax[1].plot(bins+0.5,[np.mean(permute_brier[sort_idx][inds==c_i]) for c_i in np.arange(1,thres_freq+1)],color='g',label='Average Permuted Truth',marker='v',linewidth=linewidth)
ax[1].plot(bins+0.5,[np.mean(permute_brier[sort_idx][inds==c_i])+np.std(np.array(permute_brier)[sort_idx][inds==c_i])/np.sqrt(sum(inds==c_i)) for c_i in np.arange(1,thres_freq+1)],color='g',linestyle='--',linewidth=linewidth)
ax[1].plot(bins+0.5,[np.mean(permute_brier[sort_idx][inds==c_i])-np.std(np.array(permute_brier)[sort_idx][inds==c_i])/np.sqrt(sum(inds==c_i)) for c_i in np.arange(1,thres_freq+1)],color='g',linestyle='--',linewidth=linewidth)
ax[2].plot(bins+0.5,[np.nanmean(permute_aucroc[sort_idx][inds==c_i]) for c_i in np.arange(1,thres_freq+1)],color='g',label='Average Permuted Truth',linewidth=linewidth)
ax[2].plot(bins+0.5,[np.nanmean(permute_aucroc[sort_idx][inds==c_i])+np.std(np.array(permute_aucroc)[sort_idx][inds==c_i])/np.sqrt(sum(inds==c_i)) for c_i in np.arange(1,thres_freq+1)],color='g',linestyle='--',linewidth=linewidth)
ax[2].plot(bins+0.5,[np.nanmean(permute_aucroc[sort_idx][inds==c_i])-np.std(np.array(permute_aucroc)[sort_idx][inds==c_i])/np.sqrt(sum(inds==c_i)) for c_i in np.arange(1,thres_freq+1)],color='g',linestyle='--',linewidth=linewidth)
ax[3].plot(bins+0.5,[np.nanmean(permute_aucpr[sort_idx][inds==c_i]) for c_i in np.arange(1,thres_freq+1)],color='g',label='Average Permuted Truth',linewidth=linewidth)
ax[3].plot(bins+0.5,[np.nanmean(permute_aucpr[sort_idx][inds==c_i])+np.std(np.array(permute_aucpr)[sort_idx][inds==c_i])/np.sqrt(sum(inds==c_i)) for c_i in np.arange(1,thres_freq+1)],color='g',linestyle='--',linewidth=linewidth)
ax[3].plot(bins+0.5,[np.nanmean(permute_aucpr[sort_idx][inds==c_i])-np.std(np.array(permute_aucpr)[sort_idx][inds==c_i])/np.sqrt(sum(inds==c_i)) for c_i in np.arange(1,thres_freq+1)],color='g',linestyle='--',linewidth=linewidth)
plt_p0 = [np.nanmean(permute_p0[sort_idx][inds==c_i]) for c_i in np.arange(1,thres_freq+1)]
plt_p1 = [np.nanmean(permute_p1[sort_idx][inds==c_i]) for c_i in np.arange(1,thres_freq+1)]
# marker
for c_i in np.arange(1,thres_freq+1):
    ax[1].plot(bins[c_i-1]+0.5,np.mean(permute_brier[sort_idx][inds==c_i]),'gv',markersize=marker_size[c_i-1])
    ax[2].plot(bins[c_i-1]+0.5,np.nanmean(permute_aucroc[sort_idx][inds==c_i]),'gv',markersize=marker_size[c_i-1])
    ax[3].plot(bins[c_i-1]+0.5,np.nanmean(permute_aucpr[sort_idx][inds==c_i]),'gv',markersize=marker_size[c_i-1])

# calibration curve for permutation truth
# for cal_i in np.arange(1,len(plt_p0),3):
local_color = ['b','g','r']
lc_i = 0
for cal_i in [1,5,9]:
    ax[0].plot([0,1],[plt_p0[cal_i-1],plt_p1[cal_i-1]],'--v',label=f'Average Permuted Truth (SF={cal_i})',color=local_color[lc_i],markersize=15,linewidth=linewidth)
    lc_i+=1

# ax[1].set_xlabel('mSF',fontsize=fontsize)
ax[1].grid()
# ax[1].set_title('Brier',fontsize=fontsize)
ax[1].set_ylim([0,1])
ax[2].axhline(0.5,color='k',linestyle='--')
# ax[2].set_xlabel('Monthly Seizure Freqeuncy',fontsize=fontsize)
# ax[2].set_title('AUC ROC',fontsize=fontsize)
ax[2].grid()
ax[2].set_ylim([0,1])
ax[3].axhline(0.5,color='k',linestyle='--')
ax[3].set_xlabel('Monthly Seizure Freqeuncy',fontsize=fontsize)
ax[3].grid()
ax[3].axis('auto')
# ax[3].set_title('AUC PR',fontsize=fontsize)
ax[3].set_ylim([0,1])
ax[0].plot([0,1],[0,1],'k--')
ax[0].set_xlabel('Estimated Probability',fontsize=fontsize)
ax[0].grid()
ax[0].axis('auto')
# ax[1].set_title('Calibration Curve',fontsize=fontsize)
ax[0].set_ylim([0,1])
for ax_i in range(4):
    ax[ax_i].tick_params(axis='both', which='major', labelsize=fontsize)
# ax[1].legend(fontsize=fontsize)
# ax[2].legend(fontsize=fontsize)
# ax[3].legend(fontsize=fontsize)
# ax[0].legend(fontsize=fontsize-5)
# ax[0].set_ylabel('Observed Probability',fontsize=fontsize)
# ax[1].set_ylabel('Brier',fontsize=fontsize)
# ax[2].set_ylabel('AUC ROC',fontsize=fontsize)
# ax[3].set_ylabel('AUC PR',fontsize=fontsize)
plt.tight_layout()
# plt.show()
savepath='C:/Users/chiyu/OneDrive/Desktop/BILH/Writing/sz effect/figure/'
savename='empatica.png'
plt.savefig(f'{savepath}{savename}', bbox_inches='tight', dpi=300)
plt.close(fig)

#%% PR Curve
with open(os.path.join(savepath,'plt_st_data.p'),'rb') as f:
    data = pickle.load(f)
    pr_curve_st = data['pr_curve']
    permute_pr_st = data['permute_pr']
thres_freq=9
est_mSF = np.array([np.mean(x)*30 for x in y_true_list])
sort_idx = np.argsort(est_mSF.reshape(-1))
plt_mSF = est_mSF[sort_idx]
freq_max = len(plt_mSF)
bins = np.arange(0.5,thres_freq+1,1)
inds = np.digitize(plt_mSF,bins=bins).reshape(-1)
bins = bins[:-1]
target_sf = 2
plt_pr = [x for i,(_,x) in enumerate(sorted(zip(sort_idx,pr_curve))) if inds[i]==target_sf]
plt_permute_pr = [x for i,(_,x) in enumerate(sorted(zip(sort_idx,permute_pr))) if inds[i]==target_sf]

#%%
fig,ax = plt.subplots(1,2, figsize=(10,6))
for x, y in zip(plt_pr,plt_permute_pr):
    if len(x)>0:
        ax[0].plot(x[0],x[1],'k-',alpha=0.1)
        ax[1].plot(y[0],y[1],'k-',alpha=0.1)
ax[0].set_ylabel('Precision')
ax[0].set_xlabel('Recall')
ax[1].set_xlabel('Recall')
ax[0].set_title('MA')
ax[1].set_title('Permuted truth')
plt.show()
