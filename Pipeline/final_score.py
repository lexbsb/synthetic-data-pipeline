"""File containing the final scoring code """

import pandas as pd
from Pipeline.data_processing.scoring import *



def final_scoring(ratio_results, privacy_ratio, privacy_ratio_id, reggre, classi):
    
    """
    Function calculates the final evaluation score for privacy, utility and fidelity based on custom weights.
    
    ratio_results: the fidelity evaluations compared to the holdout.
    privacy_ratio: the privacy score evaluations compared to the holdout
    reggre: the regression utility metrics
    classi: the classification utility metrics
    
    returns: A combined dataframe with single scores for privacy, utility and fidelity,
             and a dataframe with all the individual scores after the weights have been applied.  
    """
    
    end_score = abs(1 - ratio_results)
    pr_score = privacy_scoring(privacy_ratio)
    pr_score_id = privacy_scoring(privacy_ratio_id)
    pr_score_id = pr_score_id.rename(index=lambda s: s + '-id') #refactor
    ml_util_score = ml_util_scoring(reggre, classi, ratio_results)
    df_fin = pd.concat([end_score, pr_score, pr_score_id, ml_util_score]).T
  
    privacy = {
        'dupe_numbers':1,
        'sum_%_mean_diff':0,
        'sum_%_median_diff':0,
        'sum_%_std_diff':0,
        'binary_val_count_diff':0,
        'correlation_norm':0,
        'real_or_synth_acc':0,
        'jenson_shannon':0,
        'total_variational_dist':0,
        'wasserstein_dist':0,
        'DCR':1,
        'NNDR':1,
        'DCR-id':0,
        'NNDR-id':0,
        'regression':0,
        'classification':0
    }   
    
    privacy_id = {
        'dupe_numbers':1,
        'sum_%_mean_diff':0,
        'sum_%_median_diff':0,
        'sum_%_std_diff':0,
        'binary_val_count_diff':0,
        'correlation_norm':0,
        'real_or_synth_acc':0,
        'jenson_shannon':0,
        'total_variational_dist':0,
        'wasserstein_dist':0,
        'DCR':0,
        'NNDR':0,
        'DCR-id':1,
        'NNDR-id':1,
        'regression':0,
        'classification':0
    }   
    
    fidelity = {
        'dupe_numbers':0,
        'sum_%_mean_diff':0.2,
        'sum_%_median_diff':0.01,
        'sum_%_std_diff':0.01,
        'binary_val_count_diff':0.2,
        'correlation_norm':2,
        'real_or_synth_acc':3,
        'jenson_shannon':1,
        'total_variational_dist':1,
        'wasserstein_dist':0.1,
        'DCR':0,
        'NNDR':0,
        'DCR-id':0,
        'NNDR-id':0,
        'regression':0,
        'classification':0
    }   
    
    utility = {
        'dupe_numbers':0,
        'sum_%_mean_diff':0,
        'sum_%_median_diff':0,
        'sum_%_std_diff':0,
        'binary_val_count_diff':0,
        'correlation_norm':0,
        'real_or_synth_acc':0,
        'jenson_shannon':0,
        'total_variational_dist':0,
        'wasserstein_dist':0,
        'DCR':0,
        'NNDR':0,
        'DCR-id':0,
        'NNDR-id':0,
        'regression':0.5,
        'classification':2
    }   

    priv_score = df_fin.apply(lambda x: x * privacy[x.name])
    priv_score = round(priv_score.T.sum().sort_values(),2) + 2
    
    priv_score_id = df_fin.apply(lambda x: x * privacy_id[x.name])
    priv_score_id = round(priv_score_id.T.sum().sort_values(),2) + 2
    
    fidel_score = df_fin.apply(lambda x: x * fidelity[x.name])
    fidel_score = round(fidel_score.T.sum().sort_values(),2)
    
    ml_score = df_fin.apply(lambda x: x * utility[x.name])
    ml_score = round(ml_score.T.sum().sort_values(), 2)
    
    end_scr = pd.concat([priv_score, priv_score_id, fidel_score, ml_score], axis=1, keys=['privacy', 'privacy on ids', 'fidelity', 'utility'])

    return end_scr, priv_score.sort_values(), priv_score_id.sort_values(), fidel_score.sort_values(), ml_score.sort_values(), round(df_fin, 2)