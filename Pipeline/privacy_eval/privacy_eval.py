""""File contains all the functions used to evaluate the privacy scores"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_chunked as pdc

def red_func(D_chunk, start):
    """
    Reduce function applied to the generator of the distance matrix
    Returns a dataframe with the 3 shortest distances.
    """
    df = pd.DataFrame(np.sort(D_chunk, axis=1)[:, 0:3])
    return df


def data_process_priv(df, sample_per):
    """
    Function used for data processing in the privacy calculation
    Returns a scaled and sampled dataframe without duplicates.
    """
    
    df = df.drop_duplicates(keep=False)
    df = df.sample(frac=(sample_per/100), random_state=42).to_numpy()
    scalerdf = StandardScaler()
    scalerdf.fit(df)
    df = scalerdf.transform(df)
    df = df.astype('float32')
    
    return df

def privacy_synth_frame(df_real, df, key, memory):
    """
    Function used in the privacy_compare function to calculate the distances of the synthetic
    and synthetic to real data.
    Returns a dataframe with the DCR and NNDR of the synthetic data and the synthetic to real data.
    
    This function is different from the single_priv_calc function in that it calculates 
    the dcr and nndr over 1 dataset but also between two datasets.
    """
    
    frame_time = time.time()
    print('Starting privacy calculations with', key)

    dist_rs = pdc(df_real, Y=df, reduce_func=red_func, metric='minkowski', n_jobs=-1, working_memory=memory)
    dist_ss = pdc(df, Y=None, reduce_func=red_func, metric='minkowski', n_jobs=-1, working_memory=memory)
  
    results = {'real_'+key: dist_rs, key: dist_ss}
    inbetween_result, end_result, end_df_list, i = {}, {}, [], 0
    
    for frame in results:
        matrix_gen = results[frame]
        for df in matrix_gen:
            fifth_perc = np.percentile(df[i], 5)
            nn_fifth_perc = np.percentile(df[i] / df[i+1], 5)
            inbetween_result[frame] = [fifth_perc, nn_fifth_perc]
            end_df_list.append(pd.DataFrame(inbetween_result, index=['DCR','NNDR']))
        end_result[frame] = pd.concat(end_df_list, axis=1).mean(axis=1)
        i = 1
        
    print_time = round(((time.time() - frame_time) / 60), 2)
    print('Calculated all privacy evaluations for', key, 'in', print_time, 'minutes')
        
    return pd.DataFrame(end_result, index=['DCR','NNDR'])

def single_priv_calc(real, memory):
    
    frame_time = time.time()
    print('Starting privacy calculations with real')
    
    inbetween_result, end_df_list = {}, []
    
    # Creating a distance matrix generator that applies the red_func
    dist_real = pdc(real, Y=None, reduce_func=red_func, metric='minkowski', n_jobs=-1, working_memory=memory)
    # Calculating the 5th percentiles for each chunk in the distance matrix for the real data
    
    for df in dist_real:
        fifth_perc = np.percentile(df[1], 5)
        nn_fifth_perc = np.percentile(df[1] / df[2], 5)
        inbetween_result['Real'] = [fifth_perc, nn_fifth_perc]
        end_df_list.append(pd.DataFrame(inbetween_result, index=['DCR','NNDR']))
    
    end_df = pd.concat(end_df_list, axis=1).mean(axis=1)
    
    print_time = round(((time.time() - frame_time) / 60), 2)
    print('Calculated all privacy evaluations for real in', print_time, 'minutes')
    
    return end_df


def privacy_ratio(privacy_result):
    """
    Function calculates the privacy ratio by either dividing by holdout or by real_holdout
    """

    priv_ratio = {}

    for col in privacy_result.columns:
        if 'real_' not in col:
            if col != 'holdout':
                # Calculating the ratio of synth and real to the holdout set.
                priv_ratio[col] = privacy_result[col] / privacy_result['holdout']
        else:
            if 'holdout' not in col:
                # Calculating the ratio of real_synth to the real_holdout set.
                priv_ratio[col] = privacy_result[col] / privacy_result['real_holdout']

    return round(pd.DataFrame(priv_ratio), 3)