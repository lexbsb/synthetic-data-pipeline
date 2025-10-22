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
    end_result = {}
    eps = 1e-12  # protection for NNDR division
    
    for frame, matrix_gen in results.items():
        # Collect all rows across chunks before computing percentiles
        d1_list, r_list = [], []
        
        # Decide if distances include self (synth-synth) or not (real-synth)
        same_set = (frame == key)
        
        for blk in matrix_gen:
            # blk has columns [0,1,2] already sorted by red_func
            if same_set:
                # self-distance at column 0; use columns 1 (NN1) and 2 (NN2)
                d1 = blk.iloc[:, 1].to_numpy() if isinstance(blk, pd.DataFrame) else blk[:, 1]
                d2 = blk.iloc[:, 2].to_numpy() if isinstance(blk, pd.DataFrame) else blk[:, 2]
            else:
                # no self; use columns 0 (NN1) and 1 (NN2)
                d1 = blk.iloc[:, 0].to_numpy() if isinstance(blk, pd.DataFrame) else blk[:, 0]
                d2 = blk.iloc[:, 1].to_numpy() if isinstance(blk, pd.DataFrame) else blk[:, 1]
            
            # NNDR with protection and finite filtering
            d2_safe = np.maximum(d2, eps)
            r = d1 / d2_safe

            mask = np.isfinite(d1) & np.isfinite(r)
            if np.any(mask):
                d1_list.append(d1[mask])
                r_list.append(r[mask])
        
        if d1_list:
            d1_all = np.concatenate(d1_list)
            r_all = np.concatenate(r_list)
            fifth_perc = np.percentile(d1_all, 5)
            nn_fifth_perc = np.percentile(r_all, 5)
        else:
            fifth_perc = np.nan
            nn_fifth_perc = np.nan
        
        end_result[frame] = pd.Series([fifth_perc, nn_fifth_perc], index=['DCR','NNDR'])
        
    print_time = round(((time.time() - frame_time) / 60), 2)
    print('Calculated all privacy evaluations for', key, 'in', print_time, 'minutes')
        
    return pd.DataFrame(end_result)

def single_priv_calc(real, memory):
    
    frame_time = time.time()
    print('Starting privacy calculations with real')
    
    # Creating a distance matrix generator that applies the red_func
    dist_real = pdc(real, Y=None, reduce_func=red_func, metric='minkowski', n_jobs=-1, working_memory=memory)
    # Calculating the 5th percentiles for each chunk in the distance matrix for the real data
    
    d1_list, r_list = [], []
    eps = 1e-12  # protection for NNDR division
    
    for blk in dist_real:
        # blk has columns [0,1,2] with column 0 being self-distance (=0)
        if isinstance(blk, pd.DataFrame):
            d1 = blk.iloc[:, 1].to_numpy()
            d2 = blk.iloc[:, 2].to_numpy()
        else:
            d1 = blk[:, 1]
            d2 = blk[:, 2]
        
        d2_safe = np.maximum(d2, eps)
        r = d1 / d2_safe
        
        mask = np.isfinite(d1) & np.isfinite(r)
        if np.any(mask):
            d1_list.append(d1[mask])
            r_list.append(r[mask])
    
    if d1_list:
        d1_all = np.concatenate(d1_list)
        r_all = np.concatenate(r_list)
        fifth_perc = np.percentile(d1_all, 5)
        nn_fifth_perc = np.percentile(r_all, 5)
        end_df = pd.Series([fifth_perc, nn_fifth_perc], index=['DCR','NNDR'])
    else:
        end_df = pd.Series([np.nan, np.nan], index=['DCR','NNDR'])
    
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