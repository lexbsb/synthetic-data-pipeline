"""Module calculates the statistics and their ratio for the given files"""

import time
import pandas as pd
import numpy as np

from .base_functions import *
from .data_processing.data_processing import data_loader, ratio_calc
from .data_processing.result_processing import data_processer, result_combiner


def mc_utility_process(mc_utils):
    
    """
    Function splits the utility output into classification and regression results.
    Function should only be used within the df_compare function.
    """
    
    try:
        reggre = mc_utils[['max error','explained variance score','r2','mean squared error']].T
        reggre = reggre.dropna(how='all', axis=1)
    except:
        reggre = pd.DataFrame({'max error':0,'explained variance score':0,'r2':0,'mean squared error':0},index=['Not used']).T
    try:
        classi = mc_utils[['accuracy','recall','precision','f1']].T
        classi = classi.dropna(how='all', axis=1)
    except: 
        classi = pd.DataFrame({'accuracy':0,'recall':0,'precision':0,'f1':0},index=['Not used']).T
                
    return reggre, classi



def df_compare(train_loc, test_loc, synth_map_loc, c, y_columns=None, subset=None):
    
    """
    Main function used to evaluate all the synthetic dataframes.
    
    train_loc: The location of the csv file used to generate the synthetic data
    test_loc: The location of the holdout csv file.
    synth_map_loc: The folder containing all the synthetic data files
    y_columns: A list of target columns on which the utility predictions are made
    subset: A list of columns for which the duplicates are calculated. 
    
    Returns:
        results: the fidelity metrics + the duplicate score. 
        ratio_results: the results divided by the holdout data.
        reggre: the regression utility metrics
        classi: the classification utility metrics
    """
    
    df_real, synth_dict = data_loader(train_loc, test_loc, synth_map_loc)
    df_holdout = pd.read_csv(test_loc)
     
    results, reggre, classi = {}, {}, {}
    res_df = pd.DataFrame()
    
    for frame_name in synth_dict:
        
        mc_utils = []
        frame_time = time.time()
        print('Starting calculations with ', frame_name)
        
        # Calculating basic statisitcs
        df = pd.read_csv(synth_dict[frame_name])
        # Replacing synthpop error values.
        df = df.replace(99999.0, np.nan)
        results[frame_name] = result_combiner(df, frame_name, df_real, c=c, subset_cols=subset)
        
        # Calculating machine learning utility
        if y_columns is not None:
            for y_column in y_columns:
                if frame_name == 'holdout':
                    mc_utils.append(data_processer(df_real, df_holdout, y_column, fill_na_val=-9999, name=y_column)[0])
                else:
                    mc_utils.append(data_processer(df, df_holdout, y_column, fill_na_val=-9999, name=y_column)[0])
                    
            res_df = pd.concat(mc_utils, axis=1) 
        
        print_time = round(((time.time() - frame_time) / 60), 2)
        print('Calculated all evaluations for', frame_name, 'in', print_time, 'minutes')
       
        reggre[frame_name], classi[frame_name] = mc_utility_process(res_df.T)
    
    results = pd.DataFrame(results)
    # Calculating the ratio by dividing by the holdout
    ratio_results = ratio_calc(results)
    
    reggre = pd.concat(reggre)
    classi = pd.concat(classi)
    if reggre.sum().sum() == 0:
        print('no regression columns')
    if classi.sum().sum() == 0:
        print('no classification columns')
        
    return results, ratio_results, reggre, classi


