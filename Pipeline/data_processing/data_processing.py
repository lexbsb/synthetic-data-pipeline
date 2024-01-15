"""Module contains function used for the processing of data before calculations are made"""

import glob
import pandas as pd

def data_loader(train_loc, test_loc, synth_map_loc):
    """
    Function loads the data
    train_loc: The path to the train csv file
    test_loc: The path to the test/holdout csv file
    synth_map_loc: The path the the folder containing the synthetic dataframes
    
    Returns: the real dataframe and a dict with the location of the holdout and synthetic dataframes
    """
    
    df_real = pd.read_csv(train_loc)
    glob_dict = {'holdout':test_loc}
    globlist = glob.glob(synth_map_loc + "*.csv")
    for item in globlist:
        name = item.replace(synth_map_loc, '')
        name = name.replace('.csv', '')
        glob_dict[name] = item
    print('Processed: ',glob_dict.keys())
        
    if len(glob_dict) <= 1:
        raise Exception("No csvs detected at the given synth location")
    
    return df_real, glob_dict

def ratio_calc(results):
    """
    Function calculates the ratio of the results by dividing by the holdout
    """
    ratio_results = {}
    for col in results.columns:
        if col != 'holdout':
            ratio_results[col] = round(results[col] / results['holdout'], 2)
    return pd.DataFrame(ratio_results)

