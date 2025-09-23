"""Module contains function used for the processing of data before calculations are made"""

import glob
import pandas as pd
from pathlib import Path

def data_loader(train_loc, test_loc, synth_map_loc):
    """
    Loads real train df and builds a dict with the holdout and all synthetic CSVs.
    """
    # Coerce to Path objects
    train_loc = Path(train_loc)
    test_loc = Path(test_loc)
    synth_path = Path(synth_map_loc)

    # Load real data
    df_real = pd.read_csv(train_loc)

    # Initialize mapping with holdout
    glob_dict = {"holdout": str(test_loc)}

    # Find synthetic CSVs
    globlist = list(synth_path.glob("*.csv"))
    for item in globlist:
        name = item.stem            # filename without extension
        glob_dict[name] = str(item) # store as string path for compatibility

    print("Processed:", list(glob_dict.keys()))

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

