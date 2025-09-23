"""Module calculates the statistics and their ratio for the given files"""

import time
import pandas as pd
import numpy as np

from .base_functions import *
from .data_processing.data_processing import data_loader, ratio_calc
from .data_processing.result_processing import data_processer, result_combiner
from .data_processing.utils import align_columns_to_reference, _norm_name



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
    Evaluate all synthetic dataframes against the real and holdout datasets.

    Parameters
    ----------
    train_loc : str or Path
        Path to the CSV file used as the real training data.
    test_loc : str or Path
        Path to the holdout/test CSV file.
    synth_map_loc : str or Path
        Folder that contains the synthetic CSV files.
    c : int
        Interaction order for distributional similarity metrics (e.g., JSD, TVD, WSD).
    y_columns : list[str], optional
        Target columns to use for predictive utility (regression/classification).
    subset : list[str], optional
        Columns to use when computing duplicate counts.

    Returns
    -------
    results : pd.DataFrame
        Fidelity metrics per frame.
    ratio_results : pd.DataFrame
        Fidelity metrics divided by the holdout frame values.
    reggre : pd.DataFrame
        Aggregated regression utility metrics across frames.
    classi : pd.DataFrame
        Aggregated classification utility metrics across frames.
    """

    # Load real data and a dict of frame -> path (includes 'holdout')
    df_real, synth_dict = data_loader(train_loc, test_loc, synth_map_loc)

    # Read and align holdout to the real schema
    df_holdout = pd.read_csv(test_loc)
    df_holdout = align_columns_to_reference(df_holdout, df_real)

    # Normalize y_columns to the exact column names in df_real
    if y_columns is not None:
        norm_ref = {_norm_name(col): col for col in df_real.columns}
        y_columns = [norm_ref[_norm_name(y)] for y in y_columns if _norm_name(y) in norm_ref]
        if not y_columns:
            raise ValueError(
                "None of the provided y_columns match the real dataframe schema after normalization."
            )

    results = {}
    reggre = {}
    classi = {}

    # Iterate through each frame found by data_loader (holdout + synthetics)
    for frame_name, path in synth_dict.items():
        frame_start = time.time()
        print(f"Starting calculations for: {frame_name}")

        # Read and align current frame to real schema
        if frame_name == "holdout":
            df = df_holdout.copy()
        else:
            df = pd.read_csv(path)
            df = align_columns_to_reference(df, df_real)

        # Replace known sentinel values from synthpop if present
        df = df.replace(99999.0, np.nan)

        # Fidelity metrics (duplicates, summary diffs, correlations, stat distances, etc.)
        results[frame_name] = result_combiner(
            df=df,
            frame_name=frame_name,
            df_real=df_real,
            c=c,
            subset_cols=subset,
        )

        # Predictive utility metrics (regression/classification) per target y
        if y_columns is not None:
            mc_utils = []
            for y in y_columns:
                # Safety: ensure target exists in both frames at this point
                if y not in df.columns:
                    raise KeyError(
                        f"Target column '{y}' not found in frame '{frame_name}' after alignment."
                    )
                if y not in df_holdout.columns:
                    raise KeyError(
                        f"Target column '{y}' not found in aligned holdout dataframe."
                    )

                if frame_name == "holdout":
                    # Compare real vs holdout for baseline
                    mc_utils.append(
                        data_processer(
                            df1=df_real, df2=df_holdout, y_column=y, fill_na_val=-9999, name=y
                        )[0]
                    )
                else:
                    # Compare synthetic vs holdout
                    mc_utils.append(
                        data_processer(
                            df1=df, df2=df_holdout, y_column=y, fill_na_val=-9999, name=y
                        )[0]
                    )

            # Combine per-target utility metrics for this frame, then summarize
            res_df = pd.concat(mc_utils, axis=1)
            reggre[frame_name], classi[frame_name] = mc_utility_process(res_df.T)
        else:
            # If no targets provided, keep placeholders for later concatenation
            reggre[frame_name] = pd.DataFrame()
            classi[frame_name] = pd.DataFrame()

        elapsed_min = round((time.time() - frame_start) / 60.0, 2)
        print(f"Finished {frame_name} in {elapsed_min} minutes")

    # Assemble fidelity results and ratios
    results = pd.DataFrame(results)
    ratio_results = ratio_calc(results)

    # Assemble predictive utility outputs
    # Only concat non-empty frames to avoid all-zero checks failing on empty frames
    reggre_nonempty = {k: v for k, v in reggre.items() if not v.empty}
    classi_nonempty = {k: v for k, v in classi.items() if not v.empty}

    reggre = (
        pd.concat(reggre_nonempty, names=['frame'])
        if reggre_nonempty else pd.DataFrame()
    )
    classi = (
        pd.concat(classi_nonempty, names=['frame'])
        if classi_nonempty else pd.DataFrame()
    )

    if reggre.empty or reggre.sum().sum() == 0:
        print("No regression columns.")
    if classi.empty or classi.sum().sum() == 0:
        print("No classification columns.")

    return results, ratio_results, reggre, classi