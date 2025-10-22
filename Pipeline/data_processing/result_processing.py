"""This module contains functions used to calculate results"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .df_comparison.df_comparer import *
from .df_comparison.real_or_synth import real_synth
from .df_comparison.predict_metrics import metric

from Pipeline.data_processing.utils import align_columns_to_reference, common_columns


def result_combiner(df, frame_name, df_real, c, subset_cols):
    """
    Combina métricas de fidelidade em um único dict.
    """

    # 1) Aligns names between synth and real datasets
    df = align_columns_to_reference(df, df_real)

    # 2) Defines columns to be used
    if subset_cols is not None:
        # normalize column names
        norm_ref = {_norm_name(c): c for c in df_real.columns}
        resolved_subset = []
        for col in subset_cols:
            nc = _norm_name(col)
            if nc in norm_ref:
                resolved_subset.append(norm_ref[nc])
        cols = [c for c in resolved_subset if c in df_real.columns and c in df.columns]
    else:
        cols = common_columns(df_real, df)

    if not cols:
        raise ValueError(
            "[result_combiner] Nenhuma coluna em comum após alinhamento. "
            f"Real={len(df_real.columns)} Synth={len(df.columns)}"
        )

    # 3) orders and restrics to the same columns
    df_real = df_real[cols]
    df = df[cols]

    # 4) cleaning
    df = df.replace(99999.0, np.nan)

    # 5) Existing metrics
    dupe_numbers = subset_dupes(df_real, df, columns=cols)[1]
    end_result = {'dupe_numbers': dupe_numbers}

    stat_numbers = comparison_table(df_real, df, cutoff=0)[1]
    end_result['sum_%_mean_diff'] = stat_numbers[0]
    end_result['sum_%_median_diff'] = stat_numbers[1]
    end_result['sum_%_std_diff'] = stat_numbers[2]
    end_result['binary_val_count_diff'] = stat_numbers[3]

    corr_norm = round(correlation_comparison(df_real, df, cutoff=30)[1], 2)
    end_result['correlation_norm'] = corr_norm

    synth_real = (
        real_synth(df1=df_real, df2=df, name=frame_name, fill_na_val=-9999)[0]
        .loc['accuracy']
        .values[0]
    )
    end_result['real_or_synth_acc'] = synth_real

    jsws = round(stat_sim(df_real, df, c=c).mean(), 5)
    end_result['jenson_shannon'] = jsws[0]
    end_result['total_variational_dist'] = jsws[1]
    end_result['wasserstein_dist'] = jsws[2]

    return end_result


def data_processer(df1, df2, y_column, fill_na_val=-9999, name='real', plot=False):
    """
    Function transforms and scales data to train a model to predict on a single column. 
    The prediction is then evaluated.
    Returns the prediction and evaluation and a plot of the feature importances.

    Input:
    df1: The dataset used to train the model
    df2: A holdout dataset split from the real dataset.
    y_column: The column that is being predicted on.
    fill_na_val: The value whith which nans are filled.
    """
    train, test = df1, df2
    
    
    # Obtaining the column data
    columns_vals = train.columns
    columns_no_y = columns_vals[columns_vals != y_column]     
    
    # Dropping the row if the y columns is a NAN
    train = train.dropna(axis=0, subset=[y_column]) 
    test = test.dropna(axis=0, subset=[y_column]) 
 
    # Combing the dataframe into 1 and making all object values categorical
    nr_train, nr_test = train.shape[0], test.shape[0] 
    df = pd.concat([train, test]) 
    df = data_to_cat(df)[0]
    df = df.fillna(fill_na_val)
    
    # Splitting the dataframe into 2 again
    train = pd.DataFrame(df[:nr_train], columns=columns_vals)
    test  = pd.DataFrame(df[nr_train:], columns=columns_vals)
    
    # Splitting further on x features
    X_train = train.loc[:, train.columns != y_column]
    X_test  = test.loc[:, test.columns !=   y_column]
    
    # Scaling the data
    xScaler = StandardScaler()
    xScaler.fit(X_train)
    X_train = xScaler.transform(X_train)
    X_test = xScaler.transform(X_test)
    
    # Obtaining the y columns
    y_train = train[y_column]
    y_test  =  test[y_column].values.reshape(-1, 1)
    
    # Predicting the values
    metric_train, feat_imp_plot_train = metric(X_train,
                                               X_test,
                                               y_train,
                                               y_test,
                                               real_synth=[name],
                                               columns=columns_no_y,
                                               plot=plot)
    
    return metric_train, feat_imp_plot_train