"""Module contains function that predicts real or synthetic data"""


import pandas as pd
from Pipeline.base_functions import data_to_cat
from .predict_metrics import metric


def real_synth(df1, df2, name, fill_na_val):
    """
    Function uses a simple model to predict whether the data is real or fake.
    df1, df2: 2 dataframes of the same shape
    fill_na_val: The value used to fill in the NaN values 
    """
    
    df_real = df1.copy()
    df_synth = df2.copy()
    
    # Adding a 1 or a 0 based on the data being real or synthetic
    df_real['real_or_synth'] = 1
    df_synth['real_or_synth'] = 0
    df = pd.concat([df_real, df_synth])
    
    # Transforming the data
    df = data_to_cat(df)[0]
    df = df.fillna(fill_na_val)
    
    # Splitting the data
    df_train = df.sample(frac=0.7)
    df_test = df.drop(df_train.index)
    
    # Splitting the data on x and y    
    X_train = df_train.loc[:, df_train.columns != 'real_or_synth']
    X_test = df_test.loc[:, df_test.columns != 'real_or_synth']
    y_train = df_train['real_or_synth']
    y_test = df_test['real_or_synth']
    
    
    result, plot = metric(X_train, X_test, y_train, y_test, real_synth=[name], columns=X_train.columns, plot=True)
    
    return result, plot
