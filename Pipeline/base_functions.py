"""File contains function used to change the data by other functions"""

import pandas as pd
from shapely import wkt


def data_to_cat(df):
    """
    Function returns the input data frame with all object collumns as categorical str numbers.
    """

    df_cat = df.copy()
    
    cat_columns = list(df_cat.select_dtypes(['object']).columns)
    cat_dict = {}
    for c in cat_columns:
        df_cat[c] = df_cat[c].astype('category')     
        cat_dict[c] = dict(enumerate(df_cat[c].cat.categories))
    df_cat[cat_columns] = df_cat[cat_columns].apply(lambda x: x.cat.codes) 
    
    return df_cat, cat_dict

def comb_and_split(df1, df2, df3):
    """
    Function for combining the dataframes, applying the data_to_cat function to ensure its applied the same.
    """
    df1_nr, df2_nr = len(df1), len(df2)
    column_vals = df1.columns
    df = pd.concat([df1, df2, df3])
    
    df = data_to_cat(df)[0]
    df = df.fillna(0)
    
    # Splitting the dataframe again
    df1_num = pd.DataFrame(df[:df1_nr], columns=column_vals)
    df2_num = pd.DataFrame(df[df1_nr:df1_nr+df2_nr], columns=column_vals)
    df3_num = pd.DataFrame(df[df1_nr+df2_nr:], columns=column_vals)
    
    return df1_num, df2_num, df3_num
