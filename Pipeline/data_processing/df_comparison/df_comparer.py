"""
Module contains functions that directly compare two dataframes to each other.
Each function takes two dataframes + some additional parameters

"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from itertools import combinations, product

from Pipeline.base_functions import data_to_cat

def subset_dupes(df1, df2, columns):
    """
    This function checks if a certain subset of the data has duplicates
    df1, df2: 2 dataframes of similiar shape.
    columns: The columns of which a subset will be created.
    """

    nr1 = df1.duplicated(keep='first', subset=columns).sum()   
    df1 = df1.drop_duplicates(keep='first', subset=columns)
    #print('Dropping ', nr1, ' duplicate rows from the real data') 

    nr2 = df2.duplicated(keep='first', subset=columns).sum()   
    df2 = df2.drop_duplicates(keep='first', subset=columns)
    #print('Dropping ', nr2, ' duplicate rows from the synthetic data') 


    df = pd.concat([df1, df2])
    if columns != None:
        values = df[df.duplicated(keep=False, subset=columns)].sort_values(columns)
    else:
        values = df[df.duplicated(keep=False, subset=columns)].sort_values(by=[col for col in df.columns])

    amount = df.duplicated(keep='first', subset=columns).sum()
    #print('There are ', amount, 'duplicate rows between the two datasets')

    return values, amount


def comparison_table(df1, df2, cutoff=-1):
    """
    Function returns a table showing the mean, std and differences between two datasets with the same columns
    Function also returns the sum of the % differences
    df1, df2: 2 dataframes of similiar shape
    cutoff: only uses columns above the cutoff
    outliers: wether the data has outliers, if true the median is used for the sum difference instead of the mean
    
    returns a comparison table and the sum difference of the mean/median
    """

    compare_table = pd.DataFrame()
    val_count_table = {}        

    
    for col in df1.columns:
        if df1[col].nunique() == 2:
            val_count1 = list(df1[col].value_counts(normalize=True))[0]
            val_count2 = list(df2[col].value_counts(normalize=True))[0]
            val_count3 = round(abs(val_count1 - val_count2) * 100, 1)
            val_count_table[col] = val_count3
            df1, df2 = df1.drop(columns=col), df2.drop(columns=col)

            
    df1, df2 = df1.select_dtypes(['number']), df2.select_dtypes(['number'])
    #df1, df2 = np.log(df1), np.log(df2)

    df1_mean, df2_mean = df1.mean(), df2.mean()
    df1_median, df2_median = df1.median(), df2.median()
    df1_std, df2_std = df1.std(), df2.std()

    compare_table['Difference Mean'] = round(df1_mean - df2_mean, 2)
    compare_table['Difference Median'] = round(df1_median - df2_median, 2)
    compare_table['Difference Std'] = round(df1_std - df2_std, 2)

    compare_table['% diff Mean'] = round((abs(compare_table['Difference Mean']) / df1_mean) * 100, 1).replace(np.inf, 100)
    compare_table['% diff Median'] = round((abs(compare_table['Difference Median']) / df1_median) * 100, 1).replace(np.inf, 100)
    compare_table['% diff STD'] = round((abs(compare_table['Difference Std']) / df1_std) * 100, 1).replace(np.inf, 100)

    compare_table = compare_table[compare_table['% diff Mean'] > cutoff]
    comp_table2 = pd.DataFrame(val_count_table, index=['val_count_diff']).T
    comp_table2 = comp_table2[comp_table2['val_count_diff'] > cutoff]

    sum_mean_per_diff = round(compare_table['% diff Mean'].sum(), 2)
    sum_median_per_diff = round(compare_table['% diff Median'].sum(), 2)
    sum_std_per_diff = round(compare_table['% diff STD'].sum(), 2)
    sum_vc_per_diff = round(comp_table2['val_count_diff'].sum(), 2)
    
    return compare_table, [sum_mean_per_diff, sum_median_per_diff, sum_std_per_diff, sum_vc_per_diff], comp_table2


def correlation_comparison(df1, df2, cutoff=0):
    """
    Function shows the correlation and difference in correlation between two datasets with the same shape
    df1, df2: 2 dataframes of the same shape
    cutoff: The % of difference below which columns are excluded.

    Returns: a list with the highest differences and the norm of the difference matrix
    """
    # Divide the cutoff to get the actual value.
    cutoff = cutoff / 100

    columns_list = []

    # Making both the datasets categorical and obtaining their pearson correlation values
    df1 = data_to_cat(df1)[0].corr().fillna(0)
    df2 = data_to_cat(df2)[0].corr().fillna(0)

    # Calculating the absolute difference between the two correlations
    difference = abs(df1 - df2)
    # Calculating the norm
    norm = np.linalg.norm(df1 - df2)
    # Because most dataset have to many columns, they get filtered based on a cutoff.
    diffy = pd.DataFrame(difference.unstack().sort_values(ascending=False), columns=['Difference']).drop_duplicates(keep='first')
    diffy = diffy[diffy['Difference'] > cutoff]

    df11 = pd.DataFrame(df1.unstack().sort_values(ascending=False), columns=['Real corr'])
    df22 = pd.DataFrame(df2.unstack().sort_values(ascending=False), columns=['Synth corr'])

    diffy = diffy.merge(df11, how='inner', right_index=True, left_index=True)
    diffy = diffy.merge(df22, how='inner', right_index=True, left_index=True)
    diffy['Difference'] = round(diffy['Difference'], 4)

    return diffy, norm


def stat_sim(df1, df2, c=1):
    """
    Function calculates the jensenshannon, wasserstein distance and total variational difference between
    each value_counts() of a column between two dataframes.
    
    df_real: the trainings data used to train the synthetic data
    df_synth: The dataset the trainings data is compared against.
    
    c: the amount of column combinations that is being looked at. 
    c=1 has all columns, c=2 shows all bivariate combinations, c=3 shows all three-way combinations   
    
    N = number of columns
    c=1 == N
    c=2 == (N * (N - 1)) / 2
    c=3 == (N * (N - 1) * (N - 2)) / 6 
    
    Function follows the binominal coefficient formula.
 
    """ 

    df1, df2 = df1.copy(), df2.copy()   
    df_pairs, results = [], {}
    
    iter_cols = product(df1.columns, repeat=c)

    for pair in iter_cols:
        df_pairs.append(pair)

    combs = pd.DataFrame(df_pairs)

    if c == 1:
        combs = combs
    if c == 2:
        combs = combs.loc[(combs[0] < combs[1])]
    if c == 3: 
        combs = combs.loc[(combs[0] < combs[1]) & (combs[1] < combs[2])]
    
    for column in combs.values:

        real_pdf = df1[column].value_counts(normalize=True).sort_index().to_frame(name='df1')
        synth_pdf = df2[column].value_counts(normalize=True).sort_index().to_frame(name='df2')
        
        # Ensuring the two dataframes have the same shape
        joined = real_pdf.join(synth_pdf, how='outer').fillna(0.0)
        l1 = joined['df1']
        l2 = joined['df2']  
        
        jsd = jensenshannon(l1, l2, 2.0)
        tvd = np.sum(np.abs(l1 - l2)) / 2
        try:
            wsd = wasserstein_distance(l1, l2)
        except:
            wsd = 0
        results[str(column)] = [jsd, tvd, wsd]
        
    return pd.DataFrame(results, index=['jsd','tvd','wsd']).T
