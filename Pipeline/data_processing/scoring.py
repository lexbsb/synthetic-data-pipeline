"""Functions assisiting with the final scoring calulations."""


import pandas as pd


def privacy_scoring(privacy_ratio):
    
    """
    Calculates the final score of the privacy evaluations
    Only to be used within the final scoring function
    """

    real_col_val, end_col_val  = {}, {}
    #priv_score = abs(1 - privacy_ratio)
    priv_score = 1 - privacy_ratio
   # return priv_score

    for col1 in priv_score.columns:
        if 'real_' in col1:
            col1_v2 = col1.replace('real_','')
            real_col_val[col1_v2] = priv_score[col1]
    #return real_col_val, priv_score        
            
    for col1 in real_col_val:  
        if 'real' not in col1:
            end_col_val[col1] = real_col_val[col1] + priv_score[col1]
            
    return pd.DataFrame(end_col_val)



def ml_util_scoring(reggre, classi, ratio_results):  
    
    """
    Calculates the final score of the utility evaluations
    Only to be used within the final scoring function.
    """
    
    ratio_utl_reg, ratio_utl_clas, res_end, clas_end = {}, {}, {}, {}
    
    for name in ratio_results:
        if name != 'holdout':
            ratio_utl_reg[name] = reggre.loc[name] / reggre.loc['holdout']
            ratio_utl_clas[name] = classi.loc[name] / classi.loc['holdout']    
            
    rug = pd.concat(ratio_utl_reg)
    ruc = pd.concat(ratio_utl_clas)

    clas_res = abs(1 - ruc.T)
    rug_res = abs(1 - rug.T)
    
    for col in ratio_results:
        res_end[col] = rug_res[col].T.sum()
        clas_end[col] = clas_res[col].T.sum()

    see = pd.DataFrame(res_end).sum()
    see = pd.DataFrame(see, columns=['regression']).T

    see2 = pd.DataFrame(clas_end).sum()
    see2 = pd.DataFrame(see2, columns=['classification']).T

    ml_util_end = pd.concat([see, see2])
    
    return ml_util_end