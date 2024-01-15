""""File contains the function used to generate the privacy evaluations."""


from .data_processing.data_processing import data_loader
from .base_functions import comb_and_split
from .privacy_eval.privacy_eval import *

def privacy_calc(train_loc, test_loc, synth_map_loc, sample_per, memory):
    
    """
    Function used to calculate the privacy scores of the synthetic data.
    
    train_loc: The location of the csv file used to generate the synthetic data
    test_loc: The location of the holdout csv file.
    synth_map_loc: The folder containing all the synthetic data files
    sample_per: The precentage size of the dataset used.
    memory: The amount of memory the function uses to process a single chunk.
    
    Returns the privacy scores and the privacy ratio scores.
    """
    
    df_real, synth_dict = data_loader(train_loc, test_loc, synth_map_loc)
    df_holdout = pd.read_csv(test_loc)
   
    # Calculating the privacy score
    df_real_num = comb_and_split(df_real, df_holdout, df_holdout)[0]
    df_real_num = data_process_priv(df_real_num, sample_per)
    real_priv = single_priv_calc(df_real_num, memory)
    
    synth_result = []
    
    for frame_name in synth_dict:
        df = pd.read_csv(synth_dict[frame_name])
        df_num = comb_and_split(df_real, df_holdout, df)[2]
        df_num = data_process_priv(df_num, sample_per)
        synth_result.append(privacy_synth_frame(df_real_num, df_num, frame_name, memory=memory))

    end_df = pd.concat([pd.DataFrame(real_priv, columns=['real']), pd.concat(synth_result, axis=1)], axis=1)
    end_df = round(end_df, 3)
    priv_ratio = privacy_ratio(end_df)
    
    return end_df, priv_ratio
    
def privacy_calc_id(train_loc, test_loc, synth_map_loc, id_columns, sample_per, memory):
    
    """
    Function used to calculate the privacy scores of the synthetic data.
    
    train_loc: The location of the csv file used to generate the synthetic data
    test_loc: The location of the holdout csv file.
    synth_map_loc: The folder containing all the synthetic data files
    sample_per: The precentage size of the dataset used.
    memory: The amount of memory the function uses to process a single chunk.
    
    Returns the privacy scores and the privacy ratio scores.
    """
    
    df_real, synth_dict = data_loader(train_loc, test_loc, synth_map_loc)
    df_holdout = pd.read_csv(test_loc)
    
    df_real = df_real[id_columns]
    df_holdout = df_holdout[id_columns]
   
    # Calculating the privacy score
    df_real_num = comb_and_split(df_real, df_holdout, df_holdout)[0]
    df_real_num = data_process_priv(df_real_num, sample_per)
    real_priv = single_priv_calc(df_real_num, memory)
    
    synth_result = []
    
    for frame_name in synth_dict:
        df = pd.read_csv(synth_dict[frame_name])
        df = df[id_columns]
        df_num = comb_and_split(df_real, df_holdout, df)[2]
        df_num = data_process_priv(df_num, sample_per)
        synth_result.append(privacy_synth_frame(df_real_num, df_num, frame_name, memory=memory))

    end_df = pd.concat([pd.DataFrame(real_priv, columns=['real']), pd.concat(synth_result, axis=1)], axis=1)
    end_df = round(end_df, 3)
    priv_ratio = privacy_ratio(end_df)
    
    return end_df, priv_ratio