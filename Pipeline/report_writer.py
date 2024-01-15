"""This module contains the function to write all the results to a csv"""

from csv import writer
import pandas as pd

def report_writer_func(end_results, ratio_results, reggre, classi, privacy_results, privacy_ratio, output, end_score):
    
    """
    Function takes the output from large_metric_function and writes it to a csv
    """
    
    result_frame = {
        'Synthetic data evaluation scores':end_results,
        'Synthetic data evaluation ratios by dividing by holdout':ratio_results,
        'Reggresion predictions' :reggre,
        'Classification predictions':classi,
        'The results of the privacy calculation':privacy_results,
        'The ratio of the privacy columns divided by holdout':privacy_ratio,
        'Final score': pd.DataFrame(end_score)
    }
    
    with open(output, 'w') as csv_file:
        writer_obj = writer(csv_file)
        writer_obj.writerow(['Report Evaluation Scores'])
        csv_file.close()

    for result in result_frame:
        with open(output, 'a') as csv_file:
            writer_obj = writer(csv_file)
            writer_obj.writerow([''])
            writer_obj.writerow([result])
            csv_file.close()
        result_frame[result].to_csv(output, mode='a', decimal=',')
        
    print('Saved everything to: ', output)