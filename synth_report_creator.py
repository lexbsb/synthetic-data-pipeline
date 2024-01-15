""" Script used to generate all the results with command line arguments."""

import argparse as ap
import time
from Pipeline.report_writer import report_writer_func
from Pipeline.df_compare import df_compare
from Pipeline.privacy_functions import privacy_calc
from Pipeline.final_score import final_scoring

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":
    
    
    argparser = ap.ArgumentParser(description="Script that gives evaluation scores of the synthetic data")
    argparser.add_argument("-real", action="store", dest="real", required=True, type=str,
                           help="path to the trainings data")
    argparser.add_argument("-hold", action="store", dest="hold", required=True, type=str,
                           help="path to the holdout data")
    argparser.add_argument("-synth", action="store", dest="synth", required=True, type=str,
                           help="path to the maplocation of synthetic data")
    argparser.add_argument("-output", action="store", dest="output", required=True, type=str,
                           help="path to where the output csv file is created")
    argparser.add_argument("-y_cols", action="store", dest="y_cols", nargs='*', required=False, type=str, default=None,
                           help="A set of columns for which ml utility will be calculated")
    argparser.add_argument("-comb", action="store", dest="comb", required=False, type=int, default=1,
                           help="Amount of columns that get cross selected for the fidelity check")
    argparser.add_argument("-sample", action="store", dest="sample_per", required=False, type=int, default=100,
                           help="% of data thats get used for the privacy evaluation")
    argparser.add_argument("-memory", action="store", dest="memory", required=False, type=int, default=300,
                           help="The amount of memory used for each chunk in the privacy function")
    argparser.add_argument("-subset", action="store", dest="subset", nargs='*', required=False, type=str, default=None,
                           help="A subset of columns that duplicates are calculated for")
    
    start_time = time.time()
    args = argparser.parse_args()
    real, holdout, synth, output = args.real, args.hold, args.synth, args.output, 
    y_columns, comb, sample_per, memory, subset = args.y_cols, args.comb, args.sample_per, args.memory, args.subset
    
    # Calculating statistics and their ratio
    end_results, ratio_results, reggre, classi = df_compare(real, holdout, synth, comb, y_columns, subset)
    
    if sample_per != 0:
        privacy_results, privacy_ratio = privacy_calc(real, holdout, synth, sample_per, memory)
    else:
        privacy_results, privacy_ratio = 0,0
    
    # Calculating a final score based on all the evaluations.
    end_score = final_scoring(ratio_results, privacy_ratio, reggre, classi)[0]
    
    # Writes all the results to a csv specified with output
    report_writer_func(end_results, ratio_results, reggre, classi, privacy_results, privacy_ratio, output, end_score)
    
    print_time = round(((time.time() - start_time) / 60), 2)
    print(
        '\n',
        '########################################################################################################',
        '\n', 
        'Evaluation done in:    ', print_time, 'minutes',
        '\n',
        '########################################################################################################'
    )
    
    
    
        

        
