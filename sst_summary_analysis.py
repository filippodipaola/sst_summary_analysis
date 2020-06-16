import os
from collections import defaultdict
import argparse

import pandas as pd
import numpy as np

def load_csv_file(path):
    return pd.read_csv(path)[:196]

def get_csv_file_paths(parent_dir):
    return [os.path.join(parent_dir, csv_file) for csv_file in
            os.listdir(parent_dir) if csv_file.endswith(".csv")]

def calculate_outcomes(df):
    columns = ["SSD", "RTgo", "Rtgo_noposterrorRT", "Post-error RT",
               "Omission errors", "Premature errors", "Go errors",
               "RT_post_corr", "Rtgo_om_trial",
               "Rtgo_prem_trial", "RTgo_L_R_errors", "RTgo+L_R_error",
               "Rtgo_om_L_R", "RT_all", "STOPfail_RT"]

    outcomes = pd.DataFrame(index=list(range(len(df.index))), columns=columns)

    #print(outcomes)
    for index, row in df.iterrows():
        # Calculate SSD if exist
        if df['StopSignal'][index] in 'Yes':
            outcomes.loc[index, 'SSD'] = df['StimDuration'][index]

        # Calculate RTGO if exists
        if (df['StopSignal'][index] == "No" and df['IsCorrect'][
            index] == "Yes" and int(df['RT'][index]) > 0):
            outcomes.loc[index, 'RTgo'] = int(df['RT'][index])

        # Calculate Post-Error RT before Rtgo_noposterrorRT
        if (index is not 0 and df['StopSignal'][index - 1] in 'Yes' and
                df['IsCorrect'][index - 1] == "No" and int(
                    df['RT'][index - 1]) > 0) and df['RT'][index] != '.':
            outcomes.loc[index, 'Post-error RT'] = int(df['RT'][index])
            #print(df['RT'][index])

        if outcomes['RTgo'][index] and outcomes["Post-error RT"][
                index] is np.nan:
            outcomes.loc[index, 'Rtgo_noposterrorRT'] = outcomes['RTgo'][index]
            #print(outcomes['RTgo'][index])

        if df['StopSignal'][index] == 'No' and df['RT'][index] == '.':
            outcomes.loc[index, 'Omission errors'] = 1

        if df['RT'][index] is not np.nan and \
            df['RT'][index] is not '.' and int(df['RT'][index]) == 0: # TODO Change the RT to less than 100
            outcomes.loc[index, 'Premature errors'] = 1

        if df['Response'][index] != "None" and df['Stim'][index] != df['Response'][
            index] and df['StopSignal'][index] == "No":
            outcomes.loc[index, 'Go errors'] = 1

        if index is not 0 and df['StopSignal'][index-1] == 'Yes' and \
                df['IsCorrect'][index-1] == 'Yes' and \
                outcomes['Omission errors'][index-1] is np.nan and \
                outcomes['Go errors'][index-1] is np.nan and \
                df['RT'][index] != '.':

            outcomes.loc[index, 'RT_post_corr'] = int(df['RT'][index])

        if df['StopSignal'][index] == 'No' and \
                df['IsCorrect'][index] == 'No' and df['Response'][index] == 'None':

            outcomes.loc[index, 'Rtgo_om_trial'] = 1800

        if df['StopSignal'][index] == 'No' and df['RT'][index] == '0':
            outcomes.loc[index, 'Rtgo_prem_trial'] = 0
            outcomes.loc[index, 'RT_all'] = 0

        if df['StopSignal'][index] == 'No' and df['IsCorrect'][
            index] == 'No' and df['Response'][index] != 'None' and int(df['RT'][
            index]) > 0:
            outcomes.loc[index, 'RTgo_L_R_errors'] = int(df['RT'][index])

        l_r_error = int(sum(np.nan_to_num(
            [outcomes['RTgo'][index], outcomes['RTgo_L_R_errors'][index]])))
        if l_r_error > 0:
            outcomes.loc[index, 'RTgo+L_R_error'] = l_r_error

        l_r_error = int(
            sum(np.nan_to_num([l_r_error, outcomes['Rtgo_om_trial'][index]])))

        if l_r_error > 0:
            outcomes.loc[index, 'Rtgo_om_L_R'] = l_r_error
            outcomes.loc[index, 'RT_all'] = l_r_error

        if df['StopSignal'][index] == "Yes" and df[
            'IsCorrect'][index] == "No" and int(df['RT'][index]) > 0:
            outcomes.loc[index, 'STOPfail_RT'] = int(df['RT'][index])

    return outcomes


def aggregate_outcomes(df, csv_df):
    return_dict = {}
    SSD_avrg = df['SSD'].mean()
    RT_go_avrg = df['RTgo'].mean()
    return_dict['RT_go'] = RT_go_avrg
    return_dict['SDRT_go'] = df['RTgo'].std()
    return_dict['RT_go_no_post_error'] = df['Rtgo_noposterrorRT'].mean()
    return_dict['SDRT_go_no_post_error'] = df['Rtgo_noposterrorRT'].std()
    RT_post_error_avrg = df['Post-error RT'].mean()
    return_dict['RT_post_error'] = RT_post_error_avrg
    RT_post_correct = df['RT_post_corr'].mean()
    return_dict['RT_post_correct'] = RT_post_correct
    return_dict['Post_error_slowing'] = RT_post_error_avrg - RT_post_correct
    return_dict['Go_ommision'] = df['Omission errors'].sum()
    return_dict['Go_premature'] = df['Premature errors'].sum()
    return_dict['L_R_errors'] = df['Go errors'].sum()
    PI_prob_inh = int(
        [a for a in csv_df['PercentCorrect'].tolist() if a != '.'][
            -1])
    return_dict['PI_prob_inh'] = PI_prob_inh
    PI_prob_resp = 100 - PI_prob_inh
    return_dict['PI_prob_resp'] = PI_prob_resp
    return_dict['SSRT_mean'] = RT_go_avrg - SSD_avrg
    count_go_trails = csv_df['StopSignal'].tolist().count("No")
    nth_RT_rank = round(count_go_trails * (PI_prob_resp / 100))
    df['RT_all'].dropna(inplace=True)
    ranks = df['RT_all'].sort_values().tolist()
    nth_RT = ranks[nth_RT_rank - 1]
    return_dict['SSRT_intergration'] = nth_RT - SSD_avrg
    return return_dict

def get_participant_name(file_name):
    spl_file_name = file_name.split("_")
    stoptask_index = spl_file_name.index("StopTask")
    participant = spl_file_name[stoptask_index + 1]
    if file_name.startswith("RTAD"):
        participant = f"RTAD{participant}"
    return participant





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generates SST summary performance values"
                                     " from participant csv files retreived "
                                     "from MR scanner.")
    parser.add_argument("directory", metavar='directory',
                        help="The directory containing the stop tasks.")
    parser.add_argument("-o", "--output_dir",
                        help="The directory (MUST EXIST) to output the task times.",
                        default="")
    parser.add_argument("-i", "--intermediate", action='store_const', const=True,
                        help="Use this option if you want to generate the "
                             "intermediate csv files used to calculate the "
                             "summary performance.")

    args = parser.parse_args()

    DATA_DIR = args.directory
    RESULTS_DIR = args.output_dir

    out_dir = None
    if args.output_dir:
        out_dir = args.output_dir

    csv_files = get_csv_file_paths(DATA_DIR)
    dv = defaultdict(list)
    for csv_file in csv_files:
        # Load CSV from data directory
        pd_csv = load_csv_file(csv_file)
        print(f"PROCESSING CSV FILE: {csv_file}")

        # Calculate green and blue parts of the csv
        df = calculate_outcomes(pd_csv)

        # Create and write the blue and green parts to a new csv file.
        file_name = os.path.split(csv_file)[-1]
        out_file = os.path.join(RESULTS_DIR, f"outcomes_{file_name}")
        if args.intermediate:
            print(f"OUTPUTTING INTERMEDIATE CSV FILE: {out_file}")
            df.to_csv(out_file, index=False)

        dv['Participant'].append(get_participant_name(file_name))
        [dv[k].append(v) for k, v in aggregate_outcomes(df, pd_csv).items()]

    results_file = os.path.join(RESULTS_DIR, "SST_summary_performance.csv")
    print(f"PROCESSING COMPLETE. RESULTS FILE GENERATED: {results_file}")
    pd.DataFrame.from_dict(data=dv).to_csv(results_file, index=False)

