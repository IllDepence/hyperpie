""" Aggregate evaluation results of n-fold cross-validation. """

import json
import os
import statistics
import sys
# import matplotlib.pyplot as plt


def aggregate(root_dir):
    """ Aggregate evaluation results of n-fold cross-validation.
    """

    # Iterate over all subdirectories of root_dir
    # - each subdirectory contains a "ner" and a "re" directory
    #   (for named entity recognition and relation extraction respectively)
    # - each ner/re directory contains a results.json
    # - for ner/re, aggregate precision, recall, f1 and print to stdout
    #   with standard deviation

    ner_precisions = []
    ner_recalls = []
    ner_f1s = []
    re_precisions = []
    re_recalls = []
    re_f1s = []

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        print(f'Processing {subdir_path}')
        ner_fp = os.path.join(subdir_path, 'ner', 'results.json')
        re_fp = os.path.join(subdir_path, 're', 'results.json')
        skip = False
        for fp in [ner_fp, re_fp]:
            if not os.path.isfile(fp):
                print(f'No results.json found in {fp}')
                skip = True
        if skip:
            continue
        with open(ner_fp, 'r') as f:
            ner_results = json.load(f)
            ner_precisions.append(ner_results['precision_'])
            ner_recalls.append(ner_results['recall_'])
            ner_f1s.append(ner_results['f1_'])
        with open(re_fp, 'r') as f:
            re_results = json.load(f)
            re_precisions.append(re_results['prec_w_ner_'])
            re_recalls.append(re_results['rec_w_ner_'])
            re_f1s.append(re_results['f1_with_ner_'])

    # Calculate mean and standard deviation in percentages
    ner_p_mean = statistics.mean(ner_precisions) * 100
    ner_p_std = statistics.stdev(ner_precisions) * 100
    ner_r_mean = statistics.mean(ner_recalls) * 100
    ner_r_std = statistics.stdev(ner_recalls) * 100
    ner_f1_mean = statistics.mean(ner_f1s) * 100
    ner_f1_std = statistics.stdev(ner_f1s) * 100
    re_p_mean = statistics.mean(re_precisions) * 100
    re_p_std = statistics.stdev(re_precisions) * 100
    re_r_mean = statistics.mean(re_recalls) * 100
    re_r_std = statistics.stdev(re_recalls) * 100
    re_f1_mean = statistics.mean(re_f1s) * 100
    re_f1_std = statistics.stdev(re_f1s) * 100

    print(
        f'NER precision: {ner_p_mean:.1f} ± {ner_p_std:.1f}\n'
        f'NER recall: {ner_r_mean:.1f} ± {ner_r_std:.1f}\n'
        f'NER f1: {ner_f1_mean:.1f} ± {ner_f1_std:.1f}\n'
        f'RE precision: {re_p_mean:.1f} ± {re_p_std:.1f}\n'
        f'RE recall: {re_r_mean:.1f} ± {re_r_std:.1f}\n'
        f'RE f1: {re_f1_mean:.1f} ± {re_f1_std:.1f}\n'
    )


if __name__ == '__main__':
    root_dir = sys.argv[1]
    aggregate(root_dir)
