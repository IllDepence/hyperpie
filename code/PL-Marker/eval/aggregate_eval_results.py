""" Aggregate evaluation results of n-fold cross-validation. """

import json
import os
import statistics
import sys
# import matplotlib.pyplot as plt


def aggregate_predictions(root_dir):
    """ Merge NER and RE predictions.
    """

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        print(f'Processing {subdir_path}')
        # lpad data
        ner_fp = os.path.join(subdir_path, 'ner', 'ent_pred_test.json')
        re_fp = os.path.join(subdir_path, 're', 'pred_results.json')
        smpls = []
        with open(ner_fp, 'r') as f:
            for line in f:
                smpls.append(json.loads(line))
        with open(re_fp, 'r') as f:
            rel_preds = json.load(f)
        # merge in RE predictions
        for smpl_idx, smpl in enumerate(smpls):
            # create empty list of predictions
            smpl['predicted_relations'] = []
            for r in smpl['relations']:
                smpl['predicted_relations'].append([])  # one per sentence
            if str(smpl_idx) not in rel_preds:
                # no relations predicted, leave predictions empty
                continue
            rel_preds_doc = rel_preds[str(smpl_idx)]
            # iterate over predictions
            for (sent_idx, rels_raw) in rel_preds_doc:
                rels = []
                for (from_span, to_span, rel_type) in rels_raw:
                    rels.append([
                        from_span[0], from_span[1],
                        to_span[0], to_span[1],
                        rel_type
                    ])
                smpl['predicted_relations'][sent_idx] = rels

        out_fp = os.path.join(subdir_path, 'merged_preds.jsonl')
        with open(out_fp, 'w') as f:
            for smpl in smpls:
                json.dump(smpl, f)
                f.write('\n')


def aggregate_numbers(root_dir):
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
    # aggregate_numbers(root_dir)
    aggregate_predictions(root_dir)
