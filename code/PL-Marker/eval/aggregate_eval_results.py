""" Aggregate evaluation results of n-fold cross-validation. """

import json
import os
import statistics
import sys
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import lines as mlines
import matplotlib.pyplot as plt
import numpy as np


def error_analysis(root_dir, data_fn=None, filter_subdir=None):
    """ Analyze errors.
    """

    ner_class_true = []
    ner_class_pred = []
    tp_rel_types = defaultdict(int)
    fp_rel_types = defaultdict(int)
    fn_rel_types = defaultdict(int)
    per_fold_f1s = []

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        if filter_subdir is not None and subdir != filter_subdir:
            continue
        print(f'Processing {subdir_path}')
        # lpad data
        if data_fn is None:
            data_fn = 'merged_preds.jsonl'
        data_fp = os.path.join(subdir_path, data_fn)
        paras = []
        with open(data_fp, 'r') as f:
            for line in f:
                paras.append(json.loads(line))
        # iterate over paras
        for line_idx, para in enumerate(paras):
            para_ner_true = []
            para_ner_pred = []
            # iterate over sentences for NER analysis
            for sent_idx, sent in enumerate(para['sentences']):
                ner_true = para['ner'][sent_idx]
                ner_pred = para['predicted_ner'][sent_idx]
                # fill with “None” labels
                for word_idx in range(len(sent)):
                    para_ner_true.append('-')
                    para_ner_pred.append('-')
                # fill true labels
                for (start, end, label) in ner_true:
                    lbl_sent_start = start
                    lbl_sent_end = end+1
                    for word_idx in range(lbl_sent_start, lbl_sent_end):
                        para_ner_true[word_idx] = label
                # fill predicted labels
                for (start, end, label) in ner_pred:
                    lbl_sent_start = start
                    lbl_sent_end = end+1
                    for word_idx in range(lbl_sent_start, lbl_sent_end):
                        para_ner_pred[word_idx] = label
                ner_class_true.extend(para_ner_true)
                ner_class_pred.extend(para_ner_pred)
            # iterate over sentences for RE analysis
            # print(f'Processing {para["doc_key"]} (line {line_idx})')
            para_delta = 0
            for sent_idx, sent in enumerate(para['sentences']):
                re_true = para['relations'][sent_idx]
                re_pred = para['predicted_relations'][sent_idx]
                # true relations
                re_types = {
                    'true': re_true,
                    'pred': re_pred
                }
                if len(re_true) > 0 or len(re_pred) > 0:
                    # print('[' + '] ['.join(sent) + ']\n')
                    pass
                for typ, re in re_types.items():
                    if len(re_true) > 0 or len(re_pred) > 0:
                        # print(f'>> {typ} relations <<')
                        pass
                    for (
                        start_from, end_from, start_to, end_to, label
                    ) in re:
                        # get true NER labels
                        ner_from_true = para_ner_true[
                            start_from:end_from+1
                        ]
                        ner_to_true = para_ner_true[
                            start_to:end_to+1
                        ]
                        # get predicted NER labels
                        ner_from_pred = para_ner_pred[
                            start_from:end_from+1
                        ]
                        ner_to_pred = para_ner_pred[
                            start_to:end_to+1
                        ]
                        # generate simple relation keys
                        rel_key_true = (
                            set(ner_from_true).pop() +
                            '->' +
                            set(ner_to_true).pop()
                        )
                        rel_key_pred = (
                            set(ner_from_pred).pop() +
                            '->' +
                            set(ner_to_pred).pop()
                        )
                        if typ == 'true':
                            is_fn = [
                                start_from, end_from, start_to, end_to, label
                            ] not in re_pred
                            if is_fn:
                                fn_rel_types[rel_key_true] += 1
                            # print(
                            #     f'{ner_from_true} -- {label} -> {ner_to_true}'
                            # )
                        else:
                            is_tp = [
                                start_from, end_from, start_to, end_to, label
                            ] in re_true
                            is_fp = not is_tp
                            if is_tp:
                                tp_rel_types[rel_key_pred] += 1
                            if is_fp:
                                fp_rel_types[rel_key_pred] += 1
                            # print(rel_key_true, rel_key_pred)
                            # print(
                            #     f'{ner_from_true} ({ner_from_pred}) '
                            #     f'-- {label} -> '
                            #     f'{ner_to_true} ({ner_to_pred})'
                            # )
                        # get tokens
                        tokens_from = sent[
                            start_from-para_delta:end_from-para_delta+1
                        ]
                        tokens_to = sent[
                            start_to-para_delta:end_to-para_delta+1
                        ]
                        # print(f'{tokens_from} -- {label} -> {tokens_to}')
                        # input()
                para_delta += len(sent)
        fold_f1s = evaluate_re(
            tp_rel_types, fp_rel_types, fn_rel_types,
        )
        per_fold_f1s.append(fold_f1s)

    rel_type_keys = [
        'p->a', 'v->p', 'c->v'
    ]
    # for each type of relation
    for rel_type in rel_type_keys:
        # go through folds and gather results
        rtype_f1s = []
        for fld in per_fold_f1s:
            rtype_f1s.append(fld[rel_type])
        print(
            f'{rel_type}: {np.mean(rtype_f1s):.3f}'
        )
    return

    evaluate_re(
        tp_rel_types, fp_rel_types, fn_rel_types,
        verbose=True
    )
    return

    # print confusion matrix
    print('NER confusion matrix:')
    cm_labels = ['-', 'a', 'p', 'v', 'c']
    cm = confusion_matrix(
        ner_class_true,
        ner_class_pred,
        labels=cm_labels
    )
    print(cm)
    # plot confusion matrix with labels and legend, on a log scale
    cm_norm = confusion_matrix(
        ner_class_true,
        ner_class_pred,
        labels=cm_labels,
        normalize='true'
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=cm_labels,
    )
    disp.plot()
    plt.tight_layout()
    # # save as PDF
    # plt.savefig('ner_confusion_matrix.pdf')


def evaluate_re(
    tp_rel_types, fp_rel_types, fn_rel_types,
    verbose=False
):
    # sort relations by frequency and print results
    for rel_type, rel_dict in [
        ('TP', tp_rel_types),
        ('FP', fp_rel_types),
        ('FN', fn_rel_types),
    ]:
        if verbose:
            print(f'>> {rel_type} relations <<')
        for rel_key, rel_count in sorted(
            rel_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if verbose:
                print(f'{rel_key}: {rel_count}')

    # calculate precision, recall, and F1 per relation type
    rel_type_keys = [
        'p->a', 'v->p', 'c->v'
    ]
    f1s = {}
    for rel_type_key in rel_type_keys:
        tp = tp_rel_types[rel_type_key]
        fp = fp_rel_types[rel_type_key]
        fn = fn_rel_types[rel_type_key]
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0
        f1s[rel_type_key] = f1
        if verbose:
            print(f'{rel_type_key}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}')

    return f1s


def print_predictions(root_dir, data_fn=None):
    """ Print merged NER+RE predictions for inspection.
    """

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        print(f'Processing {subdir_path}')
        # lpad data
        if data_fn is None:
            data_fn = 'merged_preds.jsonl'
        data_fp = os.path.join(subdir_path, data_fn)
        paras = []
        with open(data_fp, 'r') as f:
            for line in f:
                paras.append(json.loads(line))
        # iterate over paras
        for para in paras:
            print(f'= = = {para["doc_key"]} = = =')
            para_delta = 0
            # iterate over sentences
            for sent_idx, sent in enumerate(para['sentences']):
                print('[' + '] ['.join(sent) + ']')
                # ner_true = para['ner'][sent_idx]
                # re_true = para['relations'][sent_idx]
                ner_pred = para['predicted_ner'][sent_idx]
                print('<<<NER>>>')
                for (start, end, label) in ner_pred:
                    # print(start-para_delta, end-para_delta+1, label)
                    print(sent[start-para_delta:end-para_delta+1], label)
                has_rels = False
                if 'predicted_relations' in para:
                    re_pred = para['predicted_relations'][sent_idx]
                    print('<<<RE>>>')
                    for (
                        start_from, end_from, start_to, end_to, label
                    ) in re_pred:
                        has_rels = True
                        print(
                            sent[start_from-para_delta:end_from-para_delta+1],
                            '--',
                            label,
                            '->',
                            sent[start_to-para_delta:end_to-para_delta+1]
                        )
                para_delta += len(sent)
                if has_rels:
                    input()


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


def aggregate_ffnn_re_numbers(root_dir, with_nota=False, avg_type=None):
    """ Aggregate evaluation results of n-fold cross-validation.
    """

    # Iterate over all subdirectories of root_dir
    # - each subdirectory contains a filef fnn_re_results.jsonl
    # - aggregate precision, recall, f1 and print to stdout
    #   with standard deviation

    re_precisions = []
    re_recalls = []
    re_f1s = []
    if with_nota:
        if avg_type is None:
            avg_type = 'macro'
        dict_key = f'{avg_type} avg'
        print('Including NOTA')
        print(f'Using {avg_type} average')
    else:
        if avg_type is not None:
            raise ValueError('avg can\'t be used if with_nota is False')
        dict_key = '1'
        print('Excluding NOTA')

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        print(f'Processing {subdir_path}')
        re_fp = os.path.join(subdir_path, 'ffnn_re_results.jsonl')
        if not os.path.isfile(re_fp):
            print(f'No results.json found in {re_fp}')
            continue
        with open(re_fp, 'r') as f:
            re_results = json.load(f)
            if dict_key not in re_results:
                print(f'Key {dict_key} not found in {re_fp}')
                continue
            re_precisions.append(re_results[dict_key]['precision'])
            re_recalls.append(re_results[dict_key]['recall'])
            re_f1s.append(re_results[dict_key]['f1-score'])

    # Calculate mean and standard deviation in percentages
    re_p_mean = statistics.mean(re_precisions) * 100
    re_p_std = statistics.stdev(re_precisions) * 100
    re_r_mean = statistics.mean(re_recalls) * 100
    re_r_std = statistics.stdev(re_recalls) * 100
    re_f1_mean = statistics.mean(re_f1s) * 100
    re_f1_std = statistics.stdev(re_f1s) * 100
    print(
        f'RE precision: {re_p_mean:.1f} ± {re_p_std:.1f}\n'
        f'RE recall: {re_r_mean:.1f} ± {re_r_std:.1f}\n'
        f'RE f1: {re_f1_mean:.1f} ± {re_f1_std:.1f}\n'
    )

    return (
        re_precisions,
        re_recalls,
        re_f1s,
    )


def aggregate_plmarker_numbers(root_dir):
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

    return (
        ner_precisions,
        ner_recalls,
        ner_f1s,
        re_precisions,
        re_recalls,
        re_f1s
    )


def aggregate_numbers(
    pl_root_dir, ffnn_root_dir, ffnn_with_nota=False, avg_type=None
):
    pl_res = aggregate_plmarker_numbers(pl_root_dir)
    ffnn_res = aggregate_ffnn_re_numbers(
        ffnn_root_dir,
        with_nota=ffnn_with_nota,
        avg_type=avg_type
    )
    pl_ner_p, pl_ner_r, pl_ner_f1, pl_re_p, pl_re_r, pl_re_f1 = pl_res
    fn_re_p, fn_re_r, fn_re_f1 = ffnn_res

    # plot precision, recall, f1 means with error bars
    # for ner and both sets of re
    pl_ner_p_mean = statistics.mean(pl_ner_p) * 100
    pl_ner_p_err = statistics.stdev(pl_ner_p) * 100
    pl_ner_r_mean = statistics.mean(pl_ner_r) * 100
    pl_ner_r_err = statistics.stdev(pl_ner_r) * 100
    pl_ner_f1_mean = statistics.mean(pl_ner_f1) * 100
    pl_ner_f1_err = statistics.stdev(pl_ner_f1) * 100
    pl_re_p_mean = statistics.mean(pl_re_p) * 100
    pl_re_p_err = statistics.stdev(pl_re_p) * 100
    pl_re_r_mean = statistics.mean(pl_re_r) * 100
    pl_re_r_err = statistics.stdev(pl_re_r) * 100
    pl_re_f1_mean = statistics.mean(pl_re_f1) * 100
    pl_re_f1_err = statistics.stdev(pl_re_f1) * 100
    fn_re_p_mean = statistics.mean(fn_re_p) * 100
    fn_re_p_err = statistics.stdev(fn_re_p) * 100
    fn_re_r_mean = statistics.mean(fn_re_r) * 100
    fn_re_r_err = statistics.stdev(fn_re_r) * 100
    fn_re_f1_mean = statistics.mean(fn_re_f1) * 100
    fn_re_f1_err = statistics.stdev(fn_re_f1) * 100

    pl_color = '#1f77b4'
    fn_color = '#ff7f0e'

    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 2])
    fig.set_size_inches(7, 2.5)
    # ner
    ax1.errorbar(
        ['P', 'R', 'F1'],
        [pl_ner_p_mean, pl_ner_r_mean, pl_ner_f1_mean],
        yerr=[pl_ner_p_err, pl_ner_r_err, pl_ner_f1_err],
        fmt='o',
        color=pl_color,
        # capsize=5,
    )
    ax1.set_title('Entity Recognition')
    ax1.set_ylim(0, 100)
    # re
    labels = ['P', 'P​', 'R', 'R​', 'F1', 'F1​']
    means = [
        pl_re_p_mean, fn_re_p_mean,
        pl_re_r_mean, fn_re_r_mean,
        pl_re_f1_mean, fn_re_f1_mean
    ]
    errors = [
        pl_re_p_err, fn_re_p_err,
        pl_re_r_err, fn_re_r_err,
        pl_re_f1_err, fn_re_f1_err
    ]
    markers = ['o', 'D', 'o', 'D', 'o', 'D']
    colors = [pl_color, fn_color, pl_color, fn_color, pl_color, fn_color]

    for label, mean, error, marker, color in zip(
        labels, means, errors, markers, colors
    ):
        ax2.errorbar(label, mean, yerr=error, fmt=marker, color=color)

    ax2.set_title('Relation Extraction')
    ax2.set_ylim(0, 100)

    # add global legend with "PL-Marker" (pl_color) and "Ours" (fn_color)
    handles = [
        mlines.Line2D([], [], color=pl_color, marker='o', label='PL-Marker'),
        mlines.Line2D([], [], color=fn_color, marker='D', label='Ours')
    ]
    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(1.1, 0.95),
    )

    plt.savefig('fine_tuned_eval.pdf', bbox_inches='tight')

    # # plot precision, recall, f1 for ner and both sets of re using a boxplot
    # fig = plt.figure(figsize=(8, 6))
    # # ner
    # ax1 = fig.add_subplot(121)
    # ax1.boxplot(
    #     [pl_ner_p, pl_ner_r, pl_ner_f1],
    #     labels=['P', 'R', 'F1'],
    #     showmeans=True,
    # )
    # ax1.set_title('NER')
    # ax1.set_ylim(0, 1)
    # # re
    # ax2 = fig.add_subplot(122)
    # ax2.boxplot(
    #     [pl_re_p, fn_re_p, pl_re_r, fn_re_r, pl_re_f1, fn_re_f1],
    #     labels=['P (p)', 'P (f)', 'R (p)', 'R (f)', 'F1 (p)', 'F1 (f)'],
    # )
    # ax2.set_title('RE')
    # ax2.set_ylim(0, 1)
    # plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]

        # aggregate_predictions(root_dir)
        # aggregate_plmarker_numbers(root_dir)
        # aggregate_ffnn_re_numbers(
        #     root_dir,
        #     with_nota=False,
        #     avg_type=None
        # )
        # print_predictions(root_dir)
        # error_analysis(
        #     root_dir,
        # )
    elif len(sys.argv) == 3:
        pl_root_dir = sys.argv[1]
        ffnn_root_dir = sys.argv[2]

        aggregate_numbers(
            pl_root_dir,
            ffnn_root_dir,
            ffnn_with_nota=False,
            avg_type=None
        )
