""" Post-processing for prediction on unlabeled data.
"""

import argparse
import json
import os


def merge(root_dir):
    """ Merge RE prediction output into NER prediction output.
    """

    # load data
    smpl2offs_fn = 'sample_idx_to_entity_offset_pair.json'
    pred_fn = 'y_pred.json'
    ner_pred_fp = os.path.join(
        'all',
        'ent_pred_test.json'
    )
    merged_pred_fn = 'merged_preds.jsonl'
    ner_pred = []

    with open(os.path.join(root_dir, smpl2offs_fn), 'r') as f:
        smpl2offs = json.load(f)
    with open(os.path.join(root_dir, pred_fn), 'r') as f:
        re_pred = json.load(f)
    with open(os.path.join(root_dir, ner_pred_fp), 'r') as f:
        for line in f.readlines():
            ner_pred.append(json.loads(line))

    # get predicted relations
    predicted_rels = {}
    for smpl_idx_str, rel in smpl2offs.items():
        smpl_idx = int(smpl_idx_str)
        pred = re_pred[smpl_idx]
        if pred == 0:
            # not predicted as a relation
            continue
        assert pred == 1
        # save predicted relation
        doc_key = rel['doc_key']
        sent_idx = rel['sent_idx']
        rel_offset = rel['rel']
        if doc_key not in predicted_rels:
            predicted_rels[doc_key] = []
        predicted_rels[doc_key].append(
            [sent_idx, rel_offset]
        )

    # add predicted relations to NER prediction file
    merged_pred = []
    for para in ner_pred:
        # add vessel for predicted relations
        para['predicted_relations'] = []
        for i in range(len(para['sentences'])):
            para['predicted_relations'].append([])  # one for each sentence
        # add predicted relations
        pred_rel_tup = predicted_rels.get(para['doc_key'], [])
        for pd_tup in pred_rel_tup:
            sent_idx, rel_offset = pd_tup
            para['predicted_relations'][sent_idx].append(rel_offset)
        merged_pred.append(para)

    # save merged prediction
    merged_pred_fp = os.path.join(root_dir, merged_pred_fn)
    print(f'saving merged prediction to {merged_pred_fp}')
    with open(merged_pred_fp, 'w') as f:
        for para in merged_pred:
            f.write(json.dumps(para) + '\n')

    return merged_pred


def print_predictions(paras):
    """ Print merged NER+RE predictions for inspection.
    """

    # iterate over paras
    for para in paras:
        print(f'= = = {para["doc_key"]} = = =')
        para_delta = 0
        # iterate over sentences
        for sent_idx, sent in enumerate(para['sentences']):
            # print('[' + '] ['.join(sent) + ']')
            sent_str = ' '.join(sent)
            print(sent_str)
            ner_pred = para['predicted_ner'][sent_idx]
            ner_gold = para['ner'][sent_idx]
            print('<<<NER>>>')
            offs_to_label = {}
            for (start, end, label) in ner_pred:
                print(sent[start-para_delta:end-para_delta+1], label)
                offs_to_label[(start, end)] = label
            # print('<<<NER>>> (gold)')
            for (start, end, label) in ner_gold:
                # print(sent[start-para_delta:end-para_delta+1], label)
                offs_to_label[(start, end)] = label
            has_rels = False
            if 'predicted_relations' in para:
                re_pred = para['predicted_relations'][sent_idx]
                # data cleaning
                # re_pred_clean = []
                # for (
                #     start_from, end_from, start_to, end_to, label
                # ) in re_pred:
                #     label_from = offs_to_label.get((start_from, end_from))
                #     label_to = offs_to_label.get((start_to, end_to))
                print('<<<RE>>>')
                for (
                    start_from, end_from, start_to, end_to, label
                ) in re_pred:
                    has_rels = True
                    label_from = offs_to_label.get((start_from, end_from))
                    label_to = offs_to_label.get((start_to, end_to))
                    print(
                        sent[start_from-para_delta:end_from-para_delta+1],
                        '--',
                        label,
                        '->',
                        sent[start_to-para_delta:end_to-para_delta+1],
                        f'({label_from} -> {label_to})'
                    )
            para_delta += len(sent)
            if has_rels:
                input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str)
    args = parser.parse_args()

    paras = merge(args.root_dir)
    print_predictions(paras)
