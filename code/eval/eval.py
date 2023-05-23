""" Script for caculating evaluation metrics
"""

import json
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, \
        classification_report


def _entity_recognition_single(y_true, y_pred, partial_overlap, verbose=True):
    """ For a single paragraph, determine
        TP, FP, FN for entity recognition
    """

    # get surface forms from entity annotations
    surface_forms_true = []
    surface_forms_pred = []
    for entity in y_true['annotation']['entities'].values():
        surface_forms_true.extend(entity['surface_forms'])
    for entity in y_pred['annotation']['entities'].values():
        surface_forms_pred.extend(entity['surface_forms'])

    if verbose:
        print(f'Annotator: {y_true["annotator_id"]}')
        print(f'Document: {y_true["document_id"]}')
        print(f'Paragraph: {y_true["paragraph_index"]}')
        print(f'Loaded {len(surface_forms_true)} true surface forms')
        print(f'Loaded {len(surface_forms_pred)} predicted surface forms')

    # determine TP and FN
    tp = 0
    fn = 0
    for surface_form_true in surface_forms_true:
        start_true = surface_form_true['start']
        end_true = surface_form_true['end']
        for surface_form_pred in surface_forms_pred:
            start_pred = surface_form_pred['start']
            end_pred = surface_form_pred['end']
            if start_true == start_pred and end_true == end_pred:
                tp += 1
                break
            elif partial_overlap and \
                    (start_true <= start_pred <= end_true or
                     start_true <= end_pred <= end_true):
                tp += 1
                break
        else:
            # else clause of for loop is executed
            # if no break statement was executed
            # i.e. no match for the predicted surface
            # form was found in the true surface forms
            fn += 1
            if verbose:
                print(f'FN: true {surface_form_true} not found in prediction')
    # determine FP
    fp = 0
    for surface_form_pred in surface_forms_pred:
        start_pred = surface_form_pred['start']
        end_pred = surface_form_pred['end']
        for surface_form_true in surface_forms_true:
            start_true = surface_form_true['start']
            end_true = surface_form_true['end']
            if start_true == start_pred and end_true == end_pred:
                break
            elif partial_overlap and \
                    (start_true <= start_pred <= end_true or
                     start_true <= end_pred <= end_true):
                break
        else:
            # see else clause above
            if verbose:
                print(f'FP: {surface_form_pred} predicted but not true')
            fp += 1

    return tp, fp, fn


def entity_recognition(y_true, y_pred, partial_overlap=False):
    """ Calculate precision, recall and f1-score for detection of
        entities in text (entity classes are not considered)

        Parameters
        ----------
        y_true: list of paragraphs with true annotations
        y_pred: list of paragraphs with predicted annotations
        partial_overlap: bool, whether to consider partial overlap

        Returns
        -------
        precision: float
        recall: float
        f1_score: float
    """

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        tp_i, fp_i, fn_i = _entity_recognition_single(
            y_true[i],
            y_pred[i],
            partial_overlap
        )
        tp += tp_i
        fp += fp_i
        fn += fn_i

    print(f'TP: {tp}, FP: {fp}, FN: {fn}')

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1_score}')

    return precision, recall, f1_score


def full(y_true, y_pred):
    pass


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python eval.py <ground_truth_file> <prediction_file>')
        sys.exit(1)

    y_true = json.load(open(sys.argv[1]))
    y_pred = json.load(open(sys.argv[2]))

    entity_recognition(y_true, y_pred, partial_overlap=True)
