""" Script for caculating evaluation metrics
"""

import json
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, \
        classification_report


def _spans_match(a_start, a_end, b_start, b_end):
    return a_start == b_start and a_end == b_end


def _spans_overlap(a_start, a_end, b_start, b_end):
    return (
        a_start < b_start < a_end or
        a_start < b_end < a_end
    )


def _typed_entity_surface_forms(para):
    """ Get entity typed surface forms from annitations
        of a single paragraph
    """

    surface_forms = []
    for entity in para['annotation']['entities'].values():
        e_type = entity['type']
        for surf in entity['surface_forms']:
            typed_surf = surf.copy()
            typed_surf['type'] = e_type
            surface_forms.append(typed_surf)

    return surface_forms


def _entity_recognition_single(
    y_true, y_pred, partial_overlap, check_type, verbose
):
    """ For a single paragraph, determine
        TP, FP, FN for entity recognition
        and optionally entity type classification
    """

    # get surface forms from entity annotations
    surface_forms_true = _typed_entity_surface_forms(y_true)
    surface_forms_pred = _typed_entity_surface_forms(y_pred)

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
            # check entity type if requested
            correct_type = True
            if (
                check_type and
                surface_form_true['type'] != surface_form_pred['type']
            ):
                correct_type = False
            # check if start and end positions match
            if _snans_match(start_true, end_true, start_pred, end_pred):
                if not correct_type:
                    # exact match but wrong type
                    if verbose:
                        print(
                            f'FN: {surface_form_pred} has wrong type'
                        )
                    fn += 1
                    break
                tp += 1
                break
            # check if there is a partial overlap
            elif partial_overlap and _spans_overlap(
                start_true, end_true, start_pred, end_pred
            ):
                if not correct_type:
                    # partial overlap but wrong type
                    if verbose:
                        print(
                            f'FN: {surface_form_pred} has wrong type'
                        )
                    fn += 1
                    break
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
            if _spans_match(start_true, end_true, start_pred, end_pred):
                break
            elif partial_overlap and _spans_overlap(
                start_true, end_true, start_pred, end_pred
            ):
                break
        else:
            # see else clause above
            if verbose:
                print(f'FP: {surface_form_pred} predicted but not true')
            fp += 1

    return tp, fp, fn


def entity_recognition(
        y_true, y_pred, partial_overlap=False, check_type=False, verbose=False
):
    """ Calculate precision, recall and f1-score for detection of
        entities in text (entity classes are not considered)

        Parameters
        ----------
        y_true: list of paragraphs with true annotations
        y_pred: list of paragraphs with predicted annotations
        partial_overlap: bool, whether to consider partial overlap
        check_type: bool, whether to check entity type

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
            partial_overlap,
            check_type,
            verbose
        )
        tp += tp_i
        fp += fp_i
        fn += fn_i

    if verbose:
        print(f'TP: {tp}, FP: {fp}, FN: {fn}')

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    if verbose:
        print(f'P: {precision}, R: {recall}, F1: {f1_score}')

    return precision, recall, f1_score


def _surface_form_coref(
    surf_tup_needle, surf_tup_haystack, partial_overlap=False
):
    """ For a two surface form tuples, a needle and a haystack,
        determine whether the needle is a coreference of the haystack

        Visualisation:
        needle:   (start_n1, end_n1) ---coref--- (start_n2, end_n2)
        haystack: (start_h1, end_h1) ---coref--- (start_h2, end_h2)
    """

    # order both tuples by start position
    surf_tup_needle = sorted(surf_tup_needle, key=lambda x: x['start'])
    surf_tup_haystack = sorted(surf_tup_haystack, key=lambda x: x['start'])

    # check if the needle coreference is a coreference in the haystack
    from_match = _spans_match(
        surf_tup_needle[0]['start'],
        surf_tup_needle[0]['end'],
        surf_tup_haystack[0]['start'],
        surf_tup_haystack[0]['end']
    )
    if partial_overlap:
        from_match = _spans_overlap(
            surf_tup_needle[0]['start'],
            surf_tup_needle[0]['end'],
            surf_tup_haystack[0]['start'],
            surf_tup_haystack[0]['end']
        )
    to_match = _spans_match(
        surf_tup_needle[1]['start'],
        surf_tup_needle[1]['end'],
        surf_tup_haystack[1]['start'],
        surf_tup_haystack[1]['end']
    )
    if partial_overlap:
        to_match = _spans_overlap(
            surf_tup_needle[1]['start'],
            surf_tup_needle[1]['end'],
            surf_tup_haystack[1]['start'],
            surf_tup_haystack[1]['end']
        )

    return from_match and to_match


def _co_reference_resolution_single(
    y_true, y_pred, partial_overlap=False, verbose=False
):
    """ For a single paragraph, determine
        TP, FP, FN for co-reference resolution
        between pairs of surface forms
    """

    # TODO:
    # - for all co-refs in y_true, check if they are in y_pred
    # - for all co-refs in y_pred, check if they are in y_true
    pass


def co_reference_resolution(
    y_true, y_pred, partial_overlap=False, verbose=False
):
    """ Calculate precision, recall and f1-score for the
        prediction of co-references

        NOTE: not modeled as a typical binary classification problem

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
        tp_i, fp_i, fn_i = _co_reference_resolution_single(
            y_true[i],
            y_pred[i],
            partial_overlap,
            verbose
        )
        tp += tp_i
        fp += fp_i
        fn += fn_i

    if verbose:
        print(f'TP: {tp}, FP: {fp}, FN: {fn}')

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    if verbose:
        print(f'P: {precision}, R: {recall}, F1: {f1_score}')

    return precision, recall, f1_score


def full(y_true, y_pred):
    """ Calculate all the metrics
    """

    # entity recognition (no partial overlap)
    p, r, f1 = entity_recognition(
        y_true, y_pred
    )
    print(f'\nEntity recognition (no partial overlap):')
    print(f'P: {p}, R: {r}, F1: {f1}')

    # entity recognition (partial overlap)
    p, r, f1 = entity_recognition(
        y_true, y_pred, partial_overlap=True
    )
    print(f'\nEntity recognition (partial overlap):')
    print(f'P: {p}, R: {r}, F1: {f1}')

    # entity recognition + classification (no partial overlap)
    p, r, f1 = entity_recognition(
        y_true, y_pred, check_type=True
    )
    print(f'\nEntity recognition + classification (no partial overlap):')
    print(f'P: {p}, R: {r}, F1: {f1}')

    # entity recognition + classification (partial overlap)
    p, r, f1 = entity_recognition(
        y_true, y_pred, partial_overlap=True, check_type=True
    )
    print(f'\nEntity recognition + classification (partial overlap):')
    print(f'P: {p}, R: {r}, F1: {f1}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python eval.py <ground_truth_file> <prediction_file>')
        sys.exit(1)

    y_true = json.load(open(sys.argv[1]))
    y_pred = json.load(open(sys.argv[2]))

    full(y_true, y_pred)
