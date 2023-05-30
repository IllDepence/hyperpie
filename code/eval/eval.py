""" Script for caculating evaluation metrics
"""

import json
import sys


def _spans_match(a_start, a_end, b_start, b_end, overlap=False):
    if overlap:
        return _spans_overlap(a_start, a_end, b_start, b_end)
    return a_start == b_start and a_end == b_end


def _spans_overlap(a_start, a_end, b_start, b_end):
    return (
        a_start <= b_start < a_end or
        a_start < b_end <= a_end
    )


def _calc_prec_rec_f1(tp, fp, fn):
    """ Calculate precision, recall and f1-score
    """

    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, f1


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
            if _spans_match(
                start_true, end_true,
                start_pred, end_pred,
                partial_overlap
            ):
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
            if _spans_match(
                start_true, end_true,
                start_pred, end_pred,
                partial_overlap
            ):
                break
        else:
            # see else clause above
            if verbose:
                print(f'FP: {surface_form_pred} predicted but not true')
            fp += 1

    # TN: could be determined on a character level but needed for P/R/F1

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
    # tn: could be determined on a character level but needed for P/R/F1
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

    p, r, f1 = _calc_prec_rec_f1(tp, fp, fn)

    if verbose:
        print(f'P: {p}, R: {r}, F1: {f1}')

    return p, r, f1


def _surface_form_relation_match(
    surf_tup_needle, surf_tup_haystack, partial_overlap=False, directed=True
):
    """ For a two relations of surface forms (needle and haystack),
        determine if the needle relation matches the haystack relation.

        Visualisation
        -------------
        needle:   (start_n0, end_n0) ---rel--> (start_n1, end_n1)
        haystack: (start_h0, end_h0) ---rel--> (start_h1, end_h1)

        Parameters
        ----------
        surf_tup_needle: tuple of surface forms
        surf_tup_haystack: tuple of surface forms
        partial_overlap: bool, whether to consider partial overlap
        directed: bool, whether to consider direction of relation

        Returns
        -------
        match: bool, whether the needle relation matches the haystack relation
    """

    if not directed:
        # order both tuples by start position
        surf_tup_needle = sorted(surf_tup_needle, key=lambda x: x['start'])
        surf_tup_haystack = sorted(surf_tup_haystack, key=lambda x: x['start'])

    # check if the needle is in the haystack
    from_match = _spans_match(
        surf_tup_needle[0]['start'], surf_tup_needle[0]['end'],
        surf_tup_haystack[0]['start'], surf_tup_haystack[0]['end'],
        partial_overlap
    )
    to_match = _spans_match(
        surf_tup_needle[1]['start'], surf_tup_needle[1]['end'],
        surf_tup_haystack[1]['start'], surf_tup_haystack[1]['end'],
        partial_overlap
    )

    return from_match and to_match


def _co_reference_resolution_single(
    y_true, y_pred, partial_overlap=False, verbose=False
):
    """ For a single paragraph, determine
        TP, FP, FN for co-reference resolution
        between pairs of surface forms
    """

    tp = 0
    fp = 0
    fn = 0

    # determine TP and FN
    for e_true in y_true['annotation']['entities'].values():
        # for an entity, check all combinations of its surface forms
        for surf_a in e_true['surface_forms']:
            for surf_b in e_true['surface_forms']:
                if surf_a['id'] == surf_b['id']:
                    # don't check “self” corefs
                    continue
                # check if the surface form pair is in the prediction
                for e_pred in y_pred['annotation']['entities'].values():
                    # if both surface forms are in the same entity,
                    # they are in a co-reference relation
                    found_a = False
                    found_b = False
                    for surf_pred in e_pred['surface_forms']:
                        if _spans_match(
                            surf_a['start'], surf_a['end'],
                            surf_pred['start'], surf_pred['end'],
                            partial_overlap
                        ):
                            found_a = True
                        if _spans_match(
                            surf_b['start'], surf_b['end'],
                            surf_pred['start'], surf_pred['end'],
                            partial_overlap
                        ):
                            found_b = True
                        if found_a and found_b:
                            # found, done
                            break
                        # not found, continue w/ next entity
                    if found_a and found_b:
                        # found, done
                        if verbose:
                            print(f'TP: {surf_a} and {surf_b} predicted')
                        tp += 1
                        break
                else:
                    # true coref not found in prediction
                    if verbose:
                        print(f'FN: {surf_a} and {surf_b} not predicted')
                    fn += 1

    # determine FP
    for e_pred in y_pred['annotation']['entities'].values():
        # for an entity, check all combinations of its surface forms
        for surf_a in e_pred['surface_forms']:
            for surf_b in e_pred['surface_forms']:
                if surf_a['id'] == surf_b['id']:
                    # don't check “self” corefs
                    continue
                # check if the surface form pair is in the prediction
                for e_true in y_true['annotation']['entities'].values():
                    # if both surface forms are in the same entity,
                    # they are in a co-reference relation
                    found_a = False
                    found_b = False
                    for surf_true in e_true['surface_forms']:
                        if _spans_match(
                            surf_a['start'], surf_a['end'],
                            surf_true['start'], surf_true['end'],
                            partial_overlap
                        ):
                            found_a = True
                        if _spans_match(
                            surf_b['start'], surf_b['end'],
                            surf_true['start'], surf_true['end'],
                            partial_overlap
                        ):
                            found_b = True
                        if found_a and found_b:
                            # found, done
                            break
                        # not found, continue w/ next entity
                    if found_a and found_b:
                        # found, done
                        break
                else:
                    # predicted coref not found in ground truth
                    if verbose:
                        print(f'FN: {surf_a} and {surf_b} predicted but wrong')
                    fp += 1

    # TN: could be determined efficiently with number of all surface form
    #     combinations (x(x-1))/2, but needed for P/R/F1

    return tp, fp, fn


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
    # tn: could be determined on a character level but needed for P/R/F1
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

    p, r, f1 = _calc_prec_rec_f1(tp, fp, fn)

    if verbose:
        print(f'P: {p}, R: {r}, F1: {f1}')

    return p, r, f1


def _get_entity_id_by_entity_surf_forms(
    needle_entity, haystack_entity_dict,
    partial_overlap=False, verbose=False
):
    """ For an entity to search (needle) for a set of annotations A
        and a dictionary of entities to search in (haystack) for as set B.
        Return the ID of the entity in the haystack that matches the needle.
    """

    found_eid = None
    for surf_form in needle_entity['surface_forms']:
        found_eid = _get_entity_id_by_surf_form_offset(
            surf_form['start'], surf_form['end'],
            haystack_entity_dict,
            partial_overlap,
            verbose
        )
        if found_eid is not None:
            # NOTE: we assume there is just one matching entity, but
            #       theoretically there can be multiple. I.e. needle
            #       having two surface forms which, in haystack, are
            #       associated with two different entities.
            break

    return found_eid


def _get_entity_id_by_surf_form_offset(
    surf_start, surf_end, entity_dict,
    partial_overlap=False, verbose=False
):
    """ For a dictionary of entities (e_id, e), figure out if
        a surface form is part of an entity and return its ID
    """

    for e_id, e in entity_dict.items():
        # check each surface form of the predicted entity
        for surf in e['surface_forms']:
            if _spans_match(
                surf_start, surf_end,
                surf['start'], surf['end'],
                partial_overlap
            ):
                return e_id

    return None


def _get_realtion_targets(
    source_e_id, relation_dict, verbose=False
):
    """ For a dictionary of relations (r_id, r), figure out if
        a source entity is part of one or more relation and return
        the list of target entity IDs
    """

    targets = []

    for r_id, r in relation_dict.items():
        if r['source'] == source_e_id:
            targets.append(r['target'])

    return targets


def _relation_extraction_single(
    y_true, y_pred, partial_overlap=False, verbose=False
):
    """ For a single paragraph, determine TP, FP, FN for relation extraction
    """

    tp = 0
    fp = 0
    fn = 0

    # determine TP and FN
    for rel_id, rel_true in y_true['annotation']['relations'].items():
        # source entity in groud truth relation
        from_true_entity = y_true['annotation']['entities'][rel_true['source']]
        # target entity in groud truth relation
        to_true_entity = y_true['annotation']['entities'][rel_true['target']]
        # identify matching entity in predicted entities
        from_pred_entity_id = _get_entity_id_by_entity_surf_forms(
            from_true_entity, y_pred['annotation']['entities'],
            partial_overlap, verbose
        )
        if from_pred_entity_id is None:
            # source entity not found
            if verbose:
                print(
                    f'FN: true source entity {from_true_entity["id"]} '
                    f'not found'
                )
            fn += 1
            continue
        to_pred_entity_ids = _get_realtion_targets(
            from_pred_entity_id, y_pred['annotation']['relations'],
            verbose
        )
        if len(to_pred_entity_ids) == 0:
            # source entity found but not in a relation
            if verbose:
                print((
                    f'FN: true source entity {from_true_entity["id"]} '
                    f'found but not in a relation'
                ))
            fn += 1
            continue
        # source entity found in at least one relation
        # check one matches the ground truth target entity
        found_matching_true_source = False
        for to_pred_entity_id in to_pred_entity_ids:
            to_pred_entity = y_pred['annotation']['entities'][
                to_pred_entity_id
            ]
            check_eid = _get_entity_id_by_entity_surf_forms(
                to_pred_entity, y_true['annotation']['entities'],
                partial_overlap, verbose
            )
            if check_eid == to_true_entity['id']:
                found_matching_true_source = True
                break
        if not found_matching_true_source:
            # relation to wrong target entity
            if verbose:
                print((
                    f'FN: predicted relation {from_pred_entity_id} -> '
                    f'{to_pred_entity_id} does not match ground truth '
                ))
            fn += 1
        else:
            # relation to correct target entity
            if verbose:
                print((
                    f'TP: {rel_true["source"]} -> {rel_true["target"]} '
                    f'found in prediction as\n    {from_pred_entity_id} -> '
                    f'{to_pred_entity_id}'
                ))
            tp += 1

    # determine FP
    for rel_id, rel_pred in y_pred['annotation']['relations'].items():
        # source entity in predicted relation
        from_pred_entity = y_pred['annotation']['entities'][rel_pred['source']]
        # target entity in predicted relation
        to_pred_entity = y_pred['annotation']['entities'][rel_pred['target']]
        # identify entity in ground truth matching predicted
        from_true_entity_id = _get_entity_id_by_entity_surf_forms(
            from_pred_entity, y_true['annotation']['entities'],
            partial_overlap, verbose
        )
        if from_true_entity_id is None:
            # source entity not found
            if verbose:
                print(
                    f'FP: predicted source entity {from_pred_entity["id"]} '
                    f'not found'
                )
            fp += 1
            continue
        # source entity found
        # get its corresponding target entity
        to_true_entity_ids = _get_realtion_targets(
            from_true_entity_id, y_true['annotation']['relations'],
            verbose
        )
        if len(to_true_entity_ids) == 0:
            # source entity found but not in a relation
            if verbose:
                print(
                    f'FP: predicted source entity {from_pred_entity["id"]} '
                    f'found but not in a relation'
                )
            fp += 1
            continue
        # source entity found in at least one relation
        # check if one matches the predicted target entity
        found_matching_pred_source = False
        for to_true_entity_id in to_true_entity_ids:
            to_true_entity = y_true['annotation']['entities'][
                to_true_entity_id
            ]
            check_eid = _get_entity_id_by_entity_surf_forms(
                to_true_entity, y_pred['annotation']['entities'],
                partial_overlap, verbose
            )
            if check_eid == to_pred_entity['id']:
                found_matching_pred_source = True
                break
        if not found_matching_pred_source:
            # relation to wrong target entity
            if verbose:
                print(
                    f'FP: predicted relation {from_pred_entity_id} -> '
                    f'{to_pred_entity_id} does not match ground truth '
                )
            fp += 1
        else:
            # relation to correct target entity
            # already counted as TP
            pass

    return tp, fp, fn


def relation_extraction(y_true, y_pred, partial_overlap=False, verbose=False):
    """ Calculate precision, recall and f1-score for the prediction of
        relations between entities (not surface forms)
    """

    tp = 0
    fp = 0
    fn = 0
    # tn: could be determined on a character level but needed for P/R/F1
    for i in range(len(y_true)):
        tp_i, fp_i, fn_i = _relation_extraction_single(
            y_true[i], y_pred[i], partial_overlap, verbose
        )
        tp += tp_i
        fp += fp_i
        fn += fn_i

    p, r, f1 = _calc_prec_rec_f1(tp, fp, fn)

    return p, r, f1


def full(y_true, y_pred):
    """ Calculate all the metrics
    """

    for partial_overlap in [False, True]:
        print(f'\n- - - Partial overlap: {partial_overlap} - - -')

        # entity recognition
        p, r, f1 = entity_recognition(
            y_true, y_pred, check_type=False, partial_overlap=partial_overlap
        )
        print('\nER')
        print(f'P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}')

        # entity recognition + classification
        p, r, f1 = entity_recognition(
            y_true, y_pred, check_type=True, partial_overlap=partial_overlap
        )
        print('ER + Clf')
        print(f'P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}')

        # co-reference resolution
        p, r, f1 = co_reference_resolution(
            y_true, y_pred, partial_overlap=partial_overlap
        )
        print('Co-ref resol.')
        print(f'P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}')

        # relation extraction
        p, r, f1 = relation_extraction(
            y_true, y_pred, partial_overlap=partial_overlap
        )
        print('Rel. extr.')
        print(f'P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python eval.py <ground_truth_file> <prediction_file>')
        sys.exit(1)

    y_true = json.load(open(sys.argv[1]))
    y_pred = json.load(open(sys.argv[2]))

    # make sure ground truth and prediction have the same length
    if len(y_true) != len(y_pred):
        print((
            f'Error: ground truth and prediction have different lengths: '
            f'{len(y_true)} != {len(y_pred)}'
        ))
        sys.exit(1)

    full(y_true, y_pred)
