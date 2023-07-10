""" Based on ground truth data use string matching on unannotated data.
"""

import re
import uuid
from nltk import sent_tokenize
from hyperpie import settings
from hyperpie.data.load import (
    get_artifact_param_surface_form_pairs, load_unannotated
)
from hyperpie.util.annot import (
    entity_dict, empty_para_annotation, relation_dict,
    surface_form_dict
)


def annotate(
    ground_truth_fp=None,
    target_para_fp=None,
    lim=None
):
    """ Annotate unannotated data based on ground truth data.

        Parameters
            ground_truth_fp, target_para_fp
        default to
            settings.annot_prep_fp, settings.filtered_unannot_fp
        respectively.
    """

    if ground_truth_fp is None:
        ground_truth_fp = settings.annot_prep_fp
    if target_para_fp is None:
        target_para_fp = settings.filtered_unannot_fp

    artif_param_pairs = get_artifact_param_surface_form_pairs(
        paras_fp=ground_truth_fp
    )
    target_para_fp = settings.filtered_unannot_fp  # FIXME: set to unfiltered
    target_paras = load_unannotated(
        transformed_pprs_fp=target_para_fp
    )
    if lim is None:
        lim = len(target_paras)

    annotated_paras = []

    # set up value regex pattern
    val_patt = re.compile(
        # 1e-5 / 1.234e-5 / 1.234e5 / 1.234e+5
        r'(?i)\d+(?:\.\d+)?e[+-]?\d+(?=[\s\.,)])|'
        # 1.234 / 1,234 / 1,234.567 / 1,234.567%
        r'(?<=[\s(])(\d{1,3},)?\d+(?:\.\d+)?%?|'
        # zero / one / ... / nine
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine)\b'
    )
    # go though paras and annotate
    for para in target_paras[:lim]:
        text = para['text']
        doc_id = para['document_id']
        # annotate any of
        # - known artifact param pair (anywhere in paragraph)
        # - known paragraph + closest number (within the same sentence)
        entities = {}
        rels = {}
        for param_surf, artif_surf in artif_param_pairs:
            # artifact parameter rels
            artif_patt = re.compile(
                r'\b{}\b'.format(
                    re.escape(artif_surf)
                )
            )
            param_patt = re.compile(
                r'\b{}\b'.format(
                    re.escape(param_surf)
                )
            )
            if artif_patt.search(text):
                # set up IDs
                a_id = str(uuid.uuid4())
                # create entities
                a_dict = entity_dict(a_id, 'a')
                # get character offsets of artifact surface form
                for m in artif_patt.finditer(text):
                    surf_id = str(uuid.uuid4())
                    surf = surface_form_dict(
                        str(uuid.uuid4()),
                        artif_surf,
                        m.start(),
                        m.end()
                    )
                    a_dict['surface_forms'].append(surf)
            if param_patt.search(text):
                p_id = str(uuid.uuid4())
                p_dict = entity_dict(p_id, 'p')
                # get character offsets of parameter surface form
                for m in param_patt.finditer(text):
                    surf_id = str(uuid.uuid4())
                    surf = surface_form_dict(
                        surf_id,
                        param_surf,
                        m.start(),
                        m.end()
                    )
                    p_dict['surface_forms'].append(surf)
            if artif_patt.search(text) and param_patt.search(text):
                entities[a_id] = a_dict
                entities[p_id] = p_dict
                rel_ap_id = str(uuid.uuid4())
                # create relation
                rel_dict = relation_dict(
                    p_id, a_id,
                    p_dict['surface_forms'], a_dict['surface_forms']
                )
                rels[rel_ap_id] = rel_dict
            if not param_patt.search(text):
                # no parameter, so no need to look for values
                continue
            # parameter value rels
            entities[p_id] = p_dict
            para_sents = sent_tokenize(text)
            for sent in para_sents:
                # check for numbers in sentence
                if not val_patt.search(sent):
                    continue
                # get character offsets of sentence in paragraph
                sent_start = text.find(sent)
                sent_end = sent_start + len(sent)  # noqa
                # check for parameter surface form in sentence
                if not param_patt.search(sent):
                    continue
                # get character offsets of parameter surface form
                param_start = param_patt.search(sent).start() + sent_start
                # identifty closest number to parameter surface form
                closest_num_match = None
                min_dist = len(text)
                for m in val_patt.finditer(sent):
                    val_start = m.start() + sent_start
                    curr_dist = abs(param_start - val_start)
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        closest_num_match = m
                closest_num_surf = closest_num_match.group()
                # create annotation
                v_id = str(uuid.uuid4())
                # create value entity (param already exists)
                val_dict = entity_dict(v_id, 'v')
                val_surf_dict = surface_form_dict(
                    str(uuid.uuid4()),
                    closest_num_surf,
                    closest_num_match.start(),
                    closest_num_match.end()
                )
                val_dict['surface_forms'] = [val_surf_dict]
                entities[v_id] = val_dict
                # create relation
                rel_pv_id = str(uuid.uuid4())
                rel_dict = relation_dict(
                    v_id, p_id,
                    val_dict['surface_forms'], p_dict['surface_forms']
                )
                rels[rel_pv_id] = rel_dict
        if len(entities) > 0:
            annot_dict = empty_para_annotation(
                'distant supervision',
                doc_id,
                None,
                text
            )
            annot_dict['annotation']['entities'] = entities
            annot_dict['annotation']['relations'] = rels
            annotated_paras.append(annot_dict)

    return annotated_paras
