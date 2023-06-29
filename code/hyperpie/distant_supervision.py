""" Based on ground truth data use string matching on unannotated data.
"""

import re
import uuid
from nltk import sent_tokenize
from hyperpie import settings
from hyperpie.data.load import get_artifact_param_surface_form_pairs
from hyperpie.util.annot import (
    entity_dict, empty_para_annotation, relation_dict,
    surface_form_dict
)


def annotate(
    ground_truth_fp=None,
    target_para_fp=None
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
    target_para_fp - settings.filtered_unannot_fp  # FIXME: set to unfiltered
    target_paras = data.load_unannotated(
        transformed_pprs_fp=target_para_fp
    )

    # set up value regex patterns
    # 1.234 / 1,234 / 1,234.567 / 1,234.567%
    val_num_patt = re.compile(
        r'(?<=[\s(])(\d{1,3},)?\d+(?:\.\d+)?%?(?=[\s\.,)])'
    )
    # zero / one / ... / nine
    val_num_word_patt = re.compile(
        r'(?i)\b(zero|one|two|three|four|five|six|seven|eight|nine)\b'
    )
    # 1e-5 / 1.234e-5 / 1.234e5 / 1.234e+5
    sci_not_patt = re.compile(
        r'(?<=[\s(])\d+(?:\.\d+)?e[+-]?\d+(?=[\s\.,)])'
    )
    # go though paras and annotate
    for para in target_paras:
        text = para['text']
        # annotate any of
        # - known artifact param pair (anywhere in paragraph)
        # - known paragraph + closest number (within the same sentence)
        # TODO:
        # - iterate over known artifact param pairs and create annotations
        #   (create regex patts w/ word boundaries)
        # - split para into sentences, iterate over known params, look for
        #   closest number and create annotation
        entities = {}
        rels = {}
        for artif_surf, param_surf in artif_param_pairs:
            # artifact parameter rels
            artif_patt = re.compile(r'\b{}\b'.format(artif_surf))
            param_patt = re.compile(r'\b{}\b'.format(param_surf))
            if artif_patt.search(text) and param_patt.search(text):
                # set up IDs
                a_id = str(uuid.uuid4())
                p_id = str(uuid.uuid4())
                rel_id = str(uuid.uuid4())
                # create relation
                rel_dict = relation_dict(
                    a_id, p_id, [artif_surf], [param_surf]
                )
                rels[rel_id] = rel_dict
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
                entities[a_id] = a_dict
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
                entities[p_id] = p_dict
            # parameter value rels
            para_sents = sent_tokenize(text)
            for sent in para_sents:
                # check for numbers in sentence
                if not any([
                    val_num_patt.search(sent),
                    val_num_word_patt.search(sent),
                    sci_not_patt.search(sent)
                ]):
                    continue
                # get character offsets of sentence in paragraph
                sent_start = text.find(sent)
                sent_end = sent_start + len(sent)
                # check for parameter surface form in sentence
                if not param_patt.search(sent):
                    continue
                # identifty closest number to parameter surface form
                # TODO continue here
