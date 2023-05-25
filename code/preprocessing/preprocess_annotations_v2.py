""" Turn raw recogito.js annotations into more convenient format.

    Specifically, raw annotations employ the following aspects
    for annotation efficiency:
        - surface forms of artifacts and parameters carry not only
          the entity type (a/p) but also an entity ID (1, 2, 3, ...)
          -> need to "collect" all surface forms of a given entity
        - surface forms of numbers and contexts carry not only the
          entity type (n/c)
          -> each surface form is a separate entity
        - surface forms of type value carry type information
          (vn (number), vr (range), vs (set), vo (other))
          -> turn into an (optional) attribute of the entity
        - relations are annotated between surface forms, thereby
          containing the information
          1. which entities are linked, and
          2. which sentence in the text is the evidence for the
          -> set relations between entities rather than surface forms
             and "collect" the evidence sentences for each relation
             (can be multiple)
"""


import json
import os
import re
import sys
from collections import OrderedDict
from nltk.tokenize import sent_tokenize


def flatten_annot_dict(annot_dict):
    """ Turn a dictionary
        {
            'annotator_id': '<some_id>',
            'annotations':
            [
                {'in_doc': ...},
                {'in_doc': ...},
                ...
            ]
        }
        into a list
        [
            {'annotator_id': '<some_id>', 'in_doc', ..,},
            {'annotator_id': '<some_id>', 'in_doc', ..,},
            ...
        ]
     """

    annots_flat = []
    annotator_id = annot_dict['annotator_id']
    for annot in annot_dict['annotations']:
        annot_flat = annot.copy()
        annot_flat['annotator_id'] = annotator_id
        annots_flat.append(annot_flat)
    return annots_flat


def order_and_rename_annots(annots):
    """ Put annotation dict info into a fixed order and rename keys.
    """

    annots_nice = []
    for annot in annots:
        annot_nice = OrderedDict()
        annot_nice['annotator_id'] = annot['annotator_id']
        annot_nice['document_id'] = annot['in_doc']
        annot_nice['paragraph_index'] = annot['in_doc_para_idx']
        annot_nice['text'] = annot['context']
        annot_nice['annotation'] = []
        annot_nice['annotation_raw'] = annot['annotation']
        annots_nice.append(annot_nice)

    return annots_nice


def label_shorthand_to_id_and_type(label):
    """ Turn annotation UI “shorthands” (see script docstring) into
        entity ID and type.

        a/p: separate type and ID
        n: separate base type and subtype
        n/c: assign ID
    """

    e_id = None
    e_type = None
    e_subtype = None
    if label[0] in ['a', 'p']:
        e_id = label
        e_type = label[0]
    elif label[0] in ['v', 'c']:
        e_type = label[0]
        if label[0] == 'v':
            e_subtype = label[1]
    else:
        print('ERROR: Unknown entity type in label {}'.format(label))
        raise

    return e_id, e_type, e_subtype


def untangle_single_para_annotations(para):
    """ Turn annotation UI “shorthands” (see script docstring) into
        a clearer format.
    """

    # tokenize text into sentences for evidence sentence extraction
    para_sentences = sent_tokenize(para['text'])
    # dertermine start and end offset of sentences in text
    para_sentence_offsets = {}
    for sent_idx, sent in enumerate(para_sentences):
        start = para['text'].find(sent)
        end = start + len(sent)
        para_sentence_offsets[sent_idx] = (start, end)

    # separate surface form and relation annotations
    surface_forms_annots_raw = []
    relations_annots_raw = []
    for annot in para['annotation_raw']:
        if annot.get('motivation', '') == 'linking':
            # relation annotation
            relations_annots_raw.append(annot)
        else:
            # surface form annotation
            surface_forms_annots_raw.append(annot)

    # process surface form annotations
    entities = {}
    surf_id_to_e_id = {}
    surf_id_to_offset = {}
    value_id_idx = 1
    context_id_idx = 1
    for annot in surface_forms_annots_raw:
        # put unwieldy Web Annot Data Model into easy to understand
        # variable names
        surf_id = annot['id']
        annot_label = annot['body'][0]['value']
        annot_selectors = annot['target']['selector']
        # untangle label/id/subtype shorthand
        e_id, e_type, e_subtype = label_shorthand_to_id_and_type(
            annot_label
        )
        if e_type in ['v', 'c']:
            # assign an ID
            if e_type == 'v':
                e_id = f'v{value_id_idx}'
                value_id_idx += 1
            elif e_type == 'c':
                e_id = f'c{context_id_idx}'
                context_id_idx += 1
        surf_id_to_e_id[surf_id] = e_id
        # get surface form and text offset
        for selector in annot_selectors:
            if selector['type'] == 'TextQuoteSelector':
                # surface form
                surface_form = selector['exact']
            elif selector['type'] == 'TextPositionSelector':
                # text offset
                start = selector['start']
                end = selector['end']
                surf_id_to_offset[surf_id] = (start, end)
        # save entity / surface form
        if e_id not in entities:
            # new entity
            entities[e_id] = OrderedDict({
                'id': e_id,
                'type': e_type,
                'subtype': e_subtype,
                'surface_forms': [],
            })
        entities[e_id]['surface_forms'].append({
            'id': surf_id,
            'surface_form': surface_form,
            'start': start,
            'end': end,
        })

    # create mapping from relation entity IDs to relation IDs
    rel_entities_to_id = {}
    for rel_idx, annot in enumerate(relations_annots_raw):
        # put unwieldy Web Annot Data Model into easy to understand
        # variable names
        surf_from = annot['target'][0]['id']
        surf_to = annot['target'][1]['id']
        # get entity IDs
        e_from_id = surf_id_to_e_id[surf_from]
        e_to_id = surf_id_to_e_id[surf_to]
        # save relation
        rel_id = f'r{rel_idx}'
        rel_key = f'{e_from_id}-{e_to_id}'
        rel_entities_to_id[rel_key] = rel_id
    # process relation annotations
    relations = {}
    for annot in relations_annots_raw:
        # put unwieldy Web Annot Data Model into easy to understand
        # variable names
        surf_rel_id = annot['id']
        surf_from = annot['target'][0]['id']
        surf_to = annot['target'][1]['id']
        # get pre-generated relation ID (multiple surface form relations
        # can have the same relation ID)
        rel_key = f'{e_from_id}-{e_to_id}'
        rel_id = rel_entities_to_id[rel_key]
        # get entity IDs
        e_from_id = surf_id_to_e_id[surf_from]
        e_to_id = surf_id_to_e_id[surf_to]
        # get evidence sentence(s)
        offset_from = surf_id_to_offset[surf_from]
        offset_to = surf_id_to_offset[surf_to]
        offset_left = min(offset_from[0], offset_to[0])
        offset_right = max(offset_from[1], offset_to[1])
        sent_idx_from = False
        sent_idx_to = False
        for sent_idx, (start, end) in para_sentence_offsets.items():
            # if the leftmost surface form offset lies within the sentence
            # boundaries, it is the (leftmost) evidence sentence
            if offset_left >= start and offset_left <= end:
                sent_idx_from = sent_idx
            # if the rightmost surface form offset lies within the sentence
            # boundaries, it is the (rightmost) evidence sentence
            if offset_right >= start and offset_right <= end:
                sent_idx_to = sent_idx
        # convert sentences that form relation evidence
        evidence_start = para_sentence_offsets[sent_idx_from][0]
        evidence_end = para_sentence_offsets[sent_idx_to][1]
        evidence_text = para['text'][evidence_start:evidence_end]
        # save relation
        if rel_id not in relations:
            relations[rel_id] = OrderedDict({
                'id': rel_id,
                'source': e_from_id,
                'target': e_to_id,
                'evidences': [],
            })
        relations[rel_id]['evidences'].append({
            'id': surf_rel_id,
            'source_surface_form': surf_from,
            'target_surface_form': surf_to,
            'evidence_sentence': evidence_text,
            'start': evidence_start,
            'end': evidence_end,
        })

    para['annotation'] = {
        'entities': entities,
        'relations': relations
    }

    return para


def preprocess(annots_path):
    # load and pre-process annotated text segments
    save_path = '../data/annotation'
    annots_fn = os.path.basename(annots_path)
    annots_fn_base, ext = os.path.splitext(annots_fn)
    annots_processed_fn = f'{annots_fn_base}_processed{ext}'

    # load annotations
    with open(annots_path, 'r') as f:
        annots = json.load(f)
    # process annotations
    annots_flat = flatten_annot_dict(annots)
    annots_nice = order_and_rename_annots(annots_flat)
    annots_processed = []
    for para in annots_nice:
        para = untangle_single_para_annotations(para)
        annots_processed.append(para)

    # save processed annotations
    with open(os.path.join(save_path, annots_processed_fn), 'w') as f:
        json.dump(annots_processed, f)


if __name__ == '__main__':
    # check command line arguments
    if len(sys.argv) != 2:
        print(
            'Usage: python preprocess_annotations_v2.py /path/to/annots.json'
        )
        sys.exit(1)
    annots_path = sys.argv[1]
    preprocess(annots_path)
