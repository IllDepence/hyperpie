""" Deal with LLM output and make it compatible with the evaluation script.

    Use

        convert(input_paragraph, llm_output_yaml)

    to convert LLM output to the format expected by the evaluation function.
"""

import re
import sys
import json
import uuid
import yaml
from collections import OrderedDict


def surface_form_dict(s_id, surf_form, start, end):
    """ Create a surface form dict.
    """

    surface_form = OrderedDict({
      "id": s_id,
      "surface_form": surf_form,
      "start": start,
      "end": end
    })

    return surface_form


def entity_dict(e_id, e_type):
    """ Create an entity dict.
    """

    annot = OrderedDict({
      "id": e_id,
      "type": e_type,
      "subtype": None,
      "surface_forms":
      [
      ]
    })

    return annot


def relation_evidence_dict(src_srf, trg_srf):
    """ Create a relation evidence dict.
    """

    relation_evidence = OrderedDict({
        "id": str(uuid.uuid4()),
        "source_surface_form": src_srf["id"],
        "target_surface_form": trg_srf["id"],
        "evidence_sentence": None,
        "start": None,
        "end": None
    })

    return relation_evidence


def relation_dict(src_a_id, trg_a_id, src_srfs, trg_srfs):
    """ Create a relation dict.
    """

    relation = OrderedDict({
        "id": str(uuid.uuid4()),
        "source": src_a_id,
        "target": trg_a_id,
        "evidences": []
    })

    for src_srf_id in src_srfs:
        for trg_srf_id in trg_srfs:
            relation["evidences"].append(
                relation_evidence_dict(src_srf_id, trg_srf_id)
            )

    return relation


def empty_para_annotation(annotator_id, document_id, paragraph_index):
    """ Create an empty annotation dict.
    """

    empty_para_annot = OrderedDict({
      "annotator_id": annotator_id,
      "document_id": document_id,
      "paragraph_index": paragraph_index,
      "text": "",
      "annotation":
      {
        "entities": {},
        "relations": {}
      },
      "annotation_raw": []
    })

    return empty_para_annot


def find_surface_forms_in_para(para_text, e_name):
    """ Find all surface forms of an entity in a paragraph.

    Args:
        para_text (str): Paragraph text.
        e_name (str): Entity name.

    Returns:
        list: List of dicts with surface form information.
    """

    # use regex to identify the start and end offset of
    # all occurrences of entity name in paragraph

    surfs = []
    for m in re.finditer(e_name, para_text):
        start = m.start()
        end = m.end()
        surf_id = f'{start}-{end}'
        surfs.append(
            surface_form_dict(
                surf_id,
                e_name,
                start,
                end
            )
         )

    return surfs


def yaml2json(llm_output_dict, verbose=False):
    """ Convert LLM output to evaluation script input format.

    Args:
        llm_output (str): LLM output in YAML format.

    Returns:
        str: Evaluation script input in JSON format.
    """

    # predicted annotations in YAML
    llm_output_yaml = llm_output_dict['completion']['choices'][0]['text']
    # input paragraph (used to determine text offsets)
    para = llm_output_dict['paragraph']

    # Check if LLM output is valid YAML
    try:
        llm_output = yaml.load(llm_output_yaml, Loader=yaml.Loader)
    except yaml.YAMLError as e:
        print('Error parsing LLM output YAML:')
        print(e)
        return None

    # Check if LLM output adheres to expected format
    has_entities_key = 'text_contains_entities'
    if type(llm_output) not in [list, dict]:
        # unexpected format
        print(
            f'Expected list/dict as top-level YAML element, '
            f'got {type(llm_output)}'
        )
        print(f'LLM output: {llm_output}')
        sys.exit(1)
    if type(llm_output) == dict:
        # reconstruct list from dict
        llm_output_fixed = []
        for k, v in llm_output.items():
            llm_output_fixed.append({k: v})
        llm_output = llm_output_fixed
    if len(llm_output) == 1:
        # probably expected format (the info that nothing
        # needs to be annotated in the paragraph)
        try:
            # check for expected format
            assert (
                has_entities_key in llm_output[0].keys() and
                type(llm_output[0][has_entities_key]) == bool
            )
        except AssertionError:
            print(
                f'Expected key "{has_entities_key}" in LLM output, '
                f'got {llm_output}'
            )
            sys.exit(1)
    elif len(llm_output) != 2:
        # unexpected format (if it’s not of length 1 or 2, we’re not
        # sure how to deal with it)
        print(
            f'Expected list of length 1 or 2 as top-level YAML element, '
            f'got {llm_output}'
        )
        sys.exit(1)
        # NOTE: might be possible to just get the two dict elements
        #       needed here and ignore the rest
    else:
        # expected format, good to proceed
        pass

    out = empty_para_annotation(
        para['annotator_id'],
        para['document_id'],
        para['paragraph_index']
    )

    if not llm_output[0][has_entities_key]:
        # If there are no entities, return the annotation dict empty
        return out

    entity_key = (
        'entities (datasets, models, methods, loss functions, '
        'regularization techniques)'
    )

    llm_edict = llm_output[1]

    llm_artifact_list = None
    if type(llm_edict) != dict:
        print(
            f'Expected dict as second element of LLM output, '
            f'got {type(llm_edict)}'
        )
        if type(llm_edict) == list:
            print('It is a list. Assuming it is a list of dicts.')
            #      - check that it is a list of dicts
            if not all([type(e) == dict for e in llm_edict]):
                print(
                    f'Not all elements of list are dicts: {llm_edict}'
                )
                sys.exit(1)
            print('assuming it is a list of entity dicts')
            llm_artifact_list = llm_edict
    if len(llm_edict.keys()) == 0:
        print(
            f'No entities even though {has_entities_key} is true. '
        )
        return out
    if len(llm_edict.keys()) > 1:
        print(
            f'More than one entity key in LLM output: {llm_edict.keys()}. '
        )
        if entity_key in llm_edict.keys():
            print(f'Using key "{entity_key}" and ignoring rest')
    if entity_key not in llm_edict.keys():
        print(
            f'Expected key "{entity_key}" in LLM output, '
            f'got {llm_output}. Using first key.'
        )
        llm_artifact_list = llm_edict[list(llm_edict.keys())[0]]
    else:
        llm_artifact_list = llm_edict[entity_key]

    # coarse structure checking done
    # from hereon parse entity/relation dicts and build output
    # compatible with eval script
    if verbose:
        print('coarse structure looks good :)')

    for artf_wrapper in llm_artifact_list:
        # build entity and relation dicts
        # - probably need to find surface foms in text to assign
        #   offsets
        # - might become tricky to deal with “overlapping” parameter
        #   entities mentioned for multiple artifacts

        # unwrap weird YAML->JSON conversion structure
        artf = artf_wrapper[list(artf_wrapper.keys())[0]]

        # create the artifact entity
        artif_name = artf.get('name', None)
        if artif_name is None:
            print(f'No name for artifact: {artf}')
            continue
        # check if (identically named) artifact entity already exists
        if artif_name in out['annotation']['entities']:
            # not sure if this is sensible
            print('Duplicate artifact entity name, reusing existing entity')
            artif_annot = out['annotation']['entities'][artif_name]
        else:
            artif_annot = entity_dict(artif_name, 'a')
        # find surface forms
        artif_surfs = find_surface_forms_in_para(
            para['text'],
            artif_name
        )
        artif_annot['surface_forms'] = artif_surfs
        out['annotation']['entities'][artif_name] = artif_annot

        # check for parameters
        if not artf.get('has_parameters', False):
            # no parameters, just add artifact entity
            out['annotation']['entities'][artif_name] = artif_annot
            continue
        # create parameter entities
        for prm_wrapper in artf['parameters']:
            # unwrap weird YAML->JSON conversion structure
            prm = prm_wrapper[list(prm_wrapper.keys())[0]]
            prm_name = prm.get('name', None)
            if prm_name is None:
                print(f'No name for parameter: {prm}')
                continue
            prm_annot = entity_dict(prm_name, 'p')
            prm_surfs = find_surface_forms_in_para(
                para['text'],
                prm_name
            )
            prm_annot['surface_forms'] = prm_surfs
            # add parameter entity
            out['annotation']['entities'][prm_name] = prm_annot
            # add relation between parameter and artifact
            rel_annot = relation_dict(
                prm_name,
                artif_name,
                prm_surfs,
                artif_surfs
            )
            out['annotation']['relations'][rel_annot["id"]] = rel_annot
            # check for a value
            if prm.get('value', None) is None or prm['value'] == '':
                continue
            # add value entity
            val_name = str(prm['value'])
            val_annot = entity_dict(val_name, 'v')
            val_surfs = find_surface_forms_in_para(
                para['text'],
                val_name
            )
            val_annot['surface_forms'] = val_surfs
            out['annotation']['entities'][val_name] = val_annot
            # add relation between parameter and value
            rel_annot = relation_dict(
                val_name,
                prm_name,
                val_surfs,
                prm_surfs
            )
            out['annotation']['relations'][rel_annot["id"]] = rel_annot
            # check for a context
            if prm.get('context', None) is None or prm['context'] == '':
                continue
            # add context entity
            ctx_name = prm['context']
            ctx_annot = entity_dict(ctx_name, 'c')
            ctx_surfs = find_surface_forms_in_para(
                para['text'],
                ctx_name
            )
            ctx_annot['surface_forms'] = ctx_surfs
            out['annotation']['entities'][ctx_name] = ctx_annot
            # add relation between value and context
            rel_annot = relation_dict(
                ctx_name,
                val_name,
                ctx_surfs,
                val_surfs
            )
            out['annotation']['relations'][rel_annot["id"]] = rel_annot

    if verbose:
        print(f'Found {len(out["annotation"]["entities"])} entities')
        print(f'Found {len(out["annotation"]["relations"])} relations')

    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python llm_eval.py <llm_output>')
        sys.exit(1)

    fn_in = sys.argv[1]

    with open(sys.argv[1], 'r') as f:
        llm_output_dict = json.load(f)

    llm_out_conv = yaml2json(llm_output_dict, verbose=True)

    fn_out = fn_in.replace('.json', '_conv.json')

    print(f'Writing converted output to {fn_out}')
    with open(fn_out, 'w') as f:
        json.dump(llm_out_conv, f, indent=2)
