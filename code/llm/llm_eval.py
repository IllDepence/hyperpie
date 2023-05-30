""" Deal with LLM output and make it compatible with the evaluation script.
"""

import re
import sys
import json
import uuid
import yaml
from collections import OrderedDict


# LLM prompt output format (YAML)
#
# ---
# - text_contains_entities: true/false
# - entities (datasets, models, methods, loss functions, regularization techniques):  # noqa: E501
#     - entity<N>:
#         name: <entity name>
#         type: <entity type>
#         has_parameters: true/false
#         parameters:
#             - parameter<N>:
#                 name: <parameter name>
#                 value: <parameter value>/null
#                 context: <value context>/null
# ...

# Evaluation script input format (JSON)
#
# list of dicts w/
# "annotation": {
#     "entities": {
#         "aN": {
#             "id": <entity ID>,
#             "surface_forms": [
#                 {
#                     "id": <surface form ID>,
#                     "surface_form": <entity name>,
#                     "start": <offset start>,
#                     "end": <offset end>
#                 },
#                 ...
#             ]
#         },
#         ...
#     }
#     "relations": {
#         "rN": {
#             "source": <source entity ID>,
#             "target": <target entity ID>,
#          },
#          ...
#     }
# }


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


def empty_para_annotation():
    """ Create an empty annotation dict.
    """

    empty_para_annot = OrderedDict({
      "annotator_id": "",
      "document_id": "",
      "paragraph_index": -1,
      "text": "",
      "annotation":
      {
        "entities": {},
        "relations": {}
      },
      "annotation_raw": []
    })

    return empty_para_annot


def find_surface_forms_in_para(para, e_name):
    """ Find all surface forms of an entity in a paragraph.

    Args:
        para (str): Paragraph text.
        e_name (str): Entity name.

    Returns:
        list: List of dicts with surface form information.
    """

    # use regex to identify the start and end offset of
    # all occurrences of entity name in paragraph

    surfs = []
    for m in re.finditer(e_name, para):
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


def convert(para, llm_output_yaml):
    """ Convert LLM output to evaluation script input format.

    Args:
        llm_output (str): LLM output in YAML format.

    Returns:
        str: Evaluation script input in JSON format.
    """

    # Check if LLM output is valid YAML
    try:
        llm_output = yaml.load(llm_output_yaml, Loader=yaml.Loader)
    except yaml.YAMLError as e:
        print('Error parsing LLM output YAML:')
        print(e)
        return None

    # Check if LLM output adheres to expected format
    has_entities_key = 'text_contains_entities'
    if type(llm_output) != list:
        # unexpected format
        print(
            f'Expected list as top-level YAML element, '
            f'got {type(llm_output)}'
        )
        sys.exit(1)
    elif len(llm_output) == 1:
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

    out = empty_para_annotation()

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
        assert artif_name not in out['annotation']['entities'].keys()
        # find surface forms
        artif_annot = entity_dict(artif_name, 'a')
        artif_surfs = find_surface_forms_in_para(
            para,
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
                para,
                prm_name
            )
            prm_annot['surface_forms'] = prm_surfs
            # add parameter entity
            out['annotation']['entities'][prm_name] = prm_annot
            # add relation between parameter and artifact
            rel_annot = relation_dict(
                artif_name,
                prm_name,
                artif_surfs,
                prm_surfs
            )
            out['annotation']['relations'][rel_annot["id"]] = rel_annot
            import pprint
            pprint.pprint(out)
            sys.exit(0)
        # for param in artf['parameters']:
        #     param_name = param.get('name', None)
        #     if param_name is None:
        #         print(f'No name for parameter: {param}')
        #         continue
        #     assert param_name not in out['annotation']['entities'].keys()
        #     param_annot = entity_dict(param_name, 'p')
        #     param_surfs = find_surface_forms_in_para(
        #         para,
        #         param_name
        #     )
        #     param_annot['surface_forms'] = param_surfs



        # TODO continue here
        # 2. create entities for all parameters
        # 3. create entities for all values

    # example
    # [{'entity1': {'name': 'SciBERT-base',
    #    'type': 'model',
    #    'has_parameters': False,
    #    'parameters': None}},
    #  {'entity2': {'name': 'BiLSTM',
    #    'type': 'model',
    #    'has_parameters': True,
    #    'parameters': [{'parameter1': {'name': 'hidden state',
    #       'value': '128-d',
    #       'context': None}}]}},


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python llm_eval.py <llm_output>')
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        llm_output_dict = json.load(f)

    llm_output = llm_output_dict['completion']['choices'][0]['text']

    convert(llm_output_dict['paragraph'], llm_output)
